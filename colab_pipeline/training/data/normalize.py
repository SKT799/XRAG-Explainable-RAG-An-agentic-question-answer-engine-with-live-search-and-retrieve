"""
Normalize the citation-grounded SFT sources into one unified schema.

Active sources (their gold answers carry inline [n] citations, which is what we
train the generator to produce):
  * HAGRID       (miracl/hagrid)      - attributable cited answers
  * WebGLM-QA    (THUDM/webglm-qa)    - ~43k long-form web-cited answers
Skipped (gold answers have NO inline [n]; see notes on _load_alce/_load_expertqa).

Unified SFT schema:

    {
      "query":  str,
      "docs":   [{"id": int, "text": str, "url": str, "title": str}, ...],
      "answer": str    # contains inline [n] citations (1-indexed into docs)
    }

Output is JSONL  -  one row per line  -  written to `data/train/sft.jsonl` (and
`data/test/sft.jsonl` if a test split is provided).

The actual dataset names on the HuggingFace Hub change occasionally; we keep
the loader resilient by trying a list of known IDs and falling back to a tiny
synthetic seed so the file is at least non-empty.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Iterable

from app.util import get_logger

# Refuse to fall through to the synthetic seed when the user clearly wanted
# real data. Override with `--allow-seed-only` on the CLI for unit tests.
_MIN_REAL_ROWS_DEFAULT = 50

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Robust dataset loader (checks local disk path first, falls back to HF)
# ---------------------------------------------------------------------------
def load_dataset_robust(name: str, config: str | None = None, **kwargs):
    import os
    from datasets import load_from_disk, load_dataset
    
    split = kwargs.pop("split", None)
    
    safe_name = name.replace("/", "_")
    folder_name = f"{safe_name}_{config}" if config else safe_name
    
    possible_paths = [
        os.path.join("datasets", folder_name),
        os.path.join("..", "datasets", folder_name),
        os.path.join("..", "..", "datasets", folder_name),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            log.info("Loading dataset %s (%s) from local folder: %s", name, config or "-", path)
            ds = load_from_disk(path)
            if split:
                if isinstance(ds, dict) or hasattr(ds, "keys"):
                    return ds[split]
            return ds
            
    log.info("Local folder not found for %s (%s), downloading from Hugging Face…", name, config or "-")
    if split:
        kwargs["split"] = split
    return load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)


def _load_alce(local_dir: str | None = None) -> Iterable[dict]:
    """ALCE: ASQA / QAMPARI / ELI5 subsets.

    ALCE is NOT a tabular HuggingFace dataset - the data lives as JSON files
    that the upstream repo's `download_data.sh` puts under `./data/`. We
    therefore look for the canonical files (e.g. `asqa_eval_gtr_top100.json`)
    under `local_dir` (defaults to env `ALCE_DATA_DIR` or `./data/alce/`).
    If the files are missing we log and return - we do NOT silently call
    `load_dataset("princeton-nlp/ALCE", ...)` because that path does not
    exist on the Hub and the failure used to fall through to the synthetic
    seed without anyone noticing.
    """
    # NOTE: not used by default. ALCE is an EVAL benchmark - its gold answers
    # (asqa long_answer / qampari answer / eli5 answer) contain NO inline [n]
    # citation markers (citing is the model's job, scored separately against the
    # provided docs). Training our citation-grounded SFT on uncited answers would
    # teach the generator to stop citing, so ALCE is intentionally left out of
    # the default mix. The parser below still works if you point it at
    # citation-bearing files (e.g. your own reconstruction).
    base = Path(local_dir or os.environ.get("ALCE_DATA_DIR") or "./data/alce")
    if not base.exists():
        log.warning("ALCE data dir %s missing - skipping (see note in _load_alce: "
                    "ALCE gold answers have no inline [n], so it is not in the "
                    "default SFT mix).", base)
        return []
    # ALCE files: asqa_eval_*.json, qampari_eval_*.json, eli5_eval_*.json.
    # Each row has: question, answer (str OR list[str]), docs[{title,text,url,...}].
    n = 0
    for fp in sorted(base.glob("*_eval_*.json")):
        try:
            with fp.open("r", encoding="utf-8") as f:
                blob = json.load(f)
        except Exception as e:
            log.warning("ALCE: failed to read %s (%s)", fp, e); continue
        rows = blob if isinstance(blob, list) else blob.get("data", [])
        for row in rows:
            docs_raw = row.get("docs", [])[:10]
            docs = [{"id": i + 1,
                     "text": (d.get("text") or d.get("snippet") or "")[:1200],
                     "url": d.get("url", ""),
                     "title": d.get("title", "")}
                    for i, d in enumerate(docs_raw)]
            # ALCE answer extraction. ASQA stores the gold reference under
            # `annotations[*].long_answer` (plural, list). QAMPARI uses
            # `answers` (list of short strings). ELI5 uses a top-level
            # `answer`. Try all three in order; first non-empty wins.
            ans: str = ""
            anns = row.get("annotations")
            if isinstance(anns, list):
                for a in anns:
                    if isinstance(a, dict) and a.get("long_answer"):
                        ans = a["long_answer"]; break
            if not ans:
                top = row.get("answer") or row.get("answers")
                if isinstance(top, list):
                    ans = top[0] if top else ""
                elif isinstance(top, dict):
                    ans = top.get("long_answer") or top.get("answer") or ""
                elif isinstance(top, str):
                    ans = top
            if not (docs and ans):
                continue
            yield {"query": row.get("question", ""), "docs": docs, "answer": ans}
            n += 1
    log.info("ALCE: loaded %d rows from %s", n, base)


def _load_hagrid() -> Iterable[dict]:
    """HAGRID  -  purpose-built for citation-grounded RAG.

    Schema (verified against miracl/hagrid on HF):
      query: str
      quotes: [{docid: str, idx: int, text: str}, ...]
      answers: [{answer: str, attributable: 0|1|None,
                 informative: 0|1, sentences: [...]}, ...]

    The `answers` field is a list of OBJECTS, not strings. We pick the
    first answer that is BOTH attributable AND informative; if none, we
    drop the row (training on un-attributable answers is exactly what we
    are trying NOT to teach the model).
    """
    try:
        ds = load_dataset_robust("miracl/hagrid", split="train", trust_remote_code=True)
    except Exception as e:
        log.warning("HAGRID not available (%s)  -  skipping", e)
        return []
    n_kept, n_dropped = 0, 0
    for row in ds:
        quotes = row.get("quotes") or []
        docs = [{"id": i + 1,
                 "text": (q.get("text") or "")[:1200],
                 "url": "",                         # HAGRID quotes have no URL
                 "title": ""}                       # nor a title
                for i, q in enumerate(quotes)]
        ans_text = ""
        for a in (row.get("answers") or []):
            if not isinstance(a, dict):
                continue
            # attributable can be None on un-labelled answers; require == 1.
            if a.get("attributable") == 1 and a.get("informative") == 1:
                ans_text = (a.get("answer") or "").strip()
                if ans_text:
                    break
        if docs and ans_text:
            yield {"query": row.get("query", ""), "docs": docs, "answer": ans_text}
            n_kept += 1
        else:
            n_dropped += 1
    log.info("HAGRID: kept %d / dropped %d (un-attributable or empty)",
             n_kept, n_dropped)


def _load_expertqa() -> Iterable[dict]:
    """ExpertQA: real expert-written queries with cited answers.

    NOTE: not used by default. The citation-bearing `main` config fails to
    generate on recent `datasets` (nested-schema cast error), and the clean
    `lfqa_*` configs drop the evidence entirely (empty `context`, no inline
    `[n]`). Like ALCE, ExpertQA gold answers carry no inline `[n]` markers, so
    training on them would teach the generator to STOP citing - the opposite of
    the objective. Left here for anyone who reconstructs citations from the
    per-claim evidence; it returns [] gracefully otherwise.
    """
    try:
        ds = load_dataset_robust("cmalaviya/expertqa", "main", split="train", trust_remote_code=True)
    except Exception as e:
        log.warning("ExpertQA not available (%s)  -  skipping", e)
        return []
    for row in ds:
        docs = [{"id": i + 1, "text": s.get("snippet", "")[:1200],
                 "url": s.get("url", ""), "title": s.get("title", "")}
                for i, s in enumerate(row.get("sources", []))]
        ans = row.get("answer", "")
        if docs and ans:
            yield {"query": row.get("question", ""), "docs": docs, "answer": ans}


def _load_webglm() -> Iterable[dict]:
    """WebGLM-QA (THUDM/webglm-qa): ~43k long-form, web-cited answers.

    This is the IDEAL second SFT source: every answer already carries inline
    `[n]` citation markers that 1-index into its `references` list - exactly our
    schema {query, docs[{id,text,...}], answer-with-[n]}. Adding it (alongside
    HAGRID) multiplies the SFT data and teaches a distinctive, thoroughly-cited
    long-form style, so the raw-vs-tuned difference is actually VISIBLE.

    Capped by env `XRAG_WEBGLM_MAX` (default 2000) to keep Colab SFT time sane;
    set it to 0 for all ~43k rows.
    """
    cap = int(os.environ.get("XRAG_WEBGLM_MAX", "2000") or "0")
    try:
        ds = load_dataset_robust("THUDM/webglm-qa", split="train", trust_remote_code=True)
    except Exception as e:
        log.warning("WebGLM-QA not available (%s)  -  skipping", e)
        return []
    cite = re.compile(r"\[(\d+)\]")
    rows = ds.select(range(min(cap, len(ds)))) if cap else ds
    n_kept = n_dropped = 0
    for row in rows:
        q = (row.get("question") or "").strip()
        refs = row.get("references") or []
        ans = (row.get("answer") or "").strip()
        if not (q and refs and ans):
            n_dropped += 1; continue
        # Keep only answers that actually cite, and whose [n] are in range
        # (drop the rare out-of-range citation so we never train a [k] that
        # points past the doc list).
        cited = [int(m.group(1)) for m in cite.finditer(ans)]
        if not cited or max(cited) > len(refs):
            n_dropped += 1; continue
        docs = [{"id": i + 1, "text": (t or "")[:1200], "url": "", "title": ""}
                for i, t in enumerate(refs)]
        yield {"query": q, "docs": docs, "answer": ans}
        n_kept += 1
    log.info("WebGLM-QA: kept %d / dropped %d (cap=%s)",
             n_kept, n_dropped, cap or "all")


# ---------------------------------------------------------------------------
# Tiny seed so the pipeline works even with zero internet
# ---------------------------------------------------------------------------
_SEED = [
    {
        "query": "Who won the 2022 FIFA World Cup and who was top scorer?",
        "docs": [
            {"id": 1, "text": "Argentina won the 2022 FIFA World Cup, beating France 4-2 on penalties after a 3-3 draw.",
             "url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup", "title": "2022 FIFA World Cup"},
            {"id": 2, "text": "Kylian Mbappé won the Golden Boot at the 2022 World Cup with 8 goals.",
             "url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup", "title": "2022 FIFA World Cup"},
        ],
        "answer": "Argentina won the 2022 FIFA World Cup [1]. Kylian Mbappé was the top scorer with 8 goals, winning the Golden Boot [2].",
    },
    {
        "query": "Where is the Eiffel Tower and how tall is it?",
        "docs": [
            {"id": 1, "text": "The Eiffel Tower is in Paris, France.",
             "url": "https://en.wikipedia.org/wiki/Eiffel_Tower", "title": "Eiffel Tower"},
            {"id": 2, "text": "The Eiffel Tower stands 330 metres tall including antennas.",
             "url": "https://en.wikipedia.org/wiki/Eiffel_Tower", "title": "Eiffel Tower"},
        ],
        "answer": "The Eiffel Tower is in Paris, France [1], and it stands 330 metres tall [2].",
    },
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def normalize_all(out_train: str = "data/train/sft.jsonl",
                  out_test: str = "data/test/sft.jsonl",
                  test_frac: float = 0.05,
                  seed: int = 42,
                  max_rows: int | None = None,
                  allow_seed_only: bool = False,
                  min_real_rows: int = _MIN_REAL_ROWS_DEFAULT,
                  max_test: int = 20,
                  ) -> tuple[int, int]:
    """Normalize all sources into a single SFT JSONL, optionally limited to max_rows.

    By default, if NO real dataset loads we REFUSE to silently fall through to
    the 2-row synthetic seed (which produces a useless adapter). Pass
    `allow_seed_only=True` (CLI flag `--allow-seed-only`) when running unit
    tests or smoke checks where seed-only is what you actually want.
    """
    rows: list[dict] = []
    for src, loader in [("alce", _load_alce), ("hagrid", _load_hagrid),
                        ("webglm", _load_webglm), ("expertqa", _load_expertqa)]:
        n = 0
        for r in loader():
            rows.append(r); n += 1
            if max_rows and len(rows) >= max_rows:
                break
        log.info("%s: %d rows", src, n)
        if max_rows and len(rows) >= max_rows:
            break

    if len(rows) < min_real_rows:
        msg = (f"only {len(rows)} real rows collected (<{min_real_rows}). "
               "Check dataset paths/auth; pass --allow-seed-only to write the "
               "synthetic seed anyway.")
        if not allow_seed_only:
            raise RuntimeError(msg)
        log.warning("%s - writing the synthetic seed (%d rows)", msg, len(_SEED))
        rows = list(_SEED)

    rnd = random.Random(seed)
    rnd.shuffle(rows)
    # Cap the test split so the (expensive, live-web) raw-vs-tuned eval stays fast
    # even when the training set is large - the eval still covers the WHOLE test
    # file, it's just bounded to ~max_test held-out queries.
    n_test = min(max(1, int(len(rows) * test_frac)), max_test)
    train, test = rows[n_test:], rows[:n_test]

    for path, data in [(out_train, train), (out_test, test)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("wrote %s (%d rows)", path, len(data))
    return len(train), len(test)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_train", default="data/train/sft.jsonl")
    ap.add_argument("--out_test", default="data/test/sft.jsonl")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--webglm_max", type=int, default=None,
                    help="Max rows to pull from WebGLM-QA (default 2000; set 0 for "
                         "all ~43k). More WebGLM rows = bigger, more visible "
                         "behavioural change, but longer SFT.")
    ap.add_argument("--allow-seed-only", action="store_true",
                    help="write the synthetic seed when no real dataset loaded")
    ap.add_argument("--min-real-rows", type=int, default=_MIN_REAL_ROWS_DEFAULT)
    ap.add_argument("--max_test", type=int, default=20,
                    help="Cap the held-out test split (keeps the live-web "
                         "raw-vs-tuned eval fast). Default 20.")
    a = ap.parse_args()
    if a.webglm_max is not None:
        os.environ["XRAG_WEBGLM_MAX"] = str(a.webglm_max)
    normalize_all(out_train=a.out_train, out_test=a.out_test,
                  max_rows=a.max_rows, allow_seed_only=a.allow_seed_only,
                  min_real_rows=a.min_real_rows, max_test=a.max_test)
