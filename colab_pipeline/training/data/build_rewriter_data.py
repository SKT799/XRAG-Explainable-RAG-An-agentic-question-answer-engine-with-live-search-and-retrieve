"""
Build the rewriter training set (`data/train/rewriter.jsonl`) from the SFT data.

Stage 6.4 (`training/train_rewriter.py`) expects rows shaped like:

    {"query": "...", "history": ["...", "..."], "gold_json": "<JSON answer string>"}

Ideally that gold JSON comes from real conversational-rewrite corpora (QReCC /
TREC CAsT / MS MARCO Conversational). Those aren't bundled here, and without
this file the trainer silently falls back to a 2-row synthetic smoke test, which
produces a useless adapter.

So we bootstrap gold labels from the SFT queries (HAGRID/ALCE/ExpertQA), which
are already well-formed standalone questions:

  * standalone_query = the query, lightly cleaned
  * sub_queries      = the deterministic `HeuristicRewriter` split (splits
                       multi-part "X and Y" questions, passes single-part ones
                       through unchanged)
  * intent           = "factual_lookup"

These are weak (heuristic) labels: they teach the 3B model the "clean + split
multi-part" behavior on real volume, not pronoun resolution. We therefore ALSO
prepend a couple of history-based synthetic rows so the adapter still sees the
pronoun-resolution case. For production-grade rewriting, replace this with a
real conversational-rewrite dataset.

Usage:
    python -m training.data.build_rewriter_data \
        --in_path  data/train/sft.jsonl \
        --out_path data/train/rewriter.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.planning.rewriter import HeuristicRewriter
from app.util import get_logger

log = get_logger(__name__)


# History-based examples the SFT-derived rows can't cover (no conversation
# history in HAGRID/ALCE/ExpertQA). Kept tiny - just enough to show the
# pronoun-resolution behavior at training time.
_HISTORY_SEED = [
    {"query": "how tall is it?",
     "history": ["Tell me about the Eiffel Tower."],
     "gold_json": json.dumps({
         "standalone_query": "Eiffel Tower height",
         "sub_queries": ["Eiffel Tower height in metres"],
         "intent": "factual_lookup"})},
    {"query": "and who directed it?",
     "history": ["What year did the movie Inception come out?",
                 "Inception was released in 2010."],
     "gold_json": json.dumps({
         "standalone_query": "Who directed the movie Inception?",
         "sub_queries": ["Inception movie director"],
         "intent": "factual_lookup"})},
]


def build(in_path: str = "data/train/sft.jsonl",
          out_path: str = "data/train/rewriter.jsonl",
          max_rows: int | None = None) -> int:
    """Write rewriter rows derived from the SFT queries. Returns row count."""
    src = Path(in_path)
    if not src.exists():
        raise SystemExit(
            f"{in_path} not found. Run `python -m training.data.normalize` first "
            "to build the SFT data this bootstraps from."
        )

    hr = HeuristicRewriter()
    rows: list[dict] = list(_HISTORY_SEED)
    seen: set[str] = set()
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                sft = json.loads(line)
            except Exception:
                continue
            query = (sft.get("query") or "").strip()
            if not query or query in seen:
                continue
            seen.add(query)
            rw = hr.rewrite(query)
            gold = json.dumps({
                "standalone_query": rw.standalone_query,
                "sub_queries": rw.sub_queries,
                "intent": rw.intent,
            })
            rows.append({"query": query, "history": [], "gold_json": gold})
            if max_rows and len(rows) >= max_rows:
                break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("wrote %s (%d rows: %d history-seed + %d from %s)",
             out_path, len(rows), len(_HISTORY_SEED), len(rows) - len(_HISTORY_SEED),
             in_path)
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/train/sft.jsonl")
    ap.add_argument("--out_path", default="data/train/rewriter.jsonl")
    ap.add_argument("--max_rows", type=int, default=None)
    a = ap.parse_args()
    build(a.in_path, a.out_path, max_rows=a.max_rows)
