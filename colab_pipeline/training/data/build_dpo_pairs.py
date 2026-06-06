"""
Build (prompt, chosen, rejected) preference pairs for DPO from the SFT JSONL.

Strategy:
  * `chosen`   = the gold cited answer.
  * `rejected` = a *corrupted* variant, chosen at random per row:
      - swap a citation index to a wrong doc id
      - drop a citation entirely
      - replace a key noun with a generic one (hallucination)

This is the FROM-SCRATCH preference data; we'll later add a "model-sampled
rejected with NLI auto-labeling" enhancement once Block 10 is calibrated.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from app.util import get_logger

log = get_logger(__name__)

_CITE_RE = re.compile(r"\[(\d+)\]")


def _swap_cite(answer: str, n_docs: int) -> str:
    cites = list(_CITE_RE.finditer(answer))
    if not cites:
        return answer
    m = random.choice(cites)
    wrong = random.choice([i for i in range(1, n_docs + 1) if i != int(m.group(1))] or [1])
    return answer[:m.start()] + f"[{wrong}]" + answer[m.end():]


def _drop_cite(answer: str) -> str:
    return _CITE_RE.sub("", answer, count=1)


_SENT_START = re.compile(r"(?:^|[.!?]\s+)([A-Z][a-zA-Z]+)")


def _content_nouns(answer: str) -> list[str]:
    """Capitalized words that DON'T just appear at sentence starts.

    Sentence-start words ("When", "While", "Since", "Because") get
    capitalized by grammar, not because they're proper nouns. Swapping a
    real entity name with one of those produces ungrammatical garbage
    that DPO learns to dislike for trivially shallow reasons (style,
    not faithfulness). Excluding them forces the corruption to swap
    actual entities, which is what we want the model to discriminate.
    """
    sent_starts = set(m.group(1) for m in _SENT_START.finditer(answer))
    all_caps = set(re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", answer))
    # A word counts as a content noun if it appears AT LEAST ONCE not at a
    # sentence start (so "Argentina" inside a sentence still qualifies even
    # if it also opens one).
    pure_starts = set()
    for w in all_caps:
        # All occurrences of w
        positions = [m.start() for m in re.finditer(rf"\b{re.escape(w)}\b", answer)]
        only_start = all(
            (p == 0) or (answer[max(0, p - 2):p].rstrip().endswith((".", "!", "?")))
            for p in positions
        )
        if only_start:
            pure_starts.add(w)
    return list(all_caps - pure_starts)


def _swap_subject(answer: str) -> str:
    """Replace a real content noun with another real content noun."""
    nouns = _content_nouns(answer)
    if len(nouns) < 2:
        return answer
    a, b = random.sample(nouns, 2)
    return re.sub(rf"\b{re.escape(a)}\b", b, answer, count=1)


def make_pair(row: dict, tokenizer=None) -> dict:
    """Turn one SFT row into one DPO row.

    `chosen` / `rejected` are the BARE assistant text (no headers, no
    system rule); `prompt` is the chat-template-formatted prompt up to
    and including the assistant header (i.e. add_generation_prompt=True).
    This is what TRL DPOTrainer expects.

    If `tokenizer` is None we fall back to the legacy plain-string prompt
    (for unit tests / smoke runs that don't want to spin up a tokenizer).
    """
    chosen = row["answer"]
    n = len(row["docs"])
    corrupters = [lambda a: _swap_cite(a, n), _drop_cite, _swap_subject]
    rejected = random.choice(corrupters)(chosen)
    if rejected == chosen:                       # ensure it's actually different
        rejected = _drop_cite(chosen) or (chosen + " (no citation)")
    if tokenizer is not None:
        prompt = _format_prompt_chat(row["query"], row["docs"], tokenizer)
    else:
        prompt = _format_prompt_legacy(row["query"], row["docs"])
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _format_prompt_chat(query: str, docs: list[dict], tokenizer) -> str:
    """Chat-template DPO prompt (matches generator.build_messages)."""
    from app.generation.generator import SYSTEM_RULES
    sources = "\n\n".join(
        f"[{d['id']}] ({d.get('title','')})\n{d['text'][:1200]}" for d in docs
    )
    messages = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user",   "content": f"SOURCES:\n{sources}\n\nQUESTION:\n{query}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _format_prompt_legacy(query: str, docs: list[dict]) -> str:
    """Legacy plain-string DPO prompt (no tokenizer required - used by tests)."""
    parts = []
    for d in docs:
        head = f"[{d['id']}] ({d.get('title','')})"
        parts.append(f"{head}\n{d['text'][:1200]}")
    return ("SYSTEM: Answer using ONLY the numbered sources. End every factual "
            "sentence with [n] citations.\n\nSOURCES:\n" + "\n\n".join(parts)
            + f"\n\nQUESTION:\n{query}\n\nANSWER:\n")


def _nli_margin_ok(row: dict, chosen: str, rejected: str,
                   nli, margin: float) -> bool:
    """True if `p_entail(best_doc, chosen) - p_entail(best_doc, rejected) >= margin`.

    The "best doc" for each candidate is just the one that supports it most
    among `row["docs"]`. We use the calibrated NLI scorer the runtime uses
    (after Stage 6.5) so the margin is in the same units as the trust score.
    """
    if nli is None:
        return True

    def best_p(hypothesis: str) -> float:
        best = 0.0
        for d in row["docs"]:
            p = nli.p_entail(premise=d["text"][:1200], hypothesis=hypothesis)
            if p > best:
                best = p
        return best
    return (best_p(chosen) - best_p(rejected)) >= margin


def build(in_path: str = "data/train/sft.jsonl",
          out_path: str = "data/train/dpo.jsonl",
          seed: int = 42, max_rows: int | None = None,
          tokenizer_id: str | None = None,
          nli_margin: float = 0.0) -> int:
    """Build the DPO JSONL. If `tokenizer_id` is set we materialize the
    prompts in chat-template form (matches generator.build_messages); if
    None we emit the legacy plain-string prompts.

    If `nli_margin > 0`, load the calibrated NLI scorer and DROP any pair
    where the chosen answer isn't at least `margin` more entailed than the
    rejected one. This catches degenerate corruptions where (e.g.) dropping
    a redundant citation didn't actually make the answer worse.
    """
    random.seed(seed)
    tokenizer = None
    if tokenizer_id:
        from transformers import AutoTokenizer
        log.info("loading tokenizer %s for chat-template DPO prompts…", tokenizer_id)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    nli = None
    if nli_margin > 0.0:
        from app.attribution.scorer import NLIScorer
        log.info("loading NLI scorer for margin filter (margin=%.2f)…", nli_margin)
        nli = NLIScorer.get()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n, n_filtered = 0, 0
    with open(in_path, "r", encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            try:
                pair = make_pair(row, tokenizer=tokenizer)
            except Exception as e:
                log.warning("skip row (%s)", e); continue
            if not _nli_margin_ok(row, pair["chosen"], pair["rejected"], nli, nli_margin):
                n_filtered += 1
                continue
            dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n += 1
            if max_rows and n >= max_rows:
                break
    if n_filtered:
        log.info("filtered %d weak pairs (NLI margin < %.2f)", n_filtered, nli_margin)
    log.info("wrote %s (%d pairs)", out_path, n)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/train/sft.jsonl")
    ap.add_argument("--out_path", default="data/train/dpo.jsonl")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--tokenizer_id", default=None,
                    help="If set (e.g. NousResearch/Meta-Llama-3.1-8B-Instruct), "
                         "DPO prompts use the model's chat template. Recommended "
                         "for production runs - matches generator.build_messages.")
    ap.add_argument("--nli_margin", type=float, default=0.0,
                    help="Drop pairs where the NLI margin between chosen and "
                         "rejected is < this value. 0 = keep all. Recommend "
                         "0.15-0.25 after Stage 6.5 NLI head is trained.")
    args = ap.parse_args()
    build(args.in_path, args.out_path, max_rows=args.max_rows,
          tokenizer_id=args.tokenizer_id, nli_margin=args.nli_margin)
