"""
Evaluation harness · master plan §9.

Two modes:
  * single  -  run the (tuned) pipeline on every query and report citation
               precision / recall / F1, trust, latency.
  * dual    -  run BOTH the raw base generator AND the fine-tuned generator on
               every query (same retrieval, same NLI scorer) and report a
               side-by-side raw-vs-tuned comparison with deltas. This is the
               "did fine-tuning actually help?" report.

By default it evaluates the ENTIRE test set (`data/test/sft.jsonl`) - every row,
not a sample.

Usage:
  python -m eval.harness --dual --queries data/test/sft.jsonl --out docs/eval_results.md
  python -m eval.harness            # single-pipeline eval on the full test set
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.schemas import AnswerResponse, QueryRequest
from app.util import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-response metrics
# ---------------------------------------------------------------------------
def citation_precision_recall(resp: AnswerResponse) -> tuple[float, float, float]:
    """
    Approximation following the ALCE definition:
      precision = #green citations / #citations
      recall    = #green citations / #factual_sentences_with_at_least_one_citation
    """
    if not resp.citations:
        return 0.0, 0.0, 0.0
    green = sum(1 for c in resp.citations if c.flag == "green")
    precision = green / len(resp.citations)
    # We don't have per-claim records in AnswerResponse  -  use overall_trust as recall proxy.
    recall = resp.overall_trust
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _metrics_row(resp: AnswerResponse) -> dict:
    """One response -> a flat dict of its metrics (used by both eval modes)."""
    p, r, f = citation_precision_recall(resp)
    return {"p": p, "r": r, "f": f, "trust": resp.overall_trust,
            "lat": resp.latency_ms}


def _aggregate(rows: list[dict]) -> dict:
    """Mean each metric across responses (+ latency percentiles)."""
    return {
        "cite_precision": _mean([x["p"] for x in rows]),
        "cite_recall": _mean([x["r"] for x in rows]),
        "cite_f1": _mean([x["f"] for x in rows]),
        "overall_trust": _mean([x["trust"] for x in rows]),
        "latency_p50_ms": _percentile([x["lat"] for x in rows], 50),
        "latency_p95_ms": _percentile([x["lat"] for x in rows], 95),
    }


# ---------------------------------------------------------------------------
# Query loading - the ENTIRE test set by default
# ---------------------------------------------------------------------------
def load_eval_queries(path: str = "data/test/sft.jsonl",
                      max_queries: int | None = None) -> list[str]:
    """Return the de-duplicated list of `query` strings from a JSONL file.

    `data/test/sft.jsonl` rows are `{query, docs, answer}` - we just take the
    query. If the file is missing we fall back to a tiny built-in set so the
    harness still runs. `max_queries` caps the count (None = all = the whole
    test set, which is the default).
    """
    p = Path(path)
    if not p.exists():
        log.warning("%s missing  -  using the built-in %d-query set",
                    path, len(_BUILTIN_QUERIES))
        raw = [r["query"] for r in _BUILTIN_QUERIES]
    else:
        raw = []
        for line in p.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                q = (json.loads(line).get("query") or "").strip()
            except Exception:
                continue
            if q:
                raw.append(q)
    seen, out = set(), []
    for q in raw:                                   # de-dup, preserve order
        if q not in seen:
            seen.add(q); out.append(q)
    if max_queries:
        out = out[:max_queries]
    log.info("loaded %d eval queries from %s", len(out), path)
    return out


# ---------------------------------------------------------------------------
# Single-pipeline eval (tuned only)
# ---------------------------------------------------------------------------
def evaluate(queries: list[str], out_md: str = "docs/eval_results.md",
             source: str = "the test set") -> dict:
    from app.orchestrator.engine import run
    log.info("evaluating %d queries (tuned pipeline)…", len(queries))
    P, R, F, T, L, per_query = [], [], [], [], [], []
    for q in queries:
        try:
            resp = run(QueryRequest(query=q))
        except Exception as e:
            log.warning("eval skip %r (%s)", q, e); continue
        m = _metrics_row(resp)
        P.append(m["p"]); R.append(m["r"]); F.append(m["f"])
        T.append(m["trust"]); L.append(m["lat"])
        per_query.append({"q": q, "trust": resp.overall_trust,
                          "p": m["p"], "r": m["r"], "f1": m["f"], "lat_ms": resp.latency_ms})

    summary = {
        "n": len(per_query), "source": source,
        "cite_precision": _mean(P), "cite_recall": _mean(R), "cite_f1": _mean(F),
        "overall_trust": _mean(T),
        "latency_p50_ms": _percentile(L, 50), "latency_p95_ms": _percentile(L, 95),
    }
    _write_report(out_md, summary, per_query)
    log.info("summary: %s", summary); log.info("wrote %s", out_md)
    return summary


# ---------------------------------------------------------------------------
# Dual eval (raw base vs fine-tuned) - the final comparison
# ---------------------------------------------------------------------------
def evaluate_dual(queries: list[str], out_md: str = "docs/eval_results.md",
                  source: str = "data/test/sft.jsonl") -> dict:
    """Run raw base AND tuned on every query and write a comparison report."""
    from app.orchestrator.dual import run_dual
    log.info("dual-evaluating %d queries (raw base vs tuned)…", len(queries))
    raw_rows, tuned_rows, per_q = [], [], []
    for i, q in enumerate(queries, 1):
        try:
            d = run_dual(QueryRequest(query=q))
        except Exception as e:
            log.warning("eval skip [%d/%d] %r (%s)", i, len(queries), q, e); continue
        rm, tm = _metrics_row(d.raw), _metrics_row(d.tuned)
        raw_rows.append(rm); tuned_rows.append(tm)
        per_q.append({"q": q,
                      "raw_trust": d.raw.overall_trust, "tuned_trust": d.tuned.overall_trust})
        log.info("[%d/%d] %.40s  raw=%.2f tuned=%.2f",
                 i, len(queries), q, d.raw.overall_trust, d.tuned.overall_trust)

    if not per_q:
        log.error("no queries evaluated  -  nothing to compare"); return {}
    summary = {"n": len(per_q), "source": source,
               "raw": _aggregate(raw_rows), "tuned": _aggregate(tuned_rows)}
    _write_dual_report(out_md, summary, per_q)
    log.info("RAW   : %s", summary["raw"])
    log.info("TUNED : %s", summary["tuned"])
    log.info("wrote %s", out_md)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mean(xs):
    return round(sum(xs) / max(len(xs), 1), 4)

def _percentile(xs, p):
    if not xs: return 0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def _write_report(path: str, summary: dict, per_q: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# X-RAG · evaluation results", "",
             f"Tuned pipeline on **{summary['n']}** queries from "
             f"`{summary['source']}`.", "", "## Summary", ""]
    for k, v in summary.items():
        if k == "source":
            continue
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Per-query\n")
    lines.append("| query | trust | precision | recall | F1 | lat (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in per_q:
        lines.append(f"| {r['q'][:80]} | {r['trust']:.2f} | "
                     f"{r['p']:.2f} | {r['r']:.2f} | {r['f1']:.2f} | {r['lat_ms']} |")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_dual_report(path: str, summary: dict, per_q: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    R, T = summary["raw"], summary["tuned"]
    d_prec = round(T["cite_precision"] - R["cite_precision"], 4)
    d_f1 = round(T["cite_f1"] - R["cite_f1"], 4)
    helped = d_prec > 0 or d_f1 > 0
    lines = [
        "# X-RAG · evaluation — raw base vs fine-tuned (full test set)", "",
        f"Evaluated **{summary['n']}** queries from `{summary['source']}` through "
        "the full live-web pipeline. Both answers come from the SAME retrieved "
        "chunks and are scored by the SAME NLI model, so any difference is the "
        "effect of fine-tuning.", "",
        "## Summary", "",
        "| metric | raw base | tuned (SFT+DPO) | Δ (tuned − raw) |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("cite_precision", "citation precision"),
        ("cite_recall", "citation recall"),
        ("cite_f1", "citation F1"),
    ]:
        rv, tv = R[key], T[key]
        lines.append(f"| {label} | {rv:.4f} | {tv:.4f} | {round(tv - rv, 4):+.4f} |")
    verdict = ("**Fine-tuning improved citation faithfulness**" if helped
               else "**Fine-tuning did not improve on this test set**")
    lines += ["", f"{verdict} — Δprecision {d_prec:+.3f}, ΔF1 {d_f1:+.3f}.", "",
              "## Per-query (trust, raw vs tuned)", "",
              "| query | raw trust | tuned trust | Δtrust |",
              "|---|---:|---:|---:|"]
    for r in per_q:
        lines.append(f"| {r['q'][:60]} | {r['raw_trust']:.2f} | {r['tuned_trust']:.2f} | "
                     f"{r['tuned_trust'] - r['raw_trust']:+.2f} |")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint + a small built-in fallback eval set
# ---------------------------------------------------------------------------
_BUILTIN_QUERIES = [
    {"query": "who won the 2022 world cup and who scored the most goals"},
    {"query": "how tall is the Eiffel Tower"},
    {"query": "what is the capital of Australia"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/test/sft.jsonl",
                    help="JSONL with a `query` field per row (default: the full "
                         "SFT test set).")
    ap.add_argument("--out", default="docs/eval_results.md")
    ap.add_argument("--dual", action="store_true",
                    help="Compare raw base vs fine-tuned (the final comparison). "
                         "Without it, evaluates the tuned pipeline only.")
    ap.add_argument("--max_queries", type=int, default=None,
                    help="Cap the number of queries (default: the ENTIRE file).")
    args = ap.parse_args()

    queries = load_eval_queries(args.queries, max_queries=args.max_queries)
    if not queries:
        log.error("no queries to evaluate"); return
    if args.dual:
        evaluate_dual(queries, args.out, source=args.queries)
    else:
        evaluate(queries, args.out, source=args.queries)


if __name__ == "__main__":
    main()
