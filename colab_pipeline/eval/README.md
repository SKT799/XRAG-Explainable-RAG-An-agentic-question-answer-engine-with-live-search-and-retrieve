# eval

Runs a list of queries through the full pipeline and prints scores.

`harness.py` does the work. It calls `app.orchestrator.engine.run` for each query, computes citation precision and recall (and their F1), average trust, p50 and p95 latency, and writes a markdown report.

## Run it

```bash
python -m eval.harness
```

That uses a built-in 3-query smoke set. Pass your own queries with `--queries`:

```bash
python -m eval.harness --queries eval/queries.jsonl --out docs/eval_results.md
```

Each line of `queries.jsonl` is one JSON object with at least a `query` field.

## What it measures

| Metric | What it answers |
|---|---|
| citation precision | of the citations we emitted, how many got a green flag |
| citation recall | average trust score across the run (proxy) |
| F1 | the usual harmonic mean of the two |
| overall trust | mean of per-claim trust scores |
| p50 / p95 latency | from `response.latency_ms` |

## Ablations

Toggle config knobs and rerun to compare:

- generator adapter on or off (`generator.adapter_path: null`)
- DPO vs SFT-only
- reranker on or off (bump `reranking.top_k_keep` to 50 to skip the rerank step)
- fan-out on or off (set `rewriter.max_sub_queries: 1`)
- int4 vs bf16 (`generator.quantization.mode`)

Save each run's `docs/eval_results.md` so you can diff them.
