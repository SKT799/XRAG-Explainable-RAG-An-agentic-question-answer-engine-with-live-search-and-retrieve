# app

This is the actual runtime. The 11 steps of the pipeline live here, one per folder, plus a couple of files at the top that everything shares.

## Folder per step

| Step | Folder | What it does |
|---|---|---|
| 1 | `api/` | FastAPI endpoint, builds the final JSON response |
| 2 | `safety/` | Llama Guard plus a small policy table |
| 3 | `planning/` | rewrites the query and splits it into sub-queries |
| 4-8 | `retrieval/` | search, scrape, chunk, embed, rerank |
| 9 | `generation/` | the Llama 3.1-8B generator and the citation prompt |
| 10 | `attribution/` | DeBERTa NLI plus the trust score math |
| 11 | `api/assemble.py` | packages the final response |
| - | `orchestrator/` | the async engine that runs steps 2 through 11 in order |
| - | `storage/` | Redis cache, or the in-process fallback |

## Shared bits at the top

`schemas.py` is where the pydantic types live. Every step reads and writes one of these. If you change the shape of a citation here, the rest of the code follows.

`util.py` has the small helpers everyone needs. `new_trace_id()` for request IDs, `sha1()` for cache keys, `sigmoid()` because we use it in two places, a `timed()` context manager for per-step latency, and a `get_logger()` that does not duplicate handlers.

## How one request flows

```
QueryRequest
   -> safety.guard.classify(query)        -> SafetyVerdict
   -> planning.rewriter.rewrite(query)    -> RewriteResult
   -> retrieval.pipeline.retrieve(...)    -> list of Chunk (10 of them)
   -> generation.generator.generate(...)  -> the answer text with [n] cites
   -> attribution.scorer.score_answer(...)-> list of ScoredClaim
   -> api.assemble.build_response(...)    -> AnswerResponse
```

The `orchestrator/engine.py` is the file that wires this up. It also handles timeouts and the safety re-check on the output.

## A few rules I tried to stick to

I wrote the parts that make the project mine. That means the chunker, the RRF fusion, the citation prompt packing, and the trust score math. The plumbing (HTTP, HTML parsing, model forward passes, training loops) is library code.

Every external call (search, scrape, model) has a timeout. One bad page should not stall the request.

The model IDs, thresholds, and quantization mode all live in `config/config.yaml`. Nothing about behavior is hard coded in here.

The trace ID generated in step 1 gets passed through every step and shows up in the logs. Useful when something goes wrong.
