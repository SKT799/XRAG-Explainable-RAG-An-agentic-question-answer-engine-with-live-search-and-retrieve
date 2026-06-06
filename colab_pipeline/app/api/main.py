"""
Block 1 · FastAPI entrypoint.

Run with:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000

Set the env var XRAG_WARMUP=1 to pin the embedder + reranker in VRAM at
startup (saves ~5 seconds on every query). Recommended on A100; skip on T4
where you may want lazy loading.

Routes:
    GET  /health      → {"ok": true}
    POST /v1/answer   → AnswerResponse (the cited response)
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI

from app.orchestrator.engine import run
from app.schemas import AnswerResponse, QueryRequest
from app.util import get_logger

log = get_logger(__name__)

# The pipeline is synchronous + GPU-bound; we run it in a thread so the event
# loop stays responsive (and several Uvicorn workers can each have one in-flight).
_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(
    title="X-RAG",
    description="End-to-end Explainable & Fine-Tuned Retrieval-Augmented Generation",
    version="0.1.0",
)


@app.on_event("startup")
def _warmup_if_enabled() -> None:
    if os.environ.get("XRAG_WARMUP", "").lower() not in ("1", "true", "yes"):
        log.info("XRAG_WARMUP not set; embedder/reranker will load lazily")
        return
    # Lazy imports so the FastAPI module stays importable without
    # FlagEmbedding installed (e.g. for the Gradio-only deployment).
    try:
        from app.retrieval.embed_retrieve import warmup_embedder
        from app.retrieval.rerank import warmup_reranker
        warmup_embedder()
        warmup_reranker()
    except Exception as e:
        log.warning("warmup failed (%s) - continuing with lazy loading", e)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/v1/answer", response_model=AnswerResponse)
async def answer(req: QueryRequest) -> AnswerResponse:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, run, req)
