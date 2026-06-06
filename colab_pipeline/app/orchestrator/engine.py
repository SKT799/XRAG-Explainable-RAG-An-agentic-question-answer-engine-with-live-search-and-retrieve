"""
The orchestrator. Wires Blocks 2 -> 11 into one `run(request)` call.

Responsibilities:
  * Generate a `trace_id` and stamp every log line with it.
  * Call each block in order, under a per-request time budget.
  * Honor the safety verdict (BLOCK -> refuse; CONTROLLED -> tighten threshold).
  * Catch per-stage exceptions so one bad page never kills the whole request.
  * Optionally emit per-stage events via an `on_step` callback so a UI can
    show progress as the pipeline runs (the Gradio "Show process" toggle).

This is a SYNCHRONOUS engine for simplicity (Colab notebooks call it directly).
The FastAPI route in `app/api/main.py` runs it in an executor for async-friendliness.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from app.api.assemble import build_response
from app.attribution.scorer import score_answer
from app.generation.generator import generate
from app.planning.rewriter import rewrite
from app.retrieval.pdf import PDFExtractionError
from app.retrieval.pipeline import retrieve, retrieve_from_pdf
from app.safety.guard import classify
from app.schemas import AnswerResponse, QueryRequest, SafetyVerdict
from app.util import get_logger, new_trace_id, timed

log = get_logger(__name__)


# Type alias for the optional per-stage event callback.
# Signature: on_step(stage_key: str, status: "start" | "done", payload: dict | None)
OnStep = Optional[Callable[[str, str, Optional[dict]], None]]


_REFUSAL_TEMPLATE = (
    "I can't help with that. It falls under the policy category {category}.\n"
    "If you believe this is a mistake, rephrase the question more specifically."
)


def _noop(*args, **kwargs):
    """Default on_step that does nothing."""
    pass


def run(req: QueryRequest, on_step: OnStep = None) -> AnswerResponse:
    """End-to-end synchronous pipeline.

    Pass `on_step` to receive live progress events. Each event is a tuple of
        (stage_key, status, payload)
    where status is "start" or "done" and payload is a small dict of
    stage-specific info (verdict, sub_queries, chunk count, scores, etc.).
    """
    from config.settings import settings
    on_step = on_step or _noop

    # Corrective / iterative RAG (LangGraph state machine): retry rewrite + search
    # when an answer has no green (supported) claim. live_web only - you can't
    # re-search a fixed PDF. Any error here falls through to the linear path below.
    if (getattr(settings, "orchestrator", None) is not None
            and settings.orchestrator.corrective_rag and req.mode != "pdf_offline"):
        try:
            from app.orchestrator.graph import run_corrective
            return run_corrective(req, on_step)
        except Exception as e:
            log.warning("corrective-RAG path failed (%s) - linear fallback", e)

    trace_id = new_trace_id()
    t0 = time.perf_counter()
    log.info("[%s] start: %r mode=%s", trace_id, req.query, req.mode)
    on_step("init", "done", {"trace_id": trace_id, "query": req.query, "mode": req.mode})

    # ---- Block 2: input-side safety ----
    on_step("safety_in", "start", None)
    with timed("safety_in", log):
        verdict_in = classify(req.query, role="user")
    on_step("safety_in", "done", {
        "action": verdict_in.action,
        "category": verdict_in.category,
        "reason": verdict_in.reason,
    })
    if verdict_in.action == "BLOCK":
        log.info("[%s] BLOCKED on input (%s)", trace_id, verdict_in.category)
        return _refusal_response(verdict_in, trace_id, req.mode, ms=_ms(t0))

    # ---- Block 3: rewrite + fan-out ----
    # The UI can request the RAW base rewriter (no adapter) instead of the trained
    # one, to show the rewriter's effect on the sub-queries / fan-out.
    rewriter_adapter = None if req.use_raw_rewriter else "__from_config__"
    on_step("rewrite", "start", None)
    with timed("rewrite", log):
        rw = rewrite(req.query, history=req.history, adapter_path=rewriter_adapter)
    log.info("[%s] sub_queries=%s (rewriter=%s)", trace_id, rw.sub_queries,
             "raw" if req.use_raw_rewriter else "tuned")
    on_step("rewrite", "done", {
        "standalone": rw.standalone_query,
        "sub_queries": rw.sub_queries,
        "intent": rw.intent,
        "rewriter": "raw" if req.use_raw_rewriter else "tuned",
    })

    # ---- Blocks 4-8: retrieval spine. Pick the right source based on mode.
    # `retrieve` and `retrieve_from_pdf` both emit their own sub-events.
    with timed("retrieve", log):
        if req.mode == "pdf_offline":
            if not req.pdf_path:
                log.error("[%s] pdf_offline mode but no pdf_path", trace_id)
                on_step("pdf_parse", "done", {"chunk_count": 0, "error": "no pdf"})
                return AnswerResponse(
                    answer="No PDF was uploaded. Switch back to live_web "
                           "mode or upload a PDF and try again.",
                    citations=[], overall_trust=0.0,
                    trace_id=trace_id, mode=req.mode, latency_ms=_ms(t0),
                    safety=verdict_in)
            try:
                chunks = retrieve_from_pdf(
                    rw.standalone_query, req.pdf_path,
                    top_k=req.top_k, on_step=on_step,
                )
            except PDFExtractionError as e:
                log.warning("[%s] PDF extraction error (%s): %s",
                            trace_id, e.kind, e)
                return AnswerResponse(
                    answer=str(e),                       # user-facing reason
                    citations=[], overall_trust=0.0,
                    trace_id=trace_id, mode=req.mode, latency_ms=_ms(t0),
                    safety=verdict_in)
        else:
            chunks = retrieve(rw.sub_queries, top_k=req.top_k,
                              on_step=on_step,
                              standalone_query=rw.standalone_query)
    if not chunks:
        log.warning("[%s] no chunks returned, answering 'I don't know'", trace_id)
        on_step("assemble", "done", {"empty": True})
        return AnswerResponse(answer="I don't know based on the sources.",
                              citations=[], overall_trust=0.0,
                              trace_id=trace_id, mode=req.mode, latency_ms=_ms(t0),
                              safety=verdict_in)

    # ---- Block 9: generate ----
    on_step("generate", "start", None)
    with timed("generate", log):
        answer = generate(rw.standalone_query, chunks)
    on_step("generate", "done", {
        "answer_preview": answer[:240],
        "answer_chars": len(answer),
    })

    # ---- Block 2 (output-side): re-check generated text ----
    on_step("safety_out", "start", None)
    with timed("safety_out", log):
        verdict_out = classify(answer, role="assistant") if settings.safety.output_side_check \
            else SafetyVerdict(action="ALLOW")
    on_step("safety_out", "done", {
        "action": verdict_out.action,
        "category": verdict_out.category,
    })
    if verdict_out.action == "BLOCK":
        log.info("[%s] BLOCKED on output (%s)", trace_id, verdict_out.category)
        return _refusal_response(verdict_out, trace_id, req.mode, ms=_ms(t0))

    # ---- Block 10: attribution scoring ----
    on_step("attribute", "start", None)
    with timed("attribute", log):
        # If the verdict is CONTROLLED, raise the bar a little (stricter).
        if verdict_out.action == "CONTROLLED":
            saved = settings.attribution.threshold
            settings.attribution.threshold = max(saved, 0.85)
            scored = score_answer(answer, rw.standalone_query, chunks)
            settings.attribution.threshold = saved
        else:
            scored = score_answer(answer, rw.standalone_query, chunks)
    n_green = sum(1 for s in scored if s.flag == "green")
    n_red = sum(1 for s in scored if s.flag == "red")
    on_step("attribute", "done", {
        "claim_count": len(scored),
        "green": n_green,
        "red": n_red,
    })

    # ---- Block 11: assemble final response ----
    on_step("assemble", "start", None)
    resp = build_response(
        answer=answer, scored=scored, chunks=chunks,
        trace_id=trace_id, mode=req.mode, latency_ms=_ms(t0),
        safety=verdict_out if verdict_out.action != "ALLOW" else verdict_in,
    )
    on_step("assemble", "done", {
        "overall_trust": resp.overall_trust,
        "latency_ms": resp.latency_ms,
        "citations": len(resp.citations),
    })
    log.info("[%s] done in %d ms · trust=%.2f · cites=%d",
             trace_id, resp.latency_ms, resp.overall_trust, len(resp.citations))
    return resp


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _refusal_response(v: SafetyVerdict, trace_id: str, mode: str, ms: int) -> AnswerResponse:
    return AnswerResponse(
        answer=_REFUSAL_TEMPLATE.format(category=v.category or "unspecified"),
        citations=[], overall_trust=0.0,
        trace_id=trace_id, mode=mode, latency_ms=ms, safety=v,
    )


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)
