"""
Dual-LLM A/B orchestrator. Useful for answering one question that the
question "did fine-tuning actually help with cited generation?".

Idea: do all the expensive shared work once (safety, rewrite, retrieve,
NLI scoring infrastructure), then run only STEP 9 twice with two
different generators, score both answers against the SAME retrieved
chunks, and return both `AnswerResponse`s plus a small delta dict.

The two generators are:
  * RAW   -> Llama 3.1-8B Instruct with NO adapter (pure base model)
  * TUNED -> Llama 3.1-8B Instruct with the LoRA adapter from config

Both use exactly the same prompt builder, the same decoding settings,
the same chunks, and the same attribution model, so any difference in
the trust scores is attributable to fine-tuning.

The `on_step` callback is supported. The dual orchestrator emits these
extra events alongside the regular ones (so a UI can show progress):

    generate_raw   / generate_raw   done
    attribute_raw  / attribute_raw  done
    generate_tuned / generate_tuned done
    attribute_tuned/ attribute_tuned done
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from app.api.assemble import build_response
from app.attribution.scorer import score_answer
from app.generation.generator import generate_closed_book, generate_with_adapter
from app.orchestrator.engine import OnStep, _noop, _refusal_response, _ms
from app.planning.rewriter import rewrite
from app.retrieval.pdf import PDFExtractionError
from app.retrieval.pipeline import retrieve, retrieve_from_pdf
from app.safety.guard import classify
from app.schemas import (AnswerResponse, DualResponse, QueryRequest,
                         SafetyVerdict)
from app.util import get_logger, new_trace_id, timed

log = get_logger(__name__)


def run_dual(req: QueryRequest, on_step: OnStep = None) -> DualResponse:
    """Run both raw and tuned generators against the same retrieval and the
    same NLI scorer. Returns a single `DualResponse` with both answers
    plus headline deltas (tuned - raw)."""
    from config.settings import settings
    on_step = on_step or _noop

    trace_id = new_trace_id()
    t0 = time.perf_counter()
    log.info("[%s] DUAL start: %r mode=%s", trace_id, req.query, req.mode)
    on_step("init", "done", {"trace_id": trace_id, "query": req.query, "mode": req.mode})

    # ---- Block 2: safety on the input ----
    on_step("safety_in", "start", None)
    with timed("safety_in", log):
        verdict_in = classify(req.query, role="user")
    on_step("safety_in", "done", {
        "action": verdict_in.action, "category": verdict_in.category,
    })
    if verdict_in.action == "BLOCK":
        ref = _refusal_response(verdict_in, trace_id, req.mode, ms=_ms(t0))
        # Return the same refusal in both slots. We `.model_copy()` so a
        # downstream consumer mutating dual.raw doesn't accidentally change
        # dual.tuned (pydantic models are mutable by default).
        return DualResponse(raw=ref, tuned=ref.model_copy(), trace_id=trace_id,
                            mode=req.mode, latency_ms=_ms(t0))

    # ---- Block 3: rewrite ----
    # Shared by both generator variants; the raw-rewriter toggle picks which
    # rewriter produces the (shared) sub-queries / fan-out.
    rewriter_adapter = None if req.use_raw_rewriter else "__from_config__"
    on_step("rewrite", "start", None)
    with timed("rewrite", log):
        rw = rewrite(req.query, history=req.history, adapter_path=rewriter_adapter)
    on_step("rewrite", "done", {
        "standalone": rw.standalone_query,
        "sub_queries": rw.sub_queries,
        "intent": rw.intent,
        "rewriter": "raw" if req.use_raw_rewriter else "tuned",
    })

    # ---- Blocks 4-8: retrieve (same as engine.run) ----
    with timed("retrieve", log):
        if req.mode == "pdf_offline":
            if not req.pdf_path:
                empty = AnswerResponse(
                    answer="No PDF was uploaded.",
                    citations=[], overall_trust=0.0,
                    trace_id=trace_id, mode=req.mode, latency_ms=_ms(t0),
                    safety=verdict_in)
                return DualResponse(raw=empty, tuned=empty.model_copy(),
                                    trace_id=trace_id,
                                    mode=req.mode, latency_ms=_ms(t0))
            try:
                chunks = retrieve_from_pdf(rw.standalone_query, req.pdf_path,
                                           top_k=req.top_k, on_step=on_step)
            except PDFExtractionError as e:
                err = AnswerResponse(
                    answer=str(e),
                    citations=[], overall_trust=0.0,
                    trace_id=trace_id, mode=req.mode, latency_ms=_ms(t0),
                    safety=verdict_in)
                return DualResponse(raw=err, tuned=err.model_copy(),
                                    trace_id=trace_id,
                                    mode=req.mode, latency_ms=_ms(t0))
        else:
            chunks = retrieve(rw.sub_queries, top_k=req.top_k, on_step=on_step,
                              standalone_query=rw.standalone_query)

    if not chunks:
        log.warning("[%s] dual: no chunks", trace_id)
        empty = AnswerResponse(
            answer="I don't know based on the sources.",
            citations=[], overall_trust=0.0,
            trace_id=trace_id, mode=req.mode, latency_ms=_ms(t0),
            safety=verdict_in)
        return DualResponse(raw=empty, tuned=empty.model_copy(),
                            trace_id=trace_id,
                            mode=req.mode, latency_ms=_ms(t0))

    # ---- Step 9 and Block 10 run TWICE, sequentially. ----
    # Order: TUNED first (this is the "main" answer we ship), then RAW
    # (for comparison). Both are scored with the SAME NLI model against
    # the SAME chunks so any difference is attributable to fine-tuning.
    # Each variant gets its OWN start clock so `latency_ms` measures that
    # variant's wall time, not "total since the request began" (which would
    # double-count the tuned run inside the raw response).
    tuned_resp = _generate_and_score(
        req, chunks, rw, trace_id, verdict_in,
        adapter_path=settings.generator.adapter_path, label="tuned",
        on_step=on_step,
    )
    raw_resp = _generate_and_score(
        req, chunks, rw, trace_id, verdict_in,
        adapter_path=None, label="raw", on_step=on_step,
    )

    # ---- Closed-book baseline: RAW base model, NO retrieval (just the query). ----
    # Shows what the bare LLM produces with no chunks, so the user can see how
    # much retrieval + grounding actually add. No scoring (nothing to attribute).
    on_step("generate_noretr", "start", None)
    with timed("generate_noretr", log):
        try:
            noretr_answer = generate_closed_book(req.query, adapter_path=None)
        except Exception as e:
            log.warning("[%s] closed-book generation failed (%s)", trace_id, e)
            noretr_answer = ""
    on_step("generate_noretr", "done", {
        "answer_preview": noretr_answer[:200],
        "answer_chars": len(noretr_answer),
    })

    # ---- Headline deltas (tuned - raw). Positive trust means tuned is better. ----
    d_trust = round(tuned_resp.overall_trust - raw_resp.overall_trust, 3)
    d_green = (sum(1 for c in tuned_resp.citations if c.flag == "green")
               - sum(1 for c in raw_resp.citations if c.flag == "green"))
    d_red   = (sum(1 for c in tuned_resp.citations if c.flag == "red")
               - sum(1 for c in raw_resp.citations if c.flag == "red"))
    on_step("compare", "done", {
        "delta_trust": d_trust,
        "delta_green": d_green,
        "delta_red": d_red,
    })

    return DualResponse(
        raw=raw_resp, tuned=tuned_resp,
        raw_no_retrieval=noretr_answer,
        delta_overall_trust=d_trust,
        delta_green_citations=d_green,
        delta_red_citations=d_red,
        trace_id=trace_id, mode=req.mode, latency_ms=_ms(t0),
    )


# ---------------------------------------------------------------------------
# helper: generate + score for one variant (raw or tuned)
# ---------------------------------------------------------------------------
def _generate_and_score(req: QueryRequest, chunks, rw,
                        trace_id: str,
                        verdict_in: SafetyVerdict,
                        adapter_path: str | None, label: str,
                        on_step: OnStep) -> AnswerResponse:
    """Generate an answer with the given adapter, score it, and assemble.

    `label` is one of "raw" / "tuned". It is used only for the event names
    (so the UI can show which side it is rendering).

    The latency stamped on the returned `AnswerResponse` measures THIS
    variant only - it starts here, not at the top of `run_dual`. The shared
    trace_id is left untouched so OTel tooling parsing it doesn't break.

    Output-side safety runs HERE (once per variant) - the engine path does
    the same; dual was previously skipping it, letting a poisoned source
    steer either model into unsafe output that shipped unchecked.
    """
    from config.settings import settings
    t_start = time.perf_counter()

    on_step(f"generate_{label}", "start", None)
    with timed(f"generate_{label}", log):
        answer = generate_with_adapter(rw.standalone_query, chunks, adapter_path=adapter_path)
    on_step(f"generate_{label}", "done", {
        "answer_preview": answer[:200],
        "answer_chars": len(answer),
    })

    # ---- Block 2 (output-side): re-check generated text ----
    on_step(f"safety_out_{label}", "start", None)
    with timed(f"safety_out_{label}", log):
        if settings.safety.output_side_check:
            verdict_out = classify(answer, role="assistant")
        else:
            verdict_out = SafetyVerdict(action="ALLOW")
    on_step(f"safety_out_{label}", "done", {
        "action": verdict_out.action, "category": verdict_out.category,
    })
    if verdict_out.action == "BLOCK":
        log.info("[%s] %s BLOCKED on output (%s)",
                 trace_id, label, verdict_out.category)
        return _refusal_response(verdict_out, trace_id, req.mode, ms=_ms(t_start))

    on_step(f"attribute_{label}", "start", None)
    with timed(f"attribute_{label}", log):
        # CONTROLLED tightens the bar on attribution exactly like engine.run.
        if verdict_out.action == "CONTROLLED":
            saved = settings.attribution.threshold
            settings.attribution.threshold = max(saved, 0.85)
            scored = score_answer(answer, rw.standalone_query, chunks)
            settings.attribution.threshold = saved
        else:
            scored = score_answer(answer, rw.standalone_query, chunks)
    n_green = sum(1 for s in scored if s.flag == "green")
    n_red   = sum(1 for s in scored if s.flag == "red")
    on_step(f"attribute_{label}", "done", {
        "claim_count": len(scored), "green": n_green, "red": n_red,
    })

    resp = build_response(
        answer=answer, scored=scored, chunks=chunks,
        trace_id=trace_id, mode=req.mode, latency_ms=_ms(t_start),
        safety=verdict_out if verdict_out.action != "ALLOW" else verdict_in,
    )
    return resp
