r"""
Corrective / iterative RAG orchestrator (a LangGraph state machine).

The linear `engine.run()` does rewrite -> retrieve -> generate -> attribute once.
This version adds a CYCLE: if the answer has no GREEN (supported) claim, it
re-rewrites the query with the raw 8B few-shot rewriter (asking for DIFFERENT
sub-queries that keep the same meaning), re-searches, and tries again - up to
`orchestrator.max_attempts`, bounded by `budget.total_sec`. The best attempt
(most green claims) is the one shipped.

    START -> safety_in --(BLOCK)--> refuse -> END
                      \--(ok)----> rewrite -> retrieve
                                      ^           |
                          (no green & |           |--(no chunks & retries left)--^
                           retries    |           |--(chunks)--> generate -> attribute
                           left)      \------------------------------(no green & retries left)
                                                                      |--(green | out of attempts)--> assemble -> END

Each pipeline step is a plain node function; LangGraph just wires them. If
`langgraph` isn't installed (or the graph errors) we run the SAME nodes through
an equivalent plain-Python driver, so the feature degrades gracefully.

The routing decision (`_decide_next`) is a pure function so it's unit-tested
without a GPU.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from app.api.assemble import build_response
from app.attribution.scorer import score_answer
from app.generation.generator import generate
from app.orchestrator.engine import OnStep, _ms, _noop, _refusal_response
from app.planning.rewriter import rewrite_via_generator
from app.retrieval.pipeline import retrieve
from app.safety.guard import classify
from app.schemas import AnswerResponse, QueryRequest, SafetyVerdict
from app.util import get_logger, new_trace_id, timed

log = get_logger(__name__)

State = Dict[str, Any]


# ---------------------------------------------------------------------------
# Pure routing logic (no GPU - unit-tested)
# ---------------------------------------------------------------------------
def _decide_next(green: int, attempt: int, max_attempts: int,
                 elapsed: float, budget: float) -> str:
    """RETRY if we have no green-supported claim AND attempts remain AND we're
    within the time budget; otherwise we're DONE (ship the best attempt)."""
    if green > 0:
        return "done"
    if attempt >= max_attempts:
        return "done"
    if budget and elapsed >= budget:
        return "done"
    return "retry"


def _green_count(scored) -> int:
    return sum(1 for s in scored if s.flag == "green")


# ---------------------------------------------------------------------------
# Nodes (each takes the state, mutates + returns it)
# ---------------------------------------------------------------------------
def _node_safety_in(s: State) -> State:
    on_step = s["on_step"]
    on_step("safety_in", "start", None)
    with timed("safety_in", log):
        s["verdict_in"] = classify(s["req"].query, role="user")
    on_step("safety_in", "done", {"action": s["verdict_in"].action,
                                  "category": s["verdict_in"].category})
    return s


def _node_rewrite(s: State) -> State:
    on_step, req = s["on_step"], s["req"]
    s["attempt"] += 1
    # Diversify more on each retry so we don't just re-fetch the same pages.
    temp = 0.2 if s["attempt"] == 1 else min(0.9, 0.2 + 0.3 * (s["attempt"] - 1))
    on_step("rewrite", "start", None)
    with timed("rewrite", log):
        s["rw"] = rewrite_via_generator(req.query, history=req.history,
                                        avoid_queries=s["avoid"] or None,
                                        temperature=temp)
    log.info("[%s] attempt %d sub_queries=%s", s["trace_id"], s["attempt"],
             s["rw"].sub_queries)
    on_step("rewrite", "done", {"standalone": s["rw"].standalone_query,
                                "sub_queries": s["rw"].sub_queries,
                                "intent": s["rw"].intent,
                                "rewriter": "raw-8b", "attempt": s["attempt"]})
    return s


def _node_retrieve(s: State) -> State:
    req, rw = s["req"], s["rw"]
    with timed("retrieve", log):
        s["chunks"] = retrieve(rw.sub_queries, top_k=req.top_k,
                               on_step=s["on_step"],
                               standalone_query=rw.standalone_query)
    if not s["chunks"]:
        # these sub-queries found nothing - avoid them on the next attempt.
        s["avoid"] = (s["avoid"] or []) + list(rw.sub_queries)
    return s


def _node_generate(s: State) -> State:
    on_step = s["on_step"]
    on_step("generate", "start", None)
    with timed("generate", log):
        s["answer"] = generate(s["rw"].standalone_query, s["chunks"])
    on_step("generate", "done", {"answer_preview": s["answer"][:240],
                                 "answer_chars": len(s["answer"])})
    return s


def _node_attribute(s: State) -> State:
    from config.settings import settings
    on_step = s["on_step"]
    # ---- output-side safety ----
    on_step("safety_out", "start", None)
    with timed("safety_out", log):
        s["verdict_out"] = (classify(s["answer"], role="assistant")
                            if settings.safety.output_side_check
                            else SafetyVerdict(action="ALLOW"))
    on_step("safety_out", "done", {"action": s["verdict_out"].action,
                                   "category": s["verdict_out"].category})
    if s["verdict_out"].action == "BLOCK":
        s["route"] = "refuse_out"
        return s
    # ---- attribution scoring ----
    on_step("attribute", "start", None)
    with timed("attribute", log):
        if s["verdict_out"].action == "CONTROLLED":
            saved = settings.attribution.threshold
            settings.attribution.threshold = max(saved, 0.85)
            s["scored"] = score_answer(s["answer"], s["rw"].standalone_query, s["chunks"])
            settings.attribution.threshold = saved
        else:
            s["scored"] = score_answer(s["answer"], s["rw"].standalone_query, s["chunks"])
    green = _green_count(s["scored"])
    red = sum(1 for sc in s["scored"] if sc.flag == "red")
    on_step("attribute", "done", {"claim_count": len(s["scored"]), "green": green,
                                  "red": red, "attempt": s["attempt"]})
    # Track the best attempt (most green-supported claims) so we always ship the
    # strongest one even if no attempt is fully supported.
    if s["best"] is None or green > s["best"]["green"]:
        s["best"] = {"answer": s["answer"], "scored": s["scored"],
                     "chunks": s["chunks"], "verdict_out": s["verdict_out"],
                     "green": green}
    if green == 0:
        s["avoid"] = (s["avoid"] or []) + list(s["rw"].sub_queries)
    return s


def _node_assemble(s: State) -> State:
    on_step, req = s["on_step"], s["req"]
    best = s["best"]
    if best is None:
        # never produced a scored answer (e.g. retrieval was empty every attempt)
        s["result"] = AnswerResponse(
            answer="I couldn't find well-supported sources for this after "
                   f"{s['attempt']} search attempt(s).",
            citations=[], overall_trust=0.0, trace_id=s["trace_id"],
            mode=req.mode, latency_ms=_ms(s["t0"]), safety=s["verdict_in"])
        return s
    on_step("assemble", "start", None)
    resp = build_response(
        answer=best["answer"], scored=best["scored"], chunks=best["chunks"],
        trace_id=s["trace_id"], mode=req.mode, latency_ms=_ms(s["t0"]),
        safety=best["verdict_out"] if best["verdict_out"].action != "ALLOW"
        else s["verdict_in"])
    on_step("assemble", "done", {"overall_trust": resp.overall_trust,
                                 "latency_ms": resp.latency_ms,
                                 "citations": len(resp.citations),
                                 "attempts": s["attempt"]})
    s["result"] = resp
    return s


def _node_refuse(s: State) -> State:
    v = s["verdict_out"] if s.get("route") == "refuse_out" else s["verdict_in"]
    s["result"] = _refusal_response(v, s["trace_id"], s["req"].mode, ms=_ms(s["t0"]))
    return s


# ---------------------------------------------------------------------------
# Routing functions (pure; consumed by LangGraph conditional edges)
# ---------------------------------------------------------------------------
def _route_after_safety(s: State) -> str:
    return "refuse" if s["verdict_in"].action == "BLOCK" else "rewrite"


def _route_after_retrieve(s: State) -> str:
    if s["chunks"]:
        return "generate"
    elapsed = time.perf_counter() - s["t0"]
    return ("retry" if _decide_next(0, s["attempt"], s["max_attempts"],
                                    elapsed, s["budget_sec"]) == "retry"
            else "assemble")


def _route_after_attribute(s: State) -> str:
    if s.get("route") == "refuse_out":
        return "refuse_out"
    elapsed = time.perf_counter() - s["t0"]
    return _decide_next(_green_count(s["scored"]), s["attempt"],
                        s["max_attempts"], elapsed, s["budget_sec"])


# ---------------------------------------------------------------------------
# Drivers: LangGraph (preferred) + an equivalent plain-Python fallback
# ---------------------------------------------------------------------------
def _build_app():
    """Compile the LangGraph state machine (lazy import so this module loads
    without langgraph installed)."""
    from langgraph.graph import END, START, StateGraph
    g = StateGraph(dict)
    g.add_node("safety_in", _node_safety_in)
    g.add_node("rewrite", _node_rewrite)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("generate", _node_generate)
    g.add_node("attribute", _node_attribute)
    g.add_node("assemble", _node_assemble)
    g.add_node("refuse", _node_refuse)
    g.add_edge(START, "safety_in")
    g.add_conditional_edges("safety_in", _route_after_safety,
                            {"refuse": "refuse", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("retrieve", _route_after_retrieve,
                            {"generate": "generate", "retry": "rewrite",
                             "assemble": "assemble"})
    g.add_edge("generate", "attribute")
    g.add_conditional_edges("attribute", _route_after_attribute,
                            {"retry": "rewrite", "done": "assemble",
                             "refuse_out": "refuse"})
    g.add_edge("assemble", END)
    g.add_edge("refuse", END)
    return g.compile()


def _run_plain(s: State) -> AnswerResponse:
    """Same nodes + routing as the graph, driven by a plain loop. Used when
    langgraph isn't available or the compiled graph errors."""
    s = _node_safety_in(s)
    if _route_after_safety(s) == "refuse":
        return _node_refuse(s)["result"]
    # hard cap on loop iterations as a final safety net
    for _ in range(s["max_attempts"] + 1):
        s = _node_rewrite(s)
        s = _node_retrieve(s)
        r = _route_after_retrieve(s)
        if r == "retry":
            continue
        if r == "assemble":
            return _node_assemble(s)["result"]
        s = _node_generate(s)
        s = _node_attribute(s)
        r = _route_after_attribute(s)
        if r == "refuse_out":
            return _node_refuse(s)["result"]
        if r == "retry":
            continue
        return _node_assemble(s)["result"]
    return _node_assemble(s)["result"]


def _initial_state(req: QueryRequest, on_step: OnStep, trace_id: str) -> State:
    from config.settings import settings
    return {
        "req": req, "trace_id": trace_id, "on_step": on_step,
        "t0": time.perf_counter(), "attempt": 0,
        "max_attempts": max(1, settings.orchestrator.max_attempts),
        "budget_sec": float(settings.budget.total_sec or 0.0),
        "avoid": [], "best": None,
        "verdict_in": None, "verdict_out": None, "rw": None,
        "chunks": [], "answer": "", "scored": [], "route": None, "result": None,
    }


def run_corrective(req: QueryRequest, on_step: OnStep = None) -> AnswerResponse:
    """Entry point: corrective-RAG via LangGraph, with a plain-driver fallback."""
    on_step = on_step or _noop
    trace_id = new_trace_id()
    on_step("init", "done", {"trace_id": trace_id, "query": req.query, "mode": req.mode})
    from config.settings import settings
    log.info("[%s] corrective-RAG start: %r (max_attempts=%d)", trace_id,
             req.query, max(1, settings.orchestrator.max_attempts))

    try:
        import langgraph  # noqa: F401  (probe availability)
        have_langgraph = True
    except Exception:
        have_langgraph = False

    if have_langgraph:
        try:
            app = _build_app()
            limit = 6 * max(1, settings.orchestrator.max_attempts) + 10
            final = app.invoke(_initial_state(req, on_step, trace_id),
                               {"recursion_limit": limit})
            if final.get("result") is not None:
                return final["result"]
            log.warning("[%s] graph produced no result - using plain driver", trace_id)
        except Exception as e:
            log.warning("[%s] LangGraph failed (%s) - using plain driver", trace_id, e)

    return _run_plain(_initial_state(req, on_step, trace_id))
