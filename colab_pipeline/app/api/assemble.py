"""
Block 11 · Assemble the final response.

Turn (answer string, per-claim trust scores, the chunks the model cited) into the
stable `AnswerResponse` JSON the UI / evaluators / auditors consume.
"""
from __future__ import annotations

from typing import Dict, List

from app.attribution.scorer import overall_trust
from app.schemas import (AnswerResponse, Chunk, Citation, SafetyVerdict,
                         ScoredClaim)


def _snippet(text: str, n: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def build_response(answer: str,
                   scored: List[ScoredClaim],
                   chunks: List[Chunk],
                   trace_id: str,
                   mode: str,
                   latency_ms: int,
                   safety: SafetyVerdict | None = None) -> AnswerResponse:
    """
    Produce the final response with one `Citation` per cited chunk. The per-citation
    score is the *best* score across the claims that cited that chunk (a single
    chunk used in multiple sentences gets the most favorable verdict).
    """
    by_idx: Dict[int, Chunk] = {i + 1: c for i, c in enumerate(chunks)}

    # Per-cited-chunk: collect the per-claim scores that cite it.
    # Strict `>` so the FIRST claim with the best score wins deterministically
    # - >= used to let tied later claims overwrite an earlier identical score
    # and the flag could flip between runs.
    score_by_cid: Dict[int, float] = {}
    flag_by_cid: Dict[int, str] = {}
    for sc in scored:
        for cid in sc.cited_ids:
            if cid not in by_idx:
                continue
            if sc.score > score_by_cid.get(cid, -1.0):
                score_by_cid[cid] = sc.score
                flag_by_cid[cid] = sc.flag

    citations: List[Citation] = []
    for cid in sorted(score_by_cid):
        c = by_idx[cid]
        citations.append(Citation(
            id=cid,
            url=c.provenance.url,
            title=c.provenance.title,
            locator=c.provenance.locator,
            snippet=_snippet(c.text),
            attribution_score=round(score_by_cid[cid], 3),
            flag=flag_by_cid[cid],
        ))

    return AnswerResponse(
        answer=answer.strip(),
        citations=citations,
        overall_trust=round(overall_trust(scored), 3),
        trace_id=trace_id,
        mode=mode,
        latency_ms=latency_ms,
        safety=safety,
    )
