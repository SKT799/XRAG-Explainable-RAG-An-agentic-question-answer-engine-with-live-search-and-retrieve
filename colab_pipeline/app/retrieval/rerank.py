"""
Block 8 · Cross-encoder reranking.

Pass the ~50 candidates from Block 7 through a **cross-encoder** (bge-reranker-v2-m3),
which reads `(query, chunk)` JOINTLY (full attention across the pair) and emits a
single relevance logit. We keep the **top-10** and stash the score on each chunk
as `scores['ce']`  -  Block 10 reuses it in the attribution formula.

Bi-encoder vs cross-encoder:
  * bi-encoder embeds query and doc SEPARATELY → cheap, can compare to many docs.
  * cross-encoder reads them TOGETHER → far more accurate, but O(n) model calls,
    which is why we run it only on the 50 already-shortlisted candidates.
"""
from __future__ import annotations

from typing import List

from app.schemas import Chunk
from app.util import get_logger, sigmoid

log = get_logger(__name__)


def _apply_relevance_gate(chunks: List[Chunk], floor: float) -> List[Chunk]:
    """Drop chunks whose sigmoid(CE) relevance is below `floor`. `floor <= 0`
    disables the gate. Pure (no model) so the gate is unit-testable."""
    if not floor or floor <= 0:
        return chunks
    return [c for c in chunks if sigmoid(c.scores.get("ce", 0.0)) >= floor]


class Reranker:
    """Lazy singleton around `FlagReranker`."""

    _instance: "Reranker | None" = None

    def __init__(self, model_id: str | None = None, use_fp16: bool = True):
        from config.settings import settings

        # Patch PreTrainedTokenizerBase.prepare_for_model if missing (transformers >= 4.47)
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, 'prepare_for_model'):
                log.info("Patching PreTrainedTokenizerBase.prepare_for_model for transformers >= 4.47 compatibility")
                def prepare_for_model(self, ids, pair_ids=None, **kwargs):
                    truncation = kwargs.get('truncation', 'only_second')
                    max_length = kwargs.get('max_length', None)
                    if max_length is not None:
                        total_len = len(ids) + (len(pair_ids) if pair_ids else 0)
                        if total_len > max_length:
                            if pair_ids is not None and truncation == 'only_second':
                                allowed_pair_len = max(0, max_length - len(ids))
                                pair_ids = pair_ids[:allowed_pair_len]
                            elif truncation == 'longest_first':
                                while len(ids) + (len(pair_ids) if pair_ids else 0) > max_length:
                                    if pair_ids and len(pair_ids) >= len(ids):
                                        pair_ids.pop()
                                    else:
                                        ids.pop()
                            else:
                                combined = (ids + (pair_ids or []))[:max_length]
                                ids = combined[:len(ids)]
                                pair_ids = combined[len(ids):] if pair_ids is not None else None
                    res = {'input_ids': ids + (pair_ids if pair_ids is not None else [])}
                    if kwargs.get('return_attention_mask', False):
                        res['attention_mask'] = [1] * len(res['input_ids'])
                    if kwargs.get('return_token_type_ids', False):
                        res['token_type_ids'] = [0] * len(ids) + [1] * (len(pair_ids) if pair_ids else 0)
                    return res
                PreTrainedTokenizerBase.prepare_for_model = prepare_for_model
        except Exception as e:
            log.warning("Failed to patch PreTrainedTokenizerBase: %s", e)

        from FlagEmbedding import FlagReranker             # lazy
        self.model_id = model_id or settings.reranking.model_id
        log.info("loading reranker %s (fp16=%s)…", self.model_id, use_fp16)
        self.model = FlagReranker(self.model_id, use_fp16=use_fp16)

    @classmethod
    def get(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def score_pairs(self, query: str, docs: List[str],
                    batch_size: int = 32) -> List[float]:
        """Return one logit per (query, doc) pair. Higher = more relevant.

        Scores in batches of `batch_size` so a large candidate list (e.g.
        a long PDF where `top_n_candidates` was bumped) doesn't try to
        forward 500 pairs in a single tensor and OOM the cross-encoder.
        Pure pass-through; preserves input order.
        """
        out: List[float] = []
        for i in range(0, len(docs), batch_size):
            chunk = docs[i:i + batch_size]
            pairs = [[query, d] for d in chunk]
            # `normalize=False` keeps raw logits so attribution can apply its own σ().
            scores = self.model.compute_score(pairs, normalize=False)
            # Single-pair runs sometimes return a bare float instead of a list.
            if isinstance(scores, (int, float)):
                scores = [scores]
            out.extend(float(s) for s in scores)
        return out


def warmup_reranker() -> None:
    """Pin the bge-reranker in VRAM. Call at server startup on A100."""
    log.info("warmup: loading bge-reranker…")
    r = Reranker.get()
    r.score_pairs("warmup", ["warmup target"])
    log.info("warmup: reranker ready")


def rerank(query: str, chunks: List[Chunk], top_k: int | None = None) -> List[Chunk]:
    """Score each (query, chunk) pair, attach `scores['ce']`, return top-K sorted desc."""
    from config.settings import settings
    if not chunks:
        return []
    top_k = top_k or settings.reranking.top_k_keep

    reranker = Reranker.get()
    scores = reranker.score_pairs(query, [c.text for c in chunks])
    for c, s in zip(chunks, scores):
        c.scores["ce"] = s
    ranked = sorted(chunks, key=lambda c: c.scores["ce"], reverse=True)[:top_k]

    # Relevance gate: drop chunks the cross-encoder scores as irrelevant so a
    # page that merely shares words with the query (but doesn't answer it) never
    # reaches the generator. Can return fewer than top_k - or empty, in which
    # case the orchestrator answers "I don't know" / re-searches (corrective RAG).
    floor = getattr(settings.reranking, "min_relevance", 0.0)
    before = len(ranked)
    ranked = _apply_relevance_gate(ranked, floor)
    if len(ranked) < before:
        log.info("relevance gate: dropped %d/%d chunks below relevance %.2f",
                 before - len(ranked), before, floor)

    for i, c in enumerate(ranked, start=1):
        c.rank = i
    if ranked:
        log.info("reranked → top-%d (best ce=%.3f, worst ce=%.3f)",
                 len(ranked), ranked[0].scores["ce"], ranked[-1].scores["ce"])
    else:
        log.info("reranked → 0 chunks left (all below relevance floor %.2f)", floor)
    return ranked
