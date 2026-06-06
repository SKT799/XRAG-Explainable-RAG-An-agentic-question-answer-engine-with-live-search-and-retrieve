"""
Block 7 · Embedding + hybrid retrieval.

Pipeline for this block:
    chunks ─► bge-m3 ┬─ dense vecs ─► FAISS IndexFlatIP (top-N dense)
                     └─ sparse vecs ──► top-N sparse
        ┌─────────────┴──────────────┐
        └──── RRF fuse ─► top-N candidate chunks

`rrf_fuse(...)` is pure-stdlib so the test in `tests/test_rrf.py` exercises the
ranking math without needing FAISS, torch, or any model weights.

For the FAISS index we build a *fresh* `IndexFlatIP` per request because in
live-web mode the corpus is just this query's ~300 chunks  -  flat brute-force
is faster than building HNSW.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

from app.schemas import Chunk
from app.util import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 0. Per-chunk embedding cache (bounded LRU). Chunk text is immutable, so the
# cache key is `chunk.id` and the value is `(dense_vec, sparse_lex_dict)`.
# This is what makes a 500-page PDF queryable cheaply: the first query pays
# the embedding cost, the next ones are ~free.
# ---------------------------------------------------------------------------
_EMB_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
# A100 host RAM is plentiful; 200k * (4KB dense + ~3KB sparse) ~= 1.4 GB.
# On T4 / Colab reduce to 20k.
_EMB_CACHE_MAX = 200_000


def _cache_get(chunk_id: str):
    v = _EMB_CACHE.get(chunk_id)
    if v is not None:
        _EMB_CACHE.move_to_end(chunk_id)
    return v


def _cache_put(chunk_id: str, dense_vec, sparse_lex) -> None:
    _EMB_CACHE[chunk_id] = (dense_vec, sparse_lex)
    _EMB_CACHE.move_to_end(chunk_id)
    while len(_EMB_CACHE) > _EMB_CACHE_MAX:
        _EMB_CACHE.popitem(last=False)


def clear_embedding_cache() -> int:
    """Drop the in-process embedding cache. Returns number of entries removed."""
    n = len(_EMB_CACHE)
    _EMB_CACHE.clear()
    return n


# ---------------------------------------------------------------------------
# 1. Reciprocal Rank Fusion (pure stdlib)
# ---------------------------------------------------------------------------
def rrf_fuse(ranked_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Merge several ranked lists of IDs into one final ranking.

    Each ranked list is *already* ordered (best first). For each id we sum
    1/(k + rank_in_list) over all the lists it appears in, then sort descending.

    Example: dense rank 1, sparse rank 3, k=60
        score = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

    Why this works: ranks (positions) are dimensionless across rankers, so we
    don't have to make their raw scores comparable.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank_minus_1, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_minus_1 + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# 2. The bge-m3 embedder (LIBRARY  -  FlagEmbedding gives dense + sparse + colbert)
# ---------------------------------------------------------------------------
class M3Embedder:
    """Lazy singleton wrapping `BGEM3FlagModel`."""

    _instance: "M3Embedder | None" = None

    def __init__(self, model_id: str | None = None, use_fp16: bool = True):
        from config.settings import settings
        from FlagEmbedding import BGEM3FlagModel        # lazy
        self.model_id = model_id or settings.embedding.model_id
        log.info("loading embedder %s (fp16=%s)…", self.model_id, use_fp16)
        self.model = BGEM3FlagModel(self.model_id, use_fp16=use_fp16)

    @classmethod
    def get(cls) -> "M3Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: List[str], return_sparse: bool = True) -> dict:
        """Returns {'dense_vecs': np.ndarray[N, d], 'lexical_weights': List[dict]}."""
        return self.model.encode(
            texts, return_dense=True, return_sparse=return_sparse, return_colbert_vecs=False,
        )


def warmup_embedder() -> None:
    """Pin the bge-m3 embedder in VRAM. Call at server startup on A100."""
    log.info("warmup: loading bge-m3 embedder…")
    em = M3Embedder.get()
    # One small encode pass to materialize CUDA kernels.
    em.encode(["warmup"], return_sparse=True)
    log.info("warmup: embedder ready")


# ---------------------------------------------------------------------------
# 3. Helpers for the dense + sparse rankings
# ---------------------------------------------------------------------------
def _dense_rank(query_vec, doc_vecs, ids: List[str], top_n: int
               ) -> Tuple[List[str], dict]:
    """FAISS IndexFlatIP (inner product = cosine when normalized).

    Returns `(ordered_ids, {id: ip_score})` so downstream code can record the
    REAL similarity score on each chunk (not a binary in/out flag).
    """
    import faiss
    import numpy as np
    d = doc_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    # bge-m3 returns already-normalized dense vectors, so IP == cosine.
    index.add(doc_vecs.astype(np.float32))
    D, I = index.search(query_vec.astype(np.float32), min(top_n, len(ids)))
    ordered_ids = [ids[i] for i in I[0] if i >= 0]
    scores = {ids[i]: float(D[0][k]) for k, i in enumerate(I[0]) if i >= 0}
    return ordered_ids, scores


def _sparse_rank(query_lex: dict, doc_lex_list: List[dict],
                 ids: List[str], top_n: int) -> Tuple[List[str], dict]:
    """Dot product on bge-m3 lexical weights (sparse, BM25-like).

    Returns `(ordered_ids, {id: score})`. The score is the raw weighted dot
    product so downstream code can record real numbers (not 1.0/0.0).
    """
    scores: list[tuple[str, float]] = []
    # Pre-cast the (typically tiny) query lexicon so the inner loop avoids
    # repeated float() calls on the same query tokens for every chunk.
    qw = {k: float(v) for k, v in query_lex.items()}
    for did, doc_lex in zip(ids, doc_lex_list):
        s = 0.0
        # Iterate over the SHORTER side - in practice the query is far smaller
        # than the per-doc lexicon, but we still skip docs with no overlap.
        if not doc_lex:
            scores.append((did, 0.0)); continue
        for tok, w in qw.items():
            dw = doc_lex.get(tok)
            if dw:
                s += w * float(dw)
        scores.append((did, s))
    scores.sort(key=lambda kv: kv[1], reverse=True)
    ordered = scores[:top_n]
    return [d for d, _ in ordered], {d: s for d, s in ordered}


# ---------------------------------------------------------------------------
# 4. The public function the pipeline calls
# ---------------------------------------------------------------------------
def retrieve_candidates(query: str, chunks: List[Chunk],
                        top_n: int | None = None) -> List[Chunk]:
    """
    Score every chunk against the query (dense + sparse), fuse with RRF,
    annotate each chunk with `scores['dense'|'sparse'|'rrf']`, return top-N.

    `dense` / `sparse` on the returned chunks are REAL scores (IP for dense,
    weighted dot product for sparse) when the chunk was in that ranker's
    top-N, and 0.0 otherwise. `rrf` is always populated.

    Chunk-level embeddings are cached by `chunk.id` (text is immutable), so
    repeated queries against the SAME corpus (e.g. multiple questions
    against one uploaded PDF) only pay the encoding cost once.
    """
    from config.settings import settings
    if not chunks:
        return []
    top_n = top_n or settings.embedding.top_n_candidates
    rrf_k = settings.embedding.rrf_k

    embedder = M3Embedder.get()
    ids = [c.id for c in chunks]
    import numpy as np

    # If this single batch is larger than the cache itself, the cache would
    # evict earlier misses before we read them back. Degenerate to a clean
    # encode-everything-fresh path instead of corrupting the cache.
    use_cache = len(chunks) <= _EMB_CACHE_MAX
    if not use_cache:
        log.info("batch (%d chunks) exceeds cache cap (%d) - bypassing cache",
                 len(chunks), _EMB_CACHE_MAX)
        all_enc = embedder.encode([c.text for c in chunks], return_sparse=True)
        doc_dense_arr = np.asarray(all_enc["dense_vecs"], dtype=np.float32)
        sparse_list = list(all_enc["lexical_weights"])
    else:
        # Find chunks that are NOT in the cache and encode only those.
        miss_chunks = [c for c in chunks if _cache_get(c.id) is None]
        if miss_chunks:
            log.info("embedding %d new chunks (%d cached)…",
                     len(miss_chunks), len(chunks) - len(miss_chunks))
            miss_enc = embedder.encode([c.text for c in miss_chunks], return_sparse=True)
            miss_dense = miss_enc["dense_vecs"]
            miss_sparse = miss_enc["lexical_weights"]
            for i, c in enumerate(miss_chunks):
                # Copy the dense slice so it doesn't keep the parent encode
                # array alive longer than needed, and so future LRU eviction
                # of one chunk frees real memory.
                _cache_put(c.id, np.asarray(miss_dense[i]).copy(), miss_sparse[i])

        # Gather (dense_vec, sparse_lex) for ALL chunks in input order from cache.
        dense_list, sparse_list = [], []
        for c in chunks:
            v = _cache_get(c.id)
            if v is None:
                # Should not happen given the size guard above, but defend
                # against concurrent cache_clear() races and re-encode on miss.
                log.warning("cache miss after fill for %s - re-encoding", c.id)
                one = embedder.encode([c.text], return_sparse=True)
                v = (np.asarray(one["dense_vecs"][0]).copy(),
                     one["lexical_weights"][0])
                _cache_put(c.id, *v)
            dense_list.append(v[0]); sparse_list.append(v[1])
        doc_dense_arr = np.asarray(dense_list, dtype=np.float32)

    # Query is small - never cache it.
    qry_enc = embedder.encode([query], return_sparse=True)

    dense_top, dense_scores = _dense_rank(
        qry_enc["dense_vecs"], doc_dense_arr, ids, top_n,
    )
    sparse_top, sparse_scores = _sparse_rank(
        qry_enc["lexical_weights"][0], sparse_list, ids, top_n,
    )
    fused = rrf_fuse([dense_top, sparse_top], k=rrf_k)[:top_n]

    by_id = {c.id: c for c in chunks}
    out: List[Chunk] = []
    for did, score in fused:
        c = by_id[did]
        c.scores["rrf"]    = score
        c.scores["dense"]  = dense_scores.get(did, 0.0)
        c.scores["sparse"] = sparse_scores.get(did, 0.0)
        out.append(c)
    log.info("retrieved top-%d candidates via RRF", len(out))
    return out
