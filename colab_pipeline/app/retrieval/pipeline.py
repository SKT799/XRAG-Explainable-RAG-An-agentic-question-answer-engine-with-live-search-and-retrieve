"""
Retrieval pipeline. Chains Blocks 4 -> 5 -> 6 -> 7 -> 8 into one function.

Input:  sub_queries (from the rewriter, Block 3)
Output: top-K provenance-tagged, CE-scored `Chunk`s ready for the generator (Block 9)

The orchestrator (`app/orchestrator/engine.py`) typically calls this once per
request and hands the result to the generator. It can also pass an `on_step`
callback so a UI can show progress between the five sub-steps.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from app.retrieval.embed_retrieve import retrieve_candidates
from app.retrieval.pdf import PDFExtractionError, chunk_pdf
from app.retrieval.preprocess import clean_and_chunk
from app.retrieval.rerank import rerank
from app.retrieval.scrape import fetch_many_sync
from app.retrieval.search import multi_search
from app.schemas import Chunk
from app.util import get_logger, timed

log = get_logger(__name__)

# Same shape as engine.OnStep: (stage_key, status, payload | None)
OnStep = Optional[Callable[[str, str, Optional[dict]], None]]


def _noop(*args, **kwargs):
    pass


def retrieve(sub_queries: List[str], top_k: int | None = None,
             on_step: OnStep = None,
             standalone_query: str | None = None) -> List[Chunk]:
    """The whole retrieval spine.

    Returns the top-K chunks the generator will see.

    Stages:
        4) multi_search(sub_queries)        -> list[SearchResult]
        5) fetch_many_sync(urls)            -> {url -> cleaned text}
        6) clean_and_chunk(...)             -> list[Chunk] with Provenance
        7) retrieve_candidates(query, ...)  -> top-N candidate Chunks (RRF)
        8) rerank(query, ...)               -> top-K (cross-encoder scored)

    `sub_queries` are used for SEARCH fan-out (wider recall). The unified
    `standalone_query` is used for RANKING (embed + rerank), so multi-part
    questions like "who won X and who scored most goals" don't accidentally
    rank only against the first sub-query. Falls back to `sub_queries[0]`
    when `standalone_query` is None (older callers).
    """
    from config.settings import settings
    on_step = on_step or _noop

    if not sub_queries:
        return []
    top_k = top_k or settings.reranking.top_k_keep
    main_query = (standalone_query or sub_queries[0]).strip() or sub_queries[0]

    # 4) search
    on_step("search", "start", None)
    with timed("search", log):
        results = multi_search(sub_queries, k_per_query=settings.search.k_per_query)
    on_step("search", "done", {
        "url_count": len(results),
        "sub_query_count": len(sub_queries),
        "sample_urls": [r.url for r in results[:5]],
    })
    if not results:
        log.warning("no search results, returning empty list")
        return []

    # 5) scrape
    on_step("scrape", "start", None)
    with timed("scrape", log):
        urls = [r.url for r in results]
        url_to_text = fetch_many_sync(urls)
    on_step("scrape", "done", {
        "page_count": len(url_to_text),
        "attempted": len(urls),
        "skipped": len(urls) - len(url_to_text),
    })

    # 6) preprocess (clean + chunk)
    on_step("chunk", "start", None)
    with timed("chunk", log):
        chunks: List[Chunk] = []
        title_by_url = {r.url: r.title for r in results}
        for url, text in url_to_text.items():
            chunks.extend(clean_and_chunk(text, url=url, title=title_by_url.get(url, "")))
    log.info("total chunks across all pages: %d", len(chunks))
    on_step("chunk", "done", {"chunk_count": len(chunks)})
    if not chunks:
        return []

    # 7) embed + RRF top-N candidates
    on_step("embed", "start", None)
    with timed("embed_retrieve", log):
        candidates = retrieve_candidates(main_query, chunks,
                                         top_n=settings.embedding.top_n_candidates)
    on_step("embed", "done", {
        "candidate_count": len(candidates),
        "from_chunks": len(chunks),
    })

    # 8) cross-encoder rerank -> top-K
    on_step("rerank", "start", None)
    with timed("rerank", log):
        top = rerank(main_query, candidates, top_k=top_k)
    on_step("rerank", "done", {
        "top_count": len(top),
        "best_ce": round(top[0].scores.get("ce", 0.0), 3) if top else None,
    })
    return top


# ===========================================================================
# PDF offline mode. No search, no scrape. Just parse, chunk, embed, rerank.
# Used when QueryRequest.mode == "pdf_offline" and a pdf_path is provided.
# ===========================================================================
def retrieve_from_pdf(query: str, pdf_path: str,
                      top_k: int | None = None,
                      on_step: OnStep = None) -> List[Chunk]:
    """Same shape as `retrieve()` but reads chunks from a PDF instead of the web.

    Emits these stage events (so the Live process panel can show progress):
        pdf_parse, chunk, embed, rerank

    Every Chunk it returns has a `Locator(page_start=N, page_end=N)` so the
    final response cites by page number.
    """
    from config.settings import settings
    on_step = on_step or _noop
    top_k = top_k or settings.reranking.top_k_keep

    # PDF parse + page-aware chunk (one combined step, two events).
    on_step("pdf_parse", "start", None)
    try:
        with timed("pdf_parse", log):
            chunks = chunk_pdf(pdf_path)
    except PDFExtractionError as e:
        on_step("pdf_parse", "done",
                {"chunk_count": 0, "error": e.kind, "message": str(e)})
        log.warning("PDF extraction failed (%s): %s", e.kind, e)
        # Re-raise so engine.run can render a specific message; orchestrators
        # without specific handling will see an empty `chunks` shortly anyway.
        raise
    if not chunks:
        on_step("pdf_parse", "done", {"chunk_count": 0, "pages": 0})
        log.warning("PDF returned 0 chunks: %s", pdf_path)
        return []
    # Find page-number range from the chunks' locators.
    pages = {c.provenance.locator.page_start for c in chunks
             if c.provenance.locator.page_start is not None}
    on_step("pdf_parse", "done", {
        "chunk_count": len(chunks),
        "page_count": len(pages),
        "pages": f"{min(pages)}-{max(pages)}" if pages else "0",
    })

    # 7) embed + RRF top-N candidates
    on_step("embed", "start", None)
    with timed("embed_retrieve", log):
        candidates = retrieve_candidates(query, chunks,
                                         top_n=settings.embedding.top_n_candidates)
    on_step("embed", "done", {
        "candidate_count": len(candidates),
        "from_chunks": len(chunks),
    })

    # 8) cross-encoder rerank -> top-K
    on_step("rerank", "start", None)
    with timed("rerank", log):
        top = rerank(query, candidates, top_k=top_k)
    on_step("rerank", "done", {
        "top_count": len(top),
        "best_ce": round(top[0].scores.get("ce", 0.0), 3) if top else None,
    })
    return top
