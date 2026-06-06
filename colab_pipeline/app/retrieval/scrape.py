"""
Block 5 · Scrape + cache.

Async worker pool:
  * httpx GET each URL concurrently under a per-task timeout.
  * Cache **raw HTML** by sha1(url) with `cache_ttl_sec` TTL. We cache HTML
    (not cleaned text) so the downstream boilerplate filter in
    `preprocess.html_to_clean_text` can do its link-density + text-density
    job. Previously we cached the result of a naive `soup.get_text()` which
    left navbars, cookie banners, related-posts grids, etc. in the corpus.
  * On bad/empty responses we just skip  -  one bad page never stalls the request.

Tier-2 fallback to Playwright is left as an optional extension; production setups
swap it in behind the same `fetch_many` interface.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List

from app.storage.cache import Cache
from app.util import get_logger, sha1

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# HTML → readable text (lazy BeautifulSoup import)
# ---------------------------------------------------------------------------
def html_to_text(html: str) -> str:
    """Strip <script>/<style>/<nav>/<footer>/<header>, return joined text."""
    try:
        from bs4 import BeautifulSoup
    except Exception as e:
        log.error("beautifulsoup4 not installed (%s)", e)
        return ""
    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Async fetcher
# ---------------------------------------------------------------------------
async def _fetch_one(client, url: str, timeout: float, user_agent: str) -> str:
    """Return RAW HTML for `url` or "" on failure.

    Returning raw HTML (instead of pre-cleaned text) lets the downstream
    `preprocess.clean_and_chunk` apply its boilerplate filter. Without this,
    `is_html = text.lstrip().startswith("<")` was always False on scraped
    text and the filter never ran, so menu links / footer / cookie banners
    leaked into chunks and citations.
    """
    try:
        r = await client.get(url, timeout=timeout, headers={"User-Agent": user_agent},
                             follow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            return ""                              # skip PDFs/images for the live-web path
        return r.text
    except Exception as e:
        log.info("scrape skip %s (%s)", url, e)
        return ""


async def fetch_many(urls: List[str]) -> Dict[str, str]:
    """
    Concurrently scrape `urls`. Returns {url → cleaned_text}.
    Cache HITs short-circuit the network call.
    """
    from config.settings import settings           # lazy so this file imports cheaply
    cache = Cache()
    cfg = settings.scrape

    out: Dict[str, str] = {}
    todo: List[str] = []
    for u in urls:
        cached = cache.get(sha1(u))
        if cached is not None:
            out[u] = cached
        else:
            todo.append(u)
    log.info("scrape: %d urls (%d cache hits, %d to fetch)", len(urls), len(urls) - len(todo), len(todo))

    if not todo:
        return out

    try:
        import httpx
    except Exception as e:
        log.error("httpx not installed (%s)", e)
        return out

    sem = asyncio.Semaphore(cfg.concurrency)
    async with httpx.AsyncClient() as client:
        async def bound(u: str):
            async with sem:
                return u, await _fetch_one(client, u, cfg.timeout_sec, cfg.user_agent)
        for coro in asyncio.as_completed([bound(u) for u in todo]):
            u, text = await coro
            if text:
                out[u] = text
                cache.set(sha1(u), text, ttl=cfg.cache_ttl_sec)
    return out


def fetch_many_sync(urls: List[str]) -> Dict[str, str]:
    """Synchronous wrapper for notebooks/tests.

    Notebook (Jupyter / Colab) compatibility: those environments already run
    an event loop, so plain `asyncio.run(...)` raises and the old
    `loop.run_until_complete(...)` fails with "this event loop is already
    running". We install `nest_asyncio` when we detect that case so the
    pipeline call works from a notebook cell.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop in this thread - fast path (FastAPI executor thread, scripts).
        return asyncio.run(fetch_many(urls))
    # A loop IS running (Jupyter/Colab/IPython). Patch it so nested
    # run_until_complete works, otherwise fall back to a fresh loop in a
    # dedicated thread so we don't deadlock the user's notebook.
    try:
        import nest_asyncio                         # lazy
        nest_asyncio.apply(loop)
        return loop.run_until_complete(fetch_many(urls))
    except Exception as e:
        log.info("nest_asyncio unavailable (%s) - running in a worker thread", e)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(asyncio.run, fetch_many(urls))
            return fut.result()
