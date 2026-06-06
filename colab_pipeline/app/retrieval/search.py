"""
Block 4 · Web search.

We define a small `SearchProvider` Protocol so every implementation has the same
shape: `search(query, k) -> List[SearchResult]`. Two providers are bundled:

  * DDGSearch  -  uses the `ddgs` (DuckDuckGo) library. NO API key, works on Colab.
  * SearxngSearch  -  hits a self-hosted SearXNG container's JSON API (production).

Pick the active provider via `config.yaml`:
    search:
      provider: ddgs        # or "searxng"
"""
from __future__ import annotations

import random
import time
from typing import List, Protocol

from app.schemas import SearchResult
from app.util import get_logger

log = get_logger(__name__)


class SearchProvider(Protocol):
    """Anything that can take a query and return a list of links."""

    def search(self, query: str, k: int = 10) -> List[SearchResult]: ...


# ---------------------------------------------------------------------------
# DuckDuckGo via `ddgs` (Colab default  -  no API key)
# ---------------------------------------------------------------------------
class DDGSearch:
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Run a DuckDuckGo query with light rate-limit handling.

        DDG aggressively throttles back-to-back unauth requests; the symptom
        is a sudden empty result list rather than an HTTP error. We retry
        once after a jittered backoff and add a small jitter between calls
        so multi-search fan-out doesn't trip the limiter on the 2nd / 3rd
        sub-query.
        """
        try:
            from ddgs import DDGS                  # lazy import
        except Exception as e:
            log.error("ddgs not installed (%s). pip install ddgs", e)
            return []

        def _run() -> List[SearchResult]:
            out: List[SearchResult] = []
            with DDGS() as ddg:
                for r in ddg.text(query, max_results=k):
                    out.append(SearchResult(
                        url=r.get("href") or r.get("url") or "",
                        title=r.get("title", ""),
                        snippet=r.get("body", "") or r.get("snippet", ""),
                    ))
            return out

        try:
            results = _run()
        except Exception as e:
            log.warning("ddgs first attempt failed (%s); retrying after backoff", e)
            time.sleep(random.uniform(0.4, 1.0))
            try:
                results = _run()
            except Exception as e2:
                log.error("ddgs retry failed (%s) - returning empty", e2)
                return []
        if not results:
            log.info("ddgs empty for %r; retrying after backoff (rate-limit?)", query)
            time.sleep(random.uniform(0.4, 1.0))
            try:
                results = _run()
            except Exception as e:
                log.warning("ddgs retry failed (%s)", e)
        log.info("ddgs: %d results for %r", len(results), query)
        return results


# ---------------------------------------------------------------------------
# SearXNG (production  -  needs the Docker container)
# ---------------------------------------------------------------------------
class SearxngSearch:
    def __init__(self, base_url: str | None = None) -> None:
        from config.settings import settings
        self.base_url = base_url or settings.search.searxng_url

    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        try:
            import httpx                            # lazy
        except Exception as e:
            log.error("httpx not installed (%s)", e)
            return []
        try:
            r = httpx.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json"},
                timeout=10.0,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("SearXNG fetch failed: %s  -  returning empty list", e)
            return []
        out: List[SearchResult] = []
        for item in (data.get("results") or [])[:k]:
            out.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", "") or item.get("snippet", ""),
            ))
        log.info("searxng: %d results for %r", len(out), query)
        return out


# ---------------------------------------------------------------------------
# Factory + helpers
# ---------------------------------------------------------------------------
def get_search_provider() -> SearchProvider:
    """Pick the provider based on config (default: ddgs)."""
    from config.settings import settings
    if settings.search.provider == "searxng":
        return SearxngSearch()
    return DDGSearch()


def dedupe_results(results: List[SearchResult]) -> List[SearchResult]:
    """Dedupe across sub-queries by URL, preserving first-seen order."""
    seen, out = set(), []
    for r in results:
        if r.url and r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return out


def multi_search(sub_queries: List[str], k_per_query: int = 10) -> List[SearchResult]:
    """Run each sub-query through the provider, merge + dedupe across them.

    For the DDG provider we add a small jittered pause between consecutive
    sub-queries so the unauthenticated rate limiter doesn't drop the 2nd /
    3rd query (a common cause of "search returned nothing" for multi-part
    questions).
    """
    provider = get_search_provider()
    is_ddg = isinstance(provider, DDGSearch)
    merged: List[SearchResult] = []
    for i, q in enumerate(sub_queries):
        if is_ddg and i > 0:
            time.sleep(random.uniform(0.2, 0.5))
        merged.extend(provider.search(q, k=k_per_query))
    return dedupe_results(merged)
