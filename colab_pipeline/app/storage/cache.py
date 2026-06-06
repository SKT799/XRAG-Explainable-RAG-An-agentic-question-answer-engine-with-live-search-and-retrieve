"""
Cache (Block 5 substrate).

Two-tier cache for scraped pages:
  TIER 1  -  Redis if reachable           (production: shared, survives restarts)
  TIER 2  -  in-process dict with TTL     (fallback: works in any environment, incl. Colab
                                         before `redis-server` is installed)

Both tiers expose the SAME `Cache.get(key)` / `Cache.set(key, value, ttl)` interface, so
the rest of the code is oblivious to which one is active.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Optional

from app.util import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tier-2: dict-with-TTL fallback (pure stdlib  -  always works)
# ---------------------------------------------------------------------------
class _DictTTLCache:
    """Tiny thread-safe TTL cache. Good enough for one-process Colab dev.

    A periodic prune pass runs every `_prune_interval` seconds (lazily, on
    the next `set` after the deadline) to evict expired entries that no
    one ever reads back. Without it, anything `set` but never re-read leaks
    until process exit.
    """

    _prune_interval: float = 60.0

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}   # key → (expires_at, value)
        self._lock = Lock()
        self._next_prune: float = time.time() + self._prune_interval

    def _prune_locked(self) -> None:
        now = time.time()
        if now < self._next_prune:
            return
        dead = [k for k, (exp, _) in self._store.items() if exp < now]
        for k in dead:
            self._store.pop(k, None)
        self._next_prune = now + self._prune_interval

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            self._store[key] = (time.time() + ttl, value)
            self._prune_locked()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Public Cache: tries Redis first, transparently falls back
# ---------------------------------------------------------------------------
class Cache:
    """
    Usage:
        from app.storage.cache import Cache
        cache = Cache()                       # uses settings.storage.redis_url
        cache.set("k", "hello", ttl=10)
        cache.get("k")                        # → "hello"
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        # Local fallback is always present. We attempt to use Redis on top.
        self._fallback = _DictTTLCache()
        self._redis = None
        if redis_url is None:
            try:
                from config.settings import settings
                redis_url = settings.storage.redis_url
            except Exception:
                redis_url = None
        if redis_url:
            try:
                import redis  # lazy import  -  only if installed
                client = redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._redis = client
                log.info("cache: using Redis at %s", redis_url)
            except Exception as e:
                log.warning("cache: Redis unreachable (%s)  -  using dict-TTL fallback", e)

    # ---- public API ----
    def get(self, key: str) -> Optional[str]:
        if self._redis is not None:
            try:
                return self._redis.get(key)
            except Exception as e:
                log.warning("cache: redis.get failed (%s)  -  falling back", e)
        return self._fallback.get(key)

    def set(self, key: str, value: str, ttl: int) -> None:
        if self._redis is not None:
            try:
                self._redis.setex(key, ttl, value)
                return
            except Exception as e:
                log.warning("cache: redis.set failed (%s)  -  falling back", e)
        self._fallback.set(key, value, ttl)
