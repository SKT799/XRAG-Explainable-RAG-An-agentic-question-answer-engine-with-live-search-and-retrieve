# storage

Just one file in here, `cache.py`. It is the cache layer used by the scraper.

`Cache` tries Redis first. If Redis is not running it transparently falls back to a thread-safe dict with TTL. Same interface either way:

```python
from app.storage.cache import Cache
c = Cache()
c.set("k", "hello", ttl=10)
c.get("k")   # "hello", and None after 10 seconds
```

## Why two backends

In production you run Redis as a container. Multiple workers share the cache and it survives a restart.

On a fresh Colab the install step that brings up `redis-server` may not have run yet, but the pipeline still has to work. The dict fallback covers that gap. You will never have to think about which one is active.
