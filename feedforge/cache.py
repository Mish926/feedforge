"""Cache abstraction for serving.

Two backends behind one interface. InMemoryCache (TTL dict, LRU-ish
eviction) means the demo runs with zero infrastructure; RedisCache is a
drop-in when REDIS_URL is set, which is what a real deployment uses so
cached user state survives restarts and is shared across replicas.

get_cache() picks Redis when REDIS_URL is set and reachable, otherwise
falls back to memory and says so. Values are JSON-serializable.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


class InMemoryCache:
    backend = "memory"

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 10000):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires, value = entry
        if expires < time.time():
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self.max_entries:
            # Drop the entry closest to expiry; O(n) but only at capacity.
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]
        self._store[key] = (time.time() + self.ttl, value)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"backend": self.backend, "entries": len(self._store),
                "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0}


class RedisCache:
    backend = "redis"

    def __init__(self, url: str, ttl_seconds: int = 300):
        import redis  # imported lazily so redis isn't a hard dependency

        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.client.ping()
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        raw = self.client.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any) -> None:
        self.client.setex(key, self.ttl, json.dumps(value))

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"backend": self.backend, "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0}


def get_cache(ttl_seconds: int = 300):
    url = os.environ.get("REDIS_URL")
    if url:
        try:
            cache = RedisCache(url, ttl_seconds)
            print(f"[cache] redis at {url}")
            return cache
        except Exception as e:  # noqa: BLE001 - fall back rather than fail startup
            print(f"[cache] redis unavailable ({e}); using in-memory cache")
    else:
        print("[cache] REDIS_URL not set; using in-memory cache")
    return InMemoryCache(ttl_seconds)
