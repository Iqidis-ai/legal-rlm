"""Caching utilities for improved performance.

Provides LRU caching for documents, API responses, and search results.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, TypeVar, Generic, Callable
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import os


T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self):
        """Record a cache hit."""
        self.hit_count += 1


class LRUCache(Generic[T]):
    """Simple LRU cache with TTL support."""

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: Optional[int] = None,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry[T]] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        if key not in self._cache:
            return None

        entry = self._cache[key]

        if entry.is_expired:
            self._remove(key)
            return None

        entry.touch()
        self._move_to_front(key)
        return entry.value

    def set(self, key: str, value: T, size_bytes: int = 0):
        """Set item in cache."""
        # Check if we need to evict
        while len(self._cache) >= self.max_size:
            self._evict_oldest()

        expires_at = None
        if self.ttl_seconds:
            expires_at = datetime.now() + timedelta(seconds=self.ttl_seconds)

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size_bytes=size_bytes,
        )

        self._cache[key] = entry
        self._access_order.insert(0, key)

    def _move_to_front(self, key: str):
        """Move key to front of access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.insert(0, key)

    def _evict_oldest(self):
        """Evict least recently used item."""
        if self._access_order:
            oldest_key = self._access_order.pop()
            self._cache.pop(oldest_key, None)

    def _remove(self, key: str):
        """Remove item from cache."""
        self._cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self._cache.values())
        total_size = sum(e.size_bytes for e in self._cache.values())

        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "total_size_bytes": total_size,
            "ttl_seconds": self.ttl_seconds,
        }


class DiskCache:
    """Persistent disk-based cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        max_size_mb: int = 100,
        ttl_seconds: Optional[int] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self._index_file = self.cache_dir / "_index.json"
        self._load_index()
        self.cleanup_expired()

    def _load_index(self):
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = {}
        else:
            self._index = {}

    def cleanup_expired(self):
        """Remove all entries whose TTL has expired."""
        if not self.ttl_seconds:
            return
        now = datetime.now()
        expired_keys = [
            safe_key for safe_key, entry in self._index.items()
            if datetime.fromisoformat(entry["created_at"]) + timedelta(seconds=self.ttl_seconds) < now
        ]
        for safe_key in expired_keys:
            self._remove(safe_key)

    def _save_index(self):
        """Save cache index to disk."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)

    def _make_key(self, key: str) -> str:
        """Create safe filename from key."""
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        safe_key = self._make_key(key)

        if safe_key not in self._index:
            return None

        entry = self._index[safe_key]

        # Check TTL
        if self.ttl_seconds:
            created = datetime.fromisoformat(entry["created_at"])
            if datetime.now() > created + timedelta(seconds=self.ttl_seconds):
                self._remove(safe_key)
                return None

        # Read from disk
        cache_file = self.cache_dir / f"{safe_key}.json"
        if not cache_file.exists():
            self._remove(safe_key)
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data.get("value")
        except (json.JSONDecodeError, IOError):
            self._remove(safe_key)
            return None

    def set(self, key: str, value: Any):
        """Set item in disk cache."""
        safe_key = self._make_key(key)

        # Check size limit
        self._enforce_size_limit()

        # Write to disk
        cache_file = self.cache_dir / f"{safe_key}.json"
        data = {
            "key": key,
            "value": value,
            "created_at": datetime.now().isoformat(),
        }

        with open(cache_file, "w") as f:
            json.dump(data, f)

        # Update index
        self._index[safe_key] = {
            "key": key,
            "created_at": datetime.now().isoformat(),
            "size_bytes": cache_file.stat().st_size,
        }
        self._save_index()

    def _remove(self, safe_key: str):
        """Remove item from cache."""
        cache_file = self.cache_dir / f"{safe_key}.json"
        if cache_file.exists():
            cache_file.unlink()
        self._index.pop(safe_key, None)
        self._save_index()

    def _enforce_size_limit(self):
        """Evict oldest items if cache exceeds size limit."""
        self.cleanup_expired()
        total_size = sum(e.get("size_bytes", 0) for e in self._index.values())
        max_bytes = self.max_size_mb * 1024 * 1024

        if total_size < max_bytes:
            return

        # Sort by creation time and remove oldest
        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: x[1].get("created_at", ""),
        )

        while total_size > max_bytes * 0.8 and sorted_entries:
            safe_key, _ = sorted_entries.pop(0)
            size = self._index.get(safe_key, {}).get("size_bytes", 0)
            self._remove(safe_key)
            total_size -= size

    def clear(self):
        """Clear all cache entries."""
        for safe_key in list(self._index.keys()):
            self._remove(safe_key)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(e.get("size_bytes", 0) for e in self._index.values())

        return {
            "entries": len(self._index),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_size_mb": self.max_size_mb,
            "ttl_seconds": self.ttl_seconds,
        }


class ResponseCache:
    """Cache for LLM API responses."""

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        self._cache = LRUCache[str](max_size=max_size, ttl_seconds=ttl_seconds)

    def _make_key(self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response."""
        key = self._make_key(prompt, model)
        return self._cache.get(key)

    def set(self, prompt: str, model: str, response: str):
        """Cache a response."""
        key = self._make_key(prompt, model)
        self._cache.set(key, response, size_bytes=len(response))

    def clear(self):
        """Clear the response cache."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


def cached(cache: LRUCache, key_fn: Callable[..., str]):
    """Decorator for caching function results."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            key = key_fn(*args, **kwargs)
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            result = fn(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper
    return decorator
