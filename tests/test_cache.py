"""Tests for the LRU embedding cache."""

from __future__ import annotations

import threading

import pytest

from rag_kb.core.cache import EmbeddingCache


def _vec(val: float = 1.0) -> list[float]:
    """Create a simple 3-dim embedding vector for testing."""
    return [val, val + 0.1, val + 0.2]


class TestEmbeddingCache:
    def test_cache_miss_returns_none(self) -> None:
        cache = EmbeddingCache(max_size=10)
        assert cache.get("hello") is None

    def test_cache_put_and_get(self) -> None:
        cache = EmbeddingCache(max_size=10)
        vec = _vec(1.0)
        cache.put("hello", vec)
        result = cache.get("hello")
        assert result == vec

    def test_cache_hit_increments(self) -> None:
        cache = EmbeddingCache(max_size=10)
        cache.put("hello", _vec())
        assert cache.hits == 0
        cache.get("hello")
        assert cache.hits == 1
        cache.get("hello")
        assert cache.hits == 2

    def test_cache_miss_increments(self) -> None:
        cache = EmbeddingCache(max_size=10)
        assert cache.misses == 0
        cache.get("miss1")
        assert cache.misses == 1
        cache.get("miss2")
        assert cache.misses == 2

    def test_cache_lru_eviction(self) -> None:
        cache = EmbeddingCache(max_size=2)
        cache.put("a", _vec(1.0))
        cache.put("b", _vec(2.0))
        cache.put("c", _vec(3.0))  # Should evict "a"

        assert cache.get("a") is None  # evicted
        assert cache.get("b") == _vec(2.0)
        assert cache.get("c") == _vec(3.0)

    def test_cache_lru_eviction_respects_access_order(self) -> None:
        cache = EmbeddingCache(max_size=2)
        cache.put("a", _vec(1.0))
        cache.put("b", _vec(2.0))
        # Access "a" to make it recently used
        cache.get("a")
        cache.put("c", _vec(3.0))  # Should evict "b" (least recently used)

        assert cache.get("a") == _vec(1.0)
        assert cache.get("b") is None  # evicted
        assert cache.get("c") == _vec(3.0)

    def test_cache_hit_rate(self) -> None:
        cache = EmbeddingCache(max_size=10)
        cache.put("a", _vec())
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_cache_hit_rate_empty(self) -> None:
        cache = EmbeddingCache(max_size=10)
        assert cache.hit_rate == 0.0

    def test_cache_clear(self) -> None:
        cache = EmbeddingCache(max_size=10)
        cache.put("a", _vec())
        cache.get("a")
        cache.get("miss")
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.get("a") is None

    def test_cache_size_property(self) -> None:
        cache = EmbeddingCache(max_size=10)
        assert cache.size == 0
        cache.put("a", _vec())
        assert cache.size == 1
        cache.put("b", _vec())
        assert cache.size == 2

    def test_cache_key_hashing(self) -> None:
        """Different texts produce different cache keys."""
        cache = EmbeddingCache(max_size=10)
        cache.put("hello", _vec(1.0))
        cache.put("world", _vec(2.0))
        assert cache.get("hello") == _vec(1.0)
        assert cache.get("world") == _vec(2.0)

    def test_cache_update_existing_key(self) -> None:
        """Putting the same text again updates the value."""
        cache = EmbeddingCache(max_size=10)
        cache.put("hello", _vec(1.0))
        cache.put("hello", _vec(2.0))
        assert cache.get("hello") == _vec(2.0)
        assert cache.size == 1

    def test_cache_thread_safety(self) -> None:
        """Basic thread safety: concurrent puts and gets don't crash."""
        cache = EmbeddingCache(max_size=100)
        errors: list[Exception] = []

        def worker(start: int) -> None:
            try:
                for i in range(start, start + 50):
                    cache.put(f"text-{i}", _vec(float(i)))
                    cache.get(f"text-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert cache.size <= 100
