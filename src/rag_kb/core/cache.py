"""LRU embedding cache with hit/miss counters."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors.

    Key: sha256(text), Value: list[float] (768-dim vector).
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, text: str, embedding: list[float]) -> None:
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = embedding
                return
            self._cache[key] = embedding
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
