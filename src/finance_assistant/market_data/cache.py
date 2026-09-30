"""Thread-safe in-process TTL cache with stale reads and not-found markers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

V = TypeVar("V")


class _NotFound:
    """Sentinel type: the upstream said this key does not exist."""

    _instance: _NotFound | None = None

    def __new__(cls) -> _NotFound:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NOT_FOUND"


NOT_FOUND = _NotFound()


@dataclass(frozen=True)
class CacheEntry(Generic[V]):
    """A cached value (or ``NOT_FOUND``) and whether it has outlived its TTL."""

    value: V | _NotFound
    stored_at: float
    expired: bool

    @property
    def is_not_found(self) -> bool:
        return self.value is NOT_FOUND


class TTLCache(Generic[V]):
    """Key/value cache whose entries go stale after ``ttl_seconds``.

    Expired entries are kept (not evicted) so they can be served as a last-resort
    fallback via ``get(key, allow_expired=True)``; a later ``set`` replaces them.
    """

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.time) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: dict[str, tuple[V | _NotFound, float]] = {}
        self._lock = threading.Lock()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def set(self, key: str, value: V) -> None:
        with self._lock:
            self._entries[key] = (value, self._clock())

    def set_not_found(self, key: str) -> None:
        with self._lock:
            self._entries[key] = (NOT_FOUND, self._clock())

    def get(self, key: str, *, allow_expired: bool = False) -> CacheEntry[V] | None:
        """Return the entry for ``key``; expired entries only if ``allow_expired``."""
        with self._lock:
            item = self._entries.get(key)
            now = self._clock()
        if item is None:
            return None
        value, stored_at = item
        expired = now - stored_at >= self._ttl
        if expired and not allow_expired:
            return None
        return CacheEntry(value=value, stored_at=stored_at, expired=expired)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
