"""TTLCache: freshness, stale reads, not-found markers, thread safety."""

from __future__ import annotations

import threading

import pytest

from finance_assistant.market_data.cache import NOT_FOUND, TTLCache

from .conftest import FakeClock


def test_fresh_entry_is_returned(clock: FakeClock) -> None:
    cache: TTLCache[str] = TTLCache(1800, clock=clock)
    cache.set("SPY", "q1")
    clock.advance(1799)
    entry = cache.get("SPY")
    assert entry is not None and entry.value == "q1" and not entry.expired


def test_missing_key_returns_none(clock: FakeClock) -> None:
    assert TTLCache(1800, clock=clock).get("SPY") is None


def test_expired_entry_hidden_unless_allow_expired(clock: FakeClock) -> None:
    cache: TTLCache[str] = TTLCache(1800, clock=clock)
    cache.set("SPY", "q1")
    clock.advance(1800)
    assert cache.get("SPY") is None
    stale = cache.get("SPY", allow_expired=True)
    assert stale is not None and stale.value == "q1" and stale.expired


def test_set_refreshes_timestamp(clock: FakeClock) -> None:
    cache: TTLCache[str] = TTLCache(1800, clock=clock)
    cache.set("SPY", "old")
    clock.advance(2000)
    cache.set("SPY", "new")
    entry = cache.get("SPY")
    assert entry is not None and entry.value == "new"


def test_not_found_marker(clock: FakeClock) -> None:
    cache: TTLCache[str] = TTLCache(1800, clock=clock)
    cache.set_not_found("APPL")
    entry = cache.get("APPL")
    assert entry is not None and entry.is_not_found and entry.value is NOT_FOUND
    clock.advance(1800)
    assert cache.get("APPL") is None


def test_rejects_non_positive_ttl() -> None:
    with pytest.raises(ValueError):
        TTLCache(0)


def test_concurrent_writes_and_reads(clock: FakeClock) -> None:
    cache: TTLCache[int] = TTLCache(1800, clock=clock)

    def worker(n: int) -> None:
        for i in range(200):
            cache.set(f"S{n}-{i % 10}", i)
            cache.get(f"S{n}-{i % 10}", allow_expired=True)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(cache) == 80
