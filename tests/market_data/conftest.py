"""Fixtures for market-data tests: a fake clock and a real-shaped SPY response."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

# 2026-09-30 14:00:00 UTC
START = datetime(2026, 9, 30, 14, 0, tzinfo=UTC).timestamp()

TEST_API_KEY = "TESTKEY-DO-NOT-LOG-123"

# Shape of the real Alpha Vantage GLOBAL_QUOTE response for SPY on 2026-09-29.
SPY_GLOBAL_QUOTE = {
    "Global Quote": {
        "01. symbol": "SPY",
        "02. open": "765.1400",
        "03. high": "769.0600",
        "04. low": "760.9900",
        "05. price": "764.2000",
        "06. volume": "61234567",
        "07. latest trading day": "2026-09-29",
        "08. previous close": "765.6100",
        "09. change": "-1.4100",
        "10. change percent": "-0.1842%",
    }
}


def global_quote_for(symbol: str, price: str = "100.0000") -> dict:
    body = {"Global Quote": dict(SPY_GLOBAL_QUOTE["Global Quote"])}
    body["Global Quote"]["01. symbol"] = symbol
    body["Global Quote"]["05. price"] = price
    return body


class FakeClock:
    """Thread-safe controllable epoch clock."""

    def __init__(self, start: float = START) -> None:
        self._now = start
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds

    def set(self, value: float) -> None:
        with self._lock:
            self._now = value

    async def sleep(self, seconds: float) -> None:
        """Stand-in for ``asyncio.sleep``: records the wait and advances the clock instantly."""
        self.sleeps.append(seconds)
        self.advance(seconds)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
