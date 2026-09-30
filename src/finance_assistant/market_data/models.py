"""Quote data model and market-data errors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

QuoteSource = Literal["live", "cache", "stale_cache", "mock"]
QUOTE_SOURCES: frozenset[str] = frozenset({"live", "cache", "stale_cache", "mock"})


@dataclass(frozen=True)
class Quote:
    """One stock quote. ``source`` says where it came from so the UI can label it.

    ``change_percent`` is in percent units: -0.1842 means -0.1842 %.
    ``fetched_at`` is the timezone-aware UTC time the data was obtained.
    """

    symbol: str
    price: float
    open: float
    high: float
    low: float
    previous_close: float
    change: float
    change_percent: float
    volume: int
    latest_trading_day: date
    source: QuoteSource
    fetched_at: datetime

    def __post_init__(self) -> None:
        if self.source not in QUOTE_SOURCES:
            raise ValueError(f"invalid quote source: {self.source!r}")


class MarketDataError(Exception):
    """Base class for market-data errors callers may want to handle."""


class SymbolNotFoundError(MarketDataError, LookupError):
    """Alpha Vantage returned no quote for the symbol (e.g. a typo like ``APPL``)."""

    def __init__(self, symbol: str) -> None:
        super().__init__(f"No quote found for symbol {symbol!r}")
        self.symbol = symbol


class QuoteUnavailableError(MarketDataError):
    """No live, cached, or mock quote is available for the symbol right now."""

    def __init__(self, symbol: str) -> None:
        super().__init__(f"No quote available for symbol {symbol!r} right now")
        self.symbol = symbol
