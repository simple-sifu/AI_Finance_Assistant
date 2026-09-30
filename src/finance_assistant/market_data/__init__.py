"""Market data: stock quotes that keep working when Alpha Vantage is unavailable."""

from __future__ import annotations

from .client import MarketDataClient, get_client, get_quote, normalize_symbol, reset_client
from .models import (
    MarketDataError,
    Quote,
    QuoteSource,
    QuoteUnavailableError,
    SymbolNotFoundError,
)

__all__ = [
    "MarketDataClient",
    "MarketDataError",
    "Quote",
    "QuoteSource",
    "QuoteUnavailableError",
    "SymbolNotFoundError",
    "get_client",
    "get_quote",
    "normalize_symbol",
    "reset_client",
]
