"""Bundled mock quotes: the last link of the fallback chain, so demos never go blank."""

from __future__ import annotations

import json
from datetime import date, datetime
from functools import cache
from importlib import resources
from typing import Any

from .models import Quote

_MOCK_FILE = "mock_quotes.json"


@cache
def _load() -> dict[str, Any]:
    text = resources.files(__package__).joinpath(_MOCK_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def mock_symbols() -> frozenset[str]:
    """Symbols that have a bundled mock quote."""
    return frozenset(_load()["quotes"])


def get_mock_quote(symbol: str, fetched_at: datetime) -> Quote | None:
    """Return the bundled quote for an already-normalized ``symbol``, or None."""
    data = _load()
    row = data["quotes"].get(symbol)
    if row is None:
        return None
    return Quote(
        symbol=symbol,
        price=float(row["price"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        previous_close=float(row["previous_close"]),
        change=float(row["change"]),
        change_percent=float(row["change_percent"]),
        volume=int(row["volume"]),
        latest_trading_day=date.fromisoformat(data["latest_trading_day"]),
        source="mock",
        fetched_at=fetched_at,
    )
