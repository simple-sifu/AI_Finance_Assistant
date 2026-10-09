"""Alpha Vantage quote client with a TTL cache, call budget, and fallback chain.

Lookup order: fresh cache -> live GLOBAL_QUOTE (only if the budget allows;
live calls wait to stay just over 1 s apart) -> stale cache -> bundled mock.
Quota, network, timeout, and rate-limit problems never raise; they fall
through the chain. A live "unknown symbol" answer raises
``SymbolNotFoundError`` (cached for the TTL) and is deliberately not masked
with mock data.

Streamlit runs each session in its own thread and calls ``asyncio.run`` per
interaction, so nothing here outlives one call: each live request uses a
short-lived ``httpx.AsyncClient``, and shared state uses ``threading`` locks.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx

from ..config import Settings, get_settings
from .budget import CallBudget
from .cache import TTLCache
from .mock import get_mock_quote
from .models import Quote, QuoteUnavailableError, SymbolNotFoundError

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
HTTP_TIMEOUT_SECONDS = 10.0
_SYMBOL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]{0,9}")
_LOCK_POLL_SECONDS = 0.01


class _LiveFailure(Exception):
    """Internal: the live call failed in a way the fallback chain absorbs."""


class _RedactApiKey(logging.Filter):
    """Strip ``apikey=...`` from httpx's request log lines so the key is never logged."""

    _pattern = re.compile(r"(apikey=)[^&\s\"']+", re.IGNORECASE)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "apikey=" in message.lower():
            record.msg = self._pattern.sub(r"\1***", message)
            record.args = ()
        return True


def _install_log_redaction() -> None:
    for name in ("httpx", "httpcore"):
        log = logging.getLogger(name)
        if not any(isinstance(f, _RedactApiKey) for f in log.filters):
            log.addFilter(_RedactApiKey())


_install_log_redaction()


def normalize_symbol(symbol: str) -> str:
    """Validate and upper-case a ticker; raise ValueError on anything else."""
    if not isinstance(symbol, str):
        raise ValueError("symbol must be a string")
    cleaned = symbol.strip()
    if not _SYMBOL_RE.fullmatch(cleaned):
        raise ValueError(f"invalid symbol: {symbol!r} (expected 1-10 of A-Z, 0-9, '.', '-', starting with a letter or digit)")
    return cleaned.upper()


def _parse_global_quote(symbol: str, payload: dict[str, Any], fetched_at: datetime) -> Quote:
    try:
        return Quote(
            symbol=str(payload.get("01. symbol") or symbol).upper(),
            open=float(payload["02. open"]),
            high=float(payload["03. high"]),
            low=float(payload["04. low"]),
            price=float(payload["05. price"]),
            volume=int(payload["06. volume"]),
            latest_trading_day=date.fromisoformat(payload["07. latest trading day"]),
            previous_close=float(payload["08. previous close"]),
            change=float(payload["09. change"]),
            change_percent=float(str(payload["10. change percent"]).strip().rstrip("%")),
            source="live",
            fetched_at=fetched_at,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _LiveFailure(f"unexpected GLOBAL_QUOTE shape ({type(exc).__name__})") from exc


class MarketDataClient:
    """Quote source that never goes blank during a demo. Safe across threads and event loops."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache: TTLCache[Quote] | None = None,
        budget: CallBudget | None = None,
        clock: Callable[[], float] = time.time,
        base_url: str = ALPHA_VANTAGE_URL,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        self._cache: TTLCache[Quote] = (
            cache
            if cache is not None
            else TTLCache(self._settings.quote_cache_ttl_seconds, clock=clock)
        )
        self._budget = budget if budget is not None else CallBudget(clock=clock)
        self._base_url = base_url
        self._symbol_locks: dict[str, threading.Lock] = {}
        self._symbol_locks_guard = threading.Lock()

        if not self._settings.alpha_vantage_api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY is not set; serving mock quotes only.")
        elif self._settings.market_data_mode == "mock":
            logger.info("MARKET_DATA_MODE=mock; serving mock quotes only.")

    @property
    def live_enabled(self) -> bool:
        return bool(self._settings.alpha_vantage_api_key) and self._settings.market_data_mode != "mock"

    @property
    def budget(self) -> CallBudget:
        return self._budget

    @property
    def cache(self) -> TTLCache[Quote]:
        return self._cache

    async def get_quote(self, symbol: str) -> Quote:
        """Return a quote for ``symbol``; see the module docstring for the chain.

        Raises ValueError (bad input), SymbolNotFoundError (live says unknown),
        or QuoteUnavailableError (nothing live, cached, or mocked).
        """
        sym = normalize_symbol(symbol)
        if not self.live_enabled:
            return self._mock_or_raise(sym)

        hit = self._fresh(sym)
        if hit is not None:
            return hit

        lock = self._lock_for(sym)
        await self._acquire(lock)
        try:
            # Another caller may have filled the cache while we waited.
            hit = self._fresh(sym)
            if hit is not None:
                return hit
            if await self._budget.acquire():
                try:
                    quote = await self._fetch_live(sym)
                except SymbolNotFoundError:
                    self._cache.set_not_found(sym)
                    raise
                except _LiveFailure as exc:
                    logger.warning("Live quote for %s failed: %s; using fallback.", sym, exc)
                else:
                    self._cache.set(sym, quote)
                    return quote
            else:
                logger.info("Alpha Vantage budget exhausted or paused; using fallback for %s.", sym)
        finally:
            lock.release()

        return self._fallback(sym)

    # -- chain steps ------------------------------------------------------

    def _fresh(self, sym: str) -> Quote | None:
        entry = self._cache.get(sym)
        if entry is None:
            return None
        if entry.is_not_found:
            raise SymbolNotFoundError(sym)
        return dataclasses.replace(entry.value, source="cache")

    def _fallback(self, sym: str) -> Quote:
        entry = self._cache.get(sym, allow_expired=True)
        if entry is not None and not entry.is_not_found:
            return dataclasses.replace(entry.value, source="stale_cache")
        return self._mock_or_raise(sym)

    def _mock_or_raise(self, sym: str) -> Quote:
        quote = get_mock_quote(sym, self._now())
        if quote is None:
            raise QuoteUnavailableError(sym)
        return quote

    async def _fetch_live(self, sym: str) -> Quote:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": sym,
            "apikey": self._settings.alpha_vantage_api_key or "",
        }
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as http:
                response = await http.get(self._base_url, params=params)
        except httpx.HTTPError as exc:
            # str(exc) can embed the request URL (and so the key); log the type only.
            raise _LiveFailure(f"network error ({type(exc).__name__})") from None

        if response.status_code != 200:
            raise _LiveFailure(f"HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise _LiveFailure("response was not JSON") from None
        if not isinstance(body, dict):
            raise _LiveFailure("response JSON was not an object")

        if "Note" in body or "Information" in body:
            self._budget.pause()
            raise _LiveFailure("Alpha Vantage rate-limit response; live calls paused 60 s")
        if "Error Message" in body:
            raise _LiveFailure("Alpha Vantage error response")

        payload = body.get("Global Quote")
        if payload == {}:
            raise SymbolNotFoundError(sym)
        if not isinstance(payload, dict):
            raise _LiveFailure("response had no 'Global Quote'")
        return _parse_global_quote(sym, payload, self._now())

    # -- helpers ----------------------------------------------------------

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), UTC)

    def _lock_for(self, sym: str) -> threading.Lock:
        with self._symbol_locks_guard:
            lock = self._symbol_locks.get(sym)
            if lock is None:
                lock = self._symbol_locks[sym] = threading.Lock()
            return lock

    @staticmethod
    async def _acquire(lock: threading.Lock) -> None:
        """Acquire a threading lock without blocking the event loop.

        Polls with a non-blocking acquire instead of ``asyncio.to_thread(lock.acquire)``:
        if the awaiting task were cancelled, the worker thread would still take the
        lock afterwards and nobody would release it. Polling is cancel-safe and works
        across threads and event loops.
        """
        while not lock.acquire(blocking=False):
            await asyncio.sleep(_LOCK_POLL_SECONDS)


_default_client: MarketDataClient | None = None
_default_client_lock = threading.Lock()


def get_client() -> MarketDataClient:
    """Return the process-wide client (created on first use from ``get_settings()``)."""
    global _default_client
    with _default_client_lock:
        if _default_client is None:
            _default_client = MarketDataClient()
        return _default_client


def reset_client() -> None:
    """Drop the process-wide client (its cache and budget); mainly for tests."""
    global _default_client
    with _default_client_lock:
        _default_client = None


async def get_quote(symbol: str) -> Quote:
    """Return a quote for ``symbol`` using the process-wide client."""
    return await get_client().get_quote(symbol)
