"""MarketDataClient: every row of the story's I/O & edge-case matrix, plus concurrency."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, date, datetime

import httpx
import pytest
import respx

from finance_assistant.config import Settings
from finance_assistant.market_data import (
    MarketDataClient,
    QuoteUnavailableError,
    SymbolNotFoundError,
    get_quote,
)
from finance_assistant.market_data.client import ALPHA_VANTAGE_URL
from finance_assistant.market_data.mock import mock_symbols

from .conftest import SPY_GLOBAL_QUOTE, TEST_API_KEY, FakeClock, global_quote_for

BUNDLED = ["SPY", "VOO", "VTI", "QQQ", "BND", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


@pytest.fixture
def av():
    """respx router standing in for Alpha Vantage; unmatched requests fail."""
    with respx.mock(assert_all_called=False) as router:
        yield router


def quote_route(av: respx.MockRouter, symbol: str | None = None):
    params = {"function": "GLOBAL_QUOTE"}
    if symbol is not None:
        params["symbol"] = symbol
    return av.get(ALPHA_VANTAGE_URL, params=params)


@pytest.fixture
def make_client(clock: FakeClock):
    def factory(**settings_kwargs) -> MarketDataClient:
        settings_kwargs.setdefault("alpha_vantage_api_key", TEST_API_KEY)
        return MarketDataClient(Settings(**settings_kwargs), clock=clock)

    return factory


# -- happy path and cache ----------------------------------------------------


async def test_live_call_normalizes_symbol_and_parses_quote(av, make_client, clock) -> None:
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    client = make_client()

    quote = await client.get_quote("  spy ")

    assert route.call_count == 1
    request = route.calls.last.request
    assert request.url.params["symbol"] == "SPY"
    assert request.url.params["apikey"] == TEST_API_KEY
    assert quote.source == "live"
    assert quote.symbol == "SPY"
    assert quote.price == 764.2
    assert quote.change_percent == pytest.approx(-0.1842)
    assert quote.change == pytest.approx(-1.41)
    assert quote.previous_close == pytest.approx(765.61)
    assert quote.volume == 61234567
    assert quote.latest_trading_day == date(2026, 9, 29)
    assert quote.fetched_at == datetime.fromtimestamp(clock(), UTC)
    assert client.cache.get("SPY") is not None


async def test_fresh_cache_hit_makes_no_http_call(av, make_client, clock) -> None:
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    client = make_client()
    first = await client.get_quote("SPY")
    clock.advance(29 * 60)

    second = await client.get_quote("spy")

    assert route.call_count == 1
    assert second.source == "cache"
    assert second.price == first.price
    assert second.fetched_at == first.fetched_at


async def test_expired_cache_triggers_new_live_call(av, make_client, clock) -> None:
    route = quote_route(av)
    route.side_effect = [
        httpx.Response(200, json=SPY_GLOBAL_QUOTE),
        httpx.Response(200, json=global_quote_for("SPY", "770.0000")),
    ]
    client = make_client()
    await client.get_quote("SPY")
    clock.advance(30 * 60 + 1)

    refreshed = await client.get_quote("SPY")

    assert route.call_count == 2
    assert refreshed.source == "live"
    assert refreshed.price == 770.0
    cached = await client.get_quote("SPY")
    assert cached.source == "cache" and cached.price == 770.0


async def test_custom_ttl_from_settings(av, make_client, clock) -> None:
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    client = make_client(quote_cache_ttl_seconds=60)
    await client.get_quote("SPY")
    clock.advance(61)
    assert (await client.get_quote("SPY")).source == "live"
    assert route.call_count == 2


async def test_injected_empty_cache_is_kept(av, clock) -> None:
    from finance_assistant.market_data.cache import TTLCache

    quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    cache: TTLCache = TTLCache(1800, clock=clock)
    assert len(cache) == 0
    client = MarketDataClient(Settings(alpha_vantage_api_key=TEST_API_KEY), cache=cache, clock=clock)
    assert client.cache is cache
    await client.get_quote("SPY")
    assert cache.get("SPY") is not None


# -- budget ------------------------------------------------------------------


async def test_daily_budget_exhausted_falls_back_without_http(av, make_client, clock) -> None:
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    client = make_client()
    while client.budget.remaining_today:
        assert client.budget.try_acquire()
        clock.advance(61)

    quote = await client.get_quote("SPY")

    assert route.call_count == 0
    assert quote.source == "mock"
    assert quote.symbol == "SPY"


async def test_minute_budget_exhausted_then_recovers(av, make_client, clock) -> None:
    route = quote_route(av).mock(
        side_effect=lambda request: httpx.Response(
            200, json=global_quote_for(request.url.params["symbol"])
        )
    )
    client = make_client()
    for sym in ["VOO", "VTI", "QQQ", "BND", "AAPL"]:
        assert (await client.get_quote(sym)).source == "live"

    blocked = await client.get_quote("SPY")
    assert blocked.source == "mock"
    assert route.call_count == 5

    clock.advance(60)
    assert (await client.get_quote("SPY")).source == "live"
    assert route.call_count == 6


async def test_budget_exhausted_prefers_stale_cache_over_mock(av, make_client, clock) -> None:
    route = quote_route(av).respond(json=global_quote_for("SPY", "777.0000"))
    client = make_client()
    await client.get_quote("SPY")
    clock.advance(31 * 60)
    while client.budget.remaining_today:
        client.budget.try_acquire()
        clock.advance(61)

    quote = await client.get_quote("SPY")

    assert route.call_count == 1
    assert quote.source == "stale_cache"
    assert quote.price == 777.0


# -- rate limits and failures ----------------------------------------------


@pytest.mark.parametrize("key", ["Note", "Information"])
async def test_rate_limit_body_pauses_live_calls(av, make_client, clock, caplog, key) -> None:
    route = quote_route(av).respond(json={key: "Thank you for using Alpha Vantage! ..."})
    client = make_client()

    with caplog.at_level(logging.WARNING):
        quote = await client.get_quote("SPY")

    assert quote.source == "mock"
    assert route.call_count == 1
    assert client.budget.paused
    assert any("rate-limit" in r.getMessage() for r in caplog.records)

    clock.advance(30)
    assert (await client.get_quote("AAPL")).source == "mock"
    assert route.call_count == 1  # paused: no HTTP

    clock.advance(31)
    route.respond(json=global_quote_for("AAPL"))
    assert (await client.get_quote("AAPL")).source == "live"
    assert route.call_count == 2


async def test_rate_limit_body_serves_stale_cache(av, make_client, clock) -> None:
    route = quote_route(av)
    route.side_effect = [
        httpx.Response(200, json=global_quote_for("SPY", "750.0000")),
        httpx.Response(200, json={"Note": "rate limited"}),
    ]
    client = make_client()
    await client.get_quote("SPY")
    clock.advance(31 * 60)

    quote = await client.get_quote("SPY")

    assert quote.source == "stale_cache"
    assert quote.price == 750.0


FAILURES = [
    pytest.param(httpx.ConnectError("boom"), id="connect-error"),
    pytest.param(httpx.ReadTimeout("slow"), id="timeout"),
    pytest.param(httpx.Response(500), id="http-500"),
    pytest.param(httpx.Response(503, text="unavailable"), id="http-503"),
    pytest.param(httpx.Response(200, text="<html>not json</html>"), id="not-json"),
    pytest.param(httpx.Response(200, json={"unexpected": 1}), id="no-global-quote"),
    pytest.param(
        httpx.Response(200, json={"Global Quote": {"01. symbol": "SPY"}}), id="partial-quote"
    ),
]


@pytest.mark.parametrize("failure", FAILURES)
async def test_failures_fall_back_to_mock(av, make_client, caplog, failure) -> None:
    quote_route(av).mock(side_effect=failure)
    client = make_client()

    with caplog.at_level(logging.DEBUG):
        quote = await client.get_quote("SPY")

    assert quote.source == "mock"
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert TEST_API_KEY not in caplog.text


@pytest.mark.parametrize("failure", FAILURES)
async def test_failures_fall_back_to_stale_cache(av, make_client, clock, failure) -> None:
    route = quote_route(av)
    route.side_effect = [httpx.Response(200, json=global_quote_for("SPY", "760.0000")), failure]
    client = make_client()
    await client.get_quote("SPY")
    clock.advance(31 * 60)

    quote = await client.get_quote("SPY")

    assert route.call_count == 2
    assert quote.source == "stale_cache"
    assert quote.price == 760.0


async def test_error_message_body_falls_back_and_never_logs_key(av, make_client, caplog) -> None:
    route = quote_route(av).respond(
        json={"Error Message": "Invalid API call. Please retry or visit the documentation."}
    )
    client = make_client()

    with caplog.at_level(logging.DEBUG):
        quote = await client.get_quote("SPY")

    assert route.call_count == 1
    assert quote.source == "mock"
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert TEST_API_KEY not in caplog.text
    assert not client.budget.paused


async def test_httpx_request_log_redacts_key(av, make_client, caplog) -> None:
    quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    client = make_client()
    with caplog.at_level(logging.DEBUG, logger="httpx"):
        await client.get_quote("SPY")
    assert TEST_API_KEY not in caplog.text
    for record in caplog.records:
        assert TEST_API_KEY not in record.getMessage()


# -- unknown symbols and missing mocks ---------------------------------------


async def test_unknown_symbol_raises_and_is_cached(av, make_client, clock) -> None:
    route = quote_route(av).respond(json={"Global Quote": {}})
    client = make_client()

    with pytest.raises(SymbolNotFoundError):
        await client.get_quote("APPL")
    clock.advance(29 * 60)
    with pytest.raises(SymbolNotFoundError):
        await client.get_quote("appl")
    assert route.call_count == 1

    clock.advance(2 * 60)  # miss expired: asks again
    with pytest.raises(SymbolNotFoundError):
        await client.get_quote("APPL")
    assert route.call_count == 2


async def test_unknown_symbol_is_not_masked_by_mock(av, make_client) -> None:
    assert "SPY" in mock_symbols()
    quote_route(av).respond(json={"Global Quote": {}})
    client = make_client()
    with pytest.raises(SymbolNotFoundError):
        await client.get_quote("SPY")


async def test_no_mock_for_symbol_raises_unavailable(av, make_client) -> None:
    quote_route(av).mock(side_effect=httpx.ConnectError("down"))
    client = make_client()
    with pytest.raises(QuoteUnavailableError):
        await client.get_quote("XYZ")


async def test_mock_mode_unbundled_symbol_raises_unavailable(av, make_client) -> None:
    route = quote_route(av)
    client = make_client(market_data_mode="mock")
    with pytest.raises(QuoteUnavailableError):
        await client.get_quote("XYZ")
    assert route.call_count == 0


# -- input validation --------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", "   ", ".", "-", "...", "TOOLONGSYMB", "SP Y", "$SPY", "SPY;DROP", "ÅAPL", None, 123]
)
async def test_invalid_symbol_raises_value_error_before_lookup(av, make_client, bad) -> None:
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    client = make_client()
    with pytest.raises(ValueError):
        await client.get_quote(bad)  # type: ignore[arg-type]
    assert route.call_count == 0


@pytest.mark.parametrize("good", ["BRK.B", "BF-B", "A", "0700.HK", "abcdefghij"])
async def test_valid_symbol_shapes_accepted(av, make_client, good) -> None:
    quote_route(av).mock(
        side_effect=lambda request: httpx.Response(
            200, json=global_quote_for(request.url.params["symbol"])
        )
    )
    client = make_client()
    assert (await client.get_quote(good)).symbol == good.upper()


# -- no key / mock mode --------------------------------------------------------


async def test_no_api_key_serves_mock_only_and_warns_once(av, clock, caplog) -> None:
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    with caplog.at_level(logging.WARNING):
        client = MarketDataClient(Settings(alpha_vantage_api_key=None), clock=clock)
        for sym in BUNDLED:
            assert (await client.get_quote(sym)).source == "mock"
    assert route.call_count == 0
    warnings = [r for r in caplog.records if "ALPHA_VANTAGE_API_KEY" in r.getMessage()]
    assert len(warnings) == 1


async def test_mock_mode_never_calls_http(av, make_client) -> None:
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    client = make_client(market_data_mode="mock")
    for sym in BUNDLED:
        quote = await client.get_quote(sym.lower())
        assert quote.source == "mock"
        assert quote.symbol == sym
        assert quote.price > 0
    assert route.call_count == 0


def test_mock_bundle_covers_required_symbols() -> None:
    assert set(BUNDLED) <= mock_symbols()


async def test_mock_spy_is_plausible(make_client) -> None:
    quote = await make_client(market_data_mode="mock").get_quote("SPY")
    assert 700 < quote.price < 830


async def test_module_get_quote_mock_mode_from_env(av, monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    route = quote_route(av)
    quote = await get_quote("SPY")
    assert quote.source == "mock"
    assert route.call_count == 0


# -- concurrency and event loops ---------------------------------------------


async def test_ten_concurrent_requests_make_one_http_call(av, make_client) -> None:
    async def slow_response(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=SPY_GLOBAL_QUOTE)

    route = quote_route(av).mock(side_effect=slow_response)
    client = make_client()

    quotes = await asyncio.gather(*(client.get_quote("SPY") for _ in range(10)))

    assert route.call_count == 1
    assert sorted(q.source for q in quotes).count("live") == 1
    assert all(q.price == 764.2 for q in quotes)


def test_consecutive_asyncio_run_calls_succeed(av, monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", TEST_API_KEY)
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)

    first = asyncio.run(get_quote("SPY"))
    second = asyncio.run(get_quote("SPY"))

    assert first.source == "live"
    assert second.source == "cache"
    assert route.call_count == 1


def test_consecutive_asyncio_run_live_calls_succeed(av, make_client, clock) -> None:
    route = quote_route(av).respond(json=SPY_GLOBAL_QUOTE)
    client = make_client()
    first = asyncio.run(client.get_quote("SPY"))
    clock.advance(31 * 60)
    second = asyncio.run(client.get_quote("SPY"))
    assert (first.source, second.source) == ("live", "live")
    assert route.call_count == 2


def test_ten_threads_with_asyncio_run_make_one_http_call(av, monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", TEST_API_KEY)

    async def slow_response(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=SPY_GLOBAL_QUOTE)

    route = quote_route(av).mock(side_effect=slow_response)
    barrier = threading.Barrier(10)
    results: list[object] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            outcome: object = asyncio.run(get_quote("SPY"))
        except BaseException as exc:  # surface failures to the assertion below
            outcome = exc
        with results_lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(results) == 10
    assert not [r for r in results if isinstance(r, BaseException)], results
    assert route.call_count == 1
    assert [r.source for r in results].count("live") == 1  # type: ignore[attr-defined]


async def test_cancelled_waiter_does_not_leak_lock(av, make_client, clock) -> None:
    async def slow_response(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)
        return httpx.Response(200, json=SPY_GLOBAL_QUOTE)

    quote_route(av).mock(side_effect=slow_response)
    client = make_client()
    holder = asyncio.create_task(client.get_quote("SPY"))
    await asyncio.sleep(0.01)
    waiter = asyncio.create_task(client.get_quote("SPY"))
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert (await holder).source == "live"
    clock.advance(31 * 60)  # expire the cache so the next call must take the lock again
    assert (await asyncio.wait_for(client.get_quote("SPY"), 2)).source == "live"
