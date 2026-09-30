---
title: 'Project skeleton and Alpha Vantage client with cache and mock fallback'
type: 'feature'
created: '2026-09-30'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '1990959e49c374f98a724967724c584c4f753027'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/stack.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The Alpha Vantage free tier (25 requests/day, 5/minute) is the #1 demo killer. Every later agent that shows market data (CAP-4) depends on a quote source that must never fail during a demo (CAP-9). No application code exists yet.

**Approach:** Create the async Python app package, and a market-data client that returns a stock quote. The client serves quotes from a 30-minute TTL cache, calls Alpha Vantage only within a tracked request budget, and falls back to stale cache and then bundled mock quotes. Every quote records where it came from.

## Boundaries & Constraints

**Always:**
- Async throughout: the public API is `async def get_quote(symbol) -> Quote`.
- Every `Quote` carries `source` ∈ {`live`, `cache`, `stale_cache`, `mock`}, so later UI can label demo data honestly.
- The client never raises for quota, network, timeout, or rate-limit problems. It degrades through the fallback chain instead.
- The client never exceeds 5 live calls per rolling 60 s or 25 per UTC day in-process.
- Concurrent requests for the same symbol make at most one live call, including across threads (Streamlit runs each session in its own thread).
- Safe across event loops: consecutive `asyncio.run()` calls (as Streamlit makes them) all work. No `httpx.AsyncClient` or asyncio primitive outlives one call. Shared state is guarded by `threading` locks.
- The API key comes only from the environment (`ALPHA_VANTAGE_API_KEY`, loadable from `.env`), and is never logged or committed.
- Tests never touch the real network.

**Never:**
- No agents, LangGraph, RAG, or Streamlit code (stories 2–9).
- No Docker (story 11).
- No external cache store; the cache is in process memory (stack.md).
- No endpoints beyond `GLOBAL_QUOTE`.
- Don't modify `scripts/` or `knowledge_base/`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Fresh cache hit | `SPY` fetched < 30 min ago | Cached quote, `source=cache`; no HTTP call | N/A |
| Cache miss, budget left | `spy`, key set | Live call, symbol normalized to `SPY`, `source=live`; cached | N/A |
| Budget exhausted | 25 calls today or 5 in last 60 s | No HTTP call; stale cache → mock | Falls through the chain |
| AV rate-limit body | 200 with `Note` or `Information` key | Treated as rate-limited; live calls paused 60 s; stale cache → mock | Logged warning |
| Network error / timeout / 5xx | httpx error | Stale cache → mock | Logged warning |
| AV error body | 200 with `Error Message` key (e.g. bad key) | Stale cache → mock | Logged warning; the key is never logged |
| Unknown symbol (live) | `{"Global Quote": {}}` | `SymbolNotFoundError`; the miss is cached 30 min, so repeats make no HTTP call | Not replaced by mock |
| No mock for symbol | Chain reaches mock, symbol not bundled | `QuoteUnavailableError` | Caller decides the message |
| Invalid input | empty or non `[A-Za-z0-9.\-]{1,10}` symbol | `ValueError` before any lookup | N/A |
| No API key / mock mode | key unset, or `MARKET_DATA_MODE=mock` | Mock only, never calls HTTP | One warning at startup if the key is missing |

</frozen-after-approval>

## Code Map

- `scripts/`, `knowledge_base/` -- existing KB builder + 76 articles; untouched. Their convention (`from __future__ import annotations`, dataclasses, module docstrings) is followed.
- `README.md` -- one line today; gains setup + test instructions.
- No `pyproject.toml`, `.gitignore`, or app code exists yet.

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml` -- package `finance_assistant` (src layout), `requires-python >=3.12`. Runtime deps: `httpx`, `python-dotenv`. Dev extra: `pytest`, `pytest-asyncio`, `respx`. Configure pytest with asyncio mode auto -- the single install point Docker reuses later.
- [x] `.gitignore`, `.env.example` -- ignore `.env`, `.venv`, caches. Document `ALPHA_VANTAGE_API_KEY`, `MARKET_DATA_MODE`, `QUOTE_CACHE_TTL_SECONDS` (default 1800).
- [x] `src/finance_assistant/__init__.py`, `src/finance_assistant/config.py` -- a frozen `Settings` loaded from env/.env -- one config source for all later stories.
- [x] `src/finance_assistant/market_data/models.py` -- frozen `Quote` dataclass (symbol, price, open, high, low, previous_close, change, change_percent, volume, latest_trading_day, source, fetched_at) plus the errors `SymbolNotFoundError` and `QuoteUnavailableError`.
- [x] `src/finance_assistant/market_data/cache.py` -- thread-safe TTL cache with an injectable clock. It can return expired entries on request (for `stale_cache`) and stores not-found markers.
- [x] `src/finance_assistant/market_data/budget.py` -- thread-safe rolling-minute + UTC-day call budget with an atomic check-and-reserve, a 60 s pause on rate-limit bodies, and an injectable clock.
- [x] `src/finance_assistant/market_data/mock_quotes.json`, `mock.py` -- fixed, plausible quotes for SPY (≈764), VOO, VTI, QQQ, BND, AAPL, MSFT, GOOGL, AMZN, NVDA.
- [x] `src/finance_assistant/market_data/client.py` -- `MarketDataClient` wiring cache → budget → live `GLOBAL_QUOTE` (short-lived `httpx.AsyncClient` per call, 10 s timeout) → stale → mock. Per-symbol dedupe works across threads and loops (e.g. a `threading.Lock` per symbol, acquired via `asyncio.to_thread`). Export `get_quote` from `market_data/__init__.py`.
- [x] `tests/market_data/test_cache.py`, `test_budget.py`, `test_client.py` -- cover every I/O matrix row with fake clocks and respx. Use the real SPY `GLOBAL_QUOTE` response (2026-09-29, price 764.2000, change percent -0.1842%) as the fixture shape.
- [x] `README.md` -- setup (`uv venv --python 3.12`, install with dev extra, copy `.env.example`), how to run tests, and how mock mode works.

**Acceptance Criteria:**
- Given a fresh clone with the dev extra installed and no `.env`, when `pytest` runs, then all tests pass offline.
- Given the fake clock advances past 30 min, when `SPY` is requested with budget left, then a new live call is made and the cache refreshes.
- Given 10 concurrent `get_quote("SPY")` calls on a cold cache, when they resolve, then exactly one HTTP request was made.
- Given `MARKET_DATA_MODE=mock`, when any bundled symbol is requested, then no HTTP request is made and `source=mock`.
- Given two consecutive `asyncio.run(get_quote("SPY"))` calls, when both run, then both succeed.
- Given 10 threads each calling `asyncio.run(get_quote("SPY"))` on a cold cache, when they finish, then exactly one HTTP request was made.

## Implementation Notes

- Per-symbol dedupe uses a `threading.Lock` per symbol, acquired by polling a non-blocking `acquire()` with `asyncio.sleep(0.01)`, not `asyncio.to_thread(lock.acquire)`. With `to_thread`, a waiter that is cancelled (or whose loop `asyncio.run` tears down) leaves a worker thread that still takes the lock, and nothing releases it. Polling is cancel-safe and works across threads and loops. It's covered by `test_cancelled_waiter_does_not_leak_lock`.
- One injectable epoch clock (`time.time` by default) drives the cache TTL, the budget windows, the UTC day, and `Quote.fetched_at`.
- Only live quotes are cached. Mock and stale results are never written back, so a live call is tried again once the budget allows.
- A stale `SymbolNotFoundError` marker is skipped in the fallback, so the chain goes on to mock.
- Any non-200 status, non-JSON body, missing `Global Quote`, or malformed quote counts as a live failure and falls through the chain. Only `{"Global Quote": {}}` means not found.
- Key safety: `Settings.__repr__` masks the key. Live-failure logs record only the exception type, never `str(exc)`, which contains the URL. A logging filter on `httpx` and `httpcore` redacts `apikey=`.
- The missing-key warning is logged once, when the client is built (in the process-wide client, that happens on first use).
- `change_percent` is stored in percent units (`-0.1842` means -0.1842 %).
- Tests ignore the developer's `.env` and env vars (`tests/conftest.py`), and they block socket connects to prove no network is used.
- The SPY fixture has the story's real price (764.2000) and change percent (-0.1842%). The other fields (open, high, low, volume, previous close, change) are plausible and consistent with those values, but were not captured from the real response, to avoid spending live quota.

## Spec Change Log

## Review Triage Log

| # | Source | Finding | Verdict | Evidence / route |
|---|---|---|---|---|
| 1 | blind | NaN/inf `QUOTE_CACHE_TTL_SECONDS` passes validation | low | `nan <= 0` is False, so entries never expire. The fix is a direct `isfinite` check. **patch** |
| 2 | blind | Empty `Global Quote` overwrites a good stale quote and blocks the symbol for the TTL | maybe-false | Real only if AV returns `{}` for valid tickers during a hiccup; unverified. Would be medium. **defer** |
| 3 | blind | `Information` treated as rate limit hides a bad API key | maybe-false | Depends on which key AV uses for invalid keys; the frozen matrix mandates `Information` = rate limit. Would be medium. **defer** |
| 4 | blind | Stale quotes have no max age | low | Bounded by process lifetime; `fetched_at` is on every quote for the UI to show. The fix adds a parameter. Rejected. |
| 5 | blind | `_symbol_locks` and cache entries grow without bound | low | Bytes per distinct symbol; unlikely to matter in a course demo. The fix adds eviction logic. Rejected. |
| 6 | blind | Waiters on a slow live call get no fallback; the httpx timeout is per phase | low | Needs concurrent same-symbol requests plus a trickling upstream, which is rare for a single-user demo. The fix adds deadline/state. Rejected. |
| 7 | blind | Key redaction covers only the `httpx`/`httpcore` loggers | false | httpx logs the URL only on the `httpx` logger (filtered). httpcore child loggers log `<Request [b'GET']>` without the URL. `HTTPError` is caught `from None`, and the URL is constant, so `InvalidURL` is unreachable. No leaking path is shown. Rejected. |
| 8 | blind | Mock `fetched_at` = now makes demo data look fresh; mock rows lack consistency tests | low | `source='mock'` labels the quote honestly, and `fetched_at` is accurate for when it was produced. Rejected. |
| 9 | blind | Parsed live values not sanity-checked (NaN, negative) | low | AV does not return such values in practice. The fix adds guards. Rejected. |
| 10 | blind | README hardcodes 30 min / 60 s though the TTL is configurable | low | Direct text correction. **patch** |
| 11 | blind | Setup uses `uv pip install -e` but tests use `uv run --extra dev`; `uv.lock` not addressed; deps uncapped | low | The mixed workflow is real and fixable in the README (`uv sync --extra dev`). Whether to commit the lock is the user's decision at commit time. Version caps are speculative (rejected). **patch** (README) |
| 12 | blind | Network guard misses `socket.getaddrinfo` | low | A stray lookup would still send DNS. One-line monkeypatch. **patch** |
| 13 | edge | `cache or TTLCache(...)`: an empty injected `TTLCache` is falsy (`__len__`) and silently replaced | medium | Verified: `client.cache is c` → False for an empty injected cache. Any caller sharing a cache diverges. Direct fix: `is not None`. **patch** |
| 14 | edge | NaN/inf TTL | low | Same root cause as #1. **patch** (grouped) |
| 15 | edge | Empty `Global Quote` overwrites a stale real quote | maybe-false | Same as #2. **defer** (grouped) |
| 16 | edge | Expired NOT_FOUND + exhausted budget serves a mock for a bundled symbol live called unknown | low | Requires AV to call SPY/VOO/etc. unknown, which is implausible. The fix adds a branch. Rejected. |
| 17 | edge | Waiters serialize live retries when live fails (up to 5 × 10 s) | low | Same root as #6; the budget caps it at 5. Rejected. |
| 18 | edge | HTTP 429 does not pause the budget | low | AV signals limits with 200 + `Note`/`Information`; 429 already falls back and the local budget still caps calls. Rejected. |
| 19 | edge | Punctuation-only symbol (`.`, `-`) passes validation and spends a live call | low | Verified. Direct regex correction (leading alphanumeric). **patch** |
| 20 | edge | Cache entries grow without bound | low | Same as #5. Rejected. |
| 21 | edge | Symbol lock map grows without bound | low | Same as #5. Rejected. |
| 22 | edge | Wall clock stepping backward skews the budget | low | Rare NTP edge; the fix requires monotonic + wall clocks. Rejected. |
| 23 | edge | An exported empty env var hides the `.env` value | low | Standard "env wins" semantics, and a missing-key warning is logged. Rejected. |
| 24 | edge | Returned `01. symbol` differing from the request gets cached under the requested key | low | AV echoes the requested symbol; the fix adds a guard. Rejected. |
| 25 | edge | Log filter not applied to child loggers | false | Same as #7. Rejected. |
| 26 | verification-gap | `test_cancelled_waiter_does_not_leak_lock` never re-acquires the lock, so a `to_thread` regression passes CI | high | Pre-verified by the reviewer's mutation: the test passed with the leaking implementation, and the probe hung. **patch** |

## Design Notes

Fallback order is: fresh cache → live (only if budget allows) → stale cache → mock. Stale cache beats mock because it holds a real price, even if old.

A live `SymbolNotFoundError` is deliberately **not** masked with mock data. Otherwise a typo like `APPL` would show a fake price as if it were real.

The budget is in-process, so a container restart resets it, while Alpha Vantage's server-side quota (whose reset time may not be UTC midnight) does not reset. Detecting rate-limit bodies is the backstop, so the budget is best-effort protection, not the source of truth.

## Verification

**Commands:**
- `uv run --extra dev pytest -q` -- expected: all tests pass, no network access.
- `uv run python -c "import asyncio; from finance_assistant.market_data import get_quote; print(asyncio.run(get_quote('SPY')))"` with `MARKET_DATA_MODE=mock` -- expected: a `Quote` with `source='mock'`.
