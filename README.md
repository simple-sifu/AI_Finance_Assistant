# AI_Finance_Assistant

A multi-agent AI finance tutor that teaches investing concepts and never gives advice.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --python 3.12 --extra dev   # creates .venv; installs the app + dev extra (pytest, pytest-asyncio, respx)
cp .env.example .env                # then add your ALPHA_VANTAGE_API_KEY
```

`.env` is git-ignored. Never commit a real key.

| Variable | Default | Meaning |
|---|---|---|
| `ALPHA_VANTAGE_API_KEY` | unset | Alpha Vantage key. If unset, the app serves mock quotes only. |
| `MARKET_DATA_MODE` | `live` | `live` or `mock`. `mock` never calls the network. |
| `QUOTE_CACHE_TTL_SECONDS` | `1800` | How long a live quote stays fresh in the in-process cache. |

Real environment variables take precedence over `.env`.

## Running tests

```bash
uv run --extra dev pytest -q
```

Tests run offline. They ignore your `.env`, mock Alpha Vantage with respx, and fail if anything opens a real network connection.

## Market data and mock mode

```python
import asyncio
from finance_assistant.market_data import get_quote

quote = asyncio.run(get_quote("SPY"))
print(quote.price, quote.source)   # source is one of: live, cache, stale_cache, mock
```

The Alpha Vantage free tier allows 25 requests/day and 5/minute. To keep demos working, each quote goes through this chain:

1. **Fresh cache** (`source="cache"`): a live quote younger than `QUOTE_CACHE_TTL_SECONDS` (default 1800 s).
2. **Live** (`source="live"`): a `GLOBAL_QUOTE` call, made only if the in-process budget allows it (at most 5 calls per rolling 60 s and 25 per UTC day). A rate-limit reply (`Note` / `Information`) pauses live calls for a fixed 60 s.
3. **Stale cache** (`source="stale_cache"`): an expired but real quote.
4. **Mock** (`source="mock"`): fixed demo quotes bundled for SPY, VOO, VTI, QQQ, BND, AAPL, MSFT, GOOGL, AMZN, and NVDA. These are not real-time prices.

Quota, network, timeout, and API errors never raise. They fall through the chain. Three errors can reach the caller:

- `ValueError`: the symbol is empty or not 1–10 of `A-Z 0-9 . -` starting with a letter or digit.
- `SymbolNotFoundError`: Alpha Vantage says the symbol doesn't exist. This "not found" result is cached for `QUOTE_CACHE_TTL_SECONDS` (default 1800 s) and never replaced with a mock price, so a typo like `APPL` doesn't show a fake quote.
- `QuoteUnavailableError`: no live, cached, or mock quote exists for the symbol.

Mock mode is on when `MARKET_DATA_MODE=mock` or no API key is set. In mock mode, only bundled symbols return quotes, and no HTTP request is ever made:

```bash
MARKET_DATA_MODE=mock uv run python -c "import asyncio; from finance_assistant.market_data import get_quote; print(asyncio.run(get_quote('SPY')))"
```

The call budget lives in process memory, so a restart resets it, but Alpha Vantage's server-side quota doesn't reset. Detecting rate-limit replies is the backstop.

## Knowledge base

`scripts/build_knowledge_base.py` downloads the curated articles in `knowledge_base/articles/`. See the script's docstring for usage.
