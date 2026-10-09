# AI_Finance_Assistant

A multi-agent AI finance tutor that teaches investing concepts and never gives advice.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --python 3.12 --extra dev   # creates .venv; installs the app + dev extra (pytest, pytest-asyncio, respx)
cp .env.example .env                # then add your OPENAI_API_KEY and ALPHA_VANTAGE_API_KEY
```

`.env` is git-ignored. Never commit a real key.

| Variable | Default | Meaning |
|---|---|---|
| `ALPHA_VANTAGE_API_KEY` | unset | Alpha Vantage key. If unset, the app serves mock quotes only. |
| `MARKET_DATA_MODE` | `live` | `live` or `mock`. `mock` never calls the network. |
| `QUOTE_CACHE_TTL_SECONDS` | `1800` | How long a live quote stays fresh in the in-process cache. |
| `OPENAI_API_KEY` | unset | OpenAI key for the router and agents. Required to ask questions; `ask` raises `ConfigurationError` naming it if unset. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model used by the router and agents. Must be a model your OpenAI project can access. |
| `TAVILY_API_KEY` | unset | Tavily key for the News Synthesizer's news search. If unset, news questions get a "can't look up current news right now" reply. |
| `APP_PASSWORD` | unset | Shared password for the deployed app. If set, a password screen comes before the tabs; if unset (local use), there is none. |

Real environment variables take precedence over `.env`.

## Running the app

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501. The app has five tabs:

| Tab | What it does |
|---|---|
| Chat | Any question; the router picks one of the six agents. Follow-ups use the chat history. |
| Portfolio | Upload a holdings CSV (`ticker` plus `shares` or `value` per row, e.g. `tests/data/sample_holdings.csv`), preview it, and ask the Portfolio Analysis agent about it. |
| Markets | Ask the Market Analysis agent about a ticker, and the News Synthesizer about current news. |
| Goals | Enter a target, years, an optional expected annual rate and current savings; the Goal Planning agent shows the monthly amount and the math. |
| Knowledge | Browse the knowledge-base articles (title, source, link) and ask concept or tax-account questions (routed by the router). |

Every answer comes from `ask`, so it carries the disclaimer, and cited sources are listed as links under it. The Portfolio, Markets and Goals panels send questions straight to their agent (advice-seeking questions still get the redirect). Uploads and conversations live only in the browser session; nothing is saved. The app starts with no API keys; without `OPENAI_API_KEY` each question shows "OPENAI_API_KEY is not set…" instead of an answer.

- Without `ALPHA_VANTAGE_API_KEY`, Markets (and Portfolio pricing) use the bundled mock quotes, labelled as such.
- Without `TAVILY_API_KEY`, News replies that it can't look up current news right now.

`.streamlit/config.toml` limits uploads to 1 MB (the holdings parser's limit) and turns off Streamlit's usage statistics.

## Deployment

The app ships as one Docker image (`Dockerfile`) and runs on one AWS EC2 `t3.small` at `http://<elastic-ip>`, behind `APP_PASSWORD`. [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md) has the step-by-step console guide; [`deploy/setup-server.sh`](deploy/setup-server.sh) sets up or updates the server.

The image:
- installs CPU-only PyTorch;
- builds the FAISS index during `docker build`;
- caches the embedding model in the image, and the build checks that the index loads and searches with `HF_HUB_OFFLINE=1`;
- reads keys only at run time from `--env-file`.

To build locally (needs Docker): `docker build -t finance-assistant .` then `docker run --rm -p 8501:8501 --env-file .env finance-assistant`.

## Running tests

```bash
uv run --extra dev pytest -q
```

Tests run offline. They ignore your `.env`, mock Alpha Vantage and OpenAI with respx (or inject a fake router classifier), and fail if anything opens a real network connection.

### Routing and advice evals

`tests/tutor/eval_cases.py` is the evaluation set. It has at least 6 plain questions and 2 advice-seeking prompts per agent, plus clarify cases and follow-ups. Two opt-in live evals use it and need `OPENAI_API_KEY` in `.env`:

```bash
uv run --extra dev pytest -m live tests/tutor/test_router_live.py tests/tutor/test_advice_eval_live.py -s
```

- `test_router_live.py` routes every case with the real router and prints a per-case and per-agent report. It passes when:
  - each agent gets at least 5 plain questions right;
  - overall route accuracy is at least 90%;
  - every advice prompt is flagged as advice;
  - at most 1 plain question is flagged as advice.
- `test_advice_eval_live.py` sends one advice prompt per agent through `ask()` with the real agents and reviewer. Each reply must open with the educational redirect and end with the disclaimer.
  - It costs about 20 OpenAI calls, one Tavily search and up to 2 Alpha Vantage calls.
  - Without `ALPHA_VANTAGE_API_KEY` or `TAVILY_API_KEY`, market questions use mock data and news questions get the "unavailable" reply.

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

## Tutor: router, agents, and guardrail

```python
import asyncio
from finance_assistant.tutor import ChatTurn, ask

reply = asyncio.run(ask("What about Roth?", [ChatTurn("user", "How does an IRA work?")]))
print(reply.route, reply.seeks_advice)   # e.g. tax_education False
print(reply.text)                        # answer + disclaimer footer
```

`ask` runs a LangGraph graph: **router → one agent → guardrail**.

- **Router**: an OpenAI classifier (structured output) picks one route for the latest question, using up to the last 6 history turns for follow-ups, and flags advice-seeking. Routes: `finance_qa`, `portfolio`, `market`, `goal_planning`, `news`, `tax_education`, `clarify`. If the classifier returns invalid or unparseable output, the question routes to `clarify`. If it raises (OpenAI outage, timeout, 429, network, bad key), the question routes to `unavailable` and the reply is "Sorry, the tutor is temporarily unavailable. Please try again in a moment." A 401/403 logs an error naming `OPENAI_API_KEY`; logs carry only exception types, never the key.
- **Agents**: six stub agents return placeholder answers for now. A real agent implements `async run(request: AgentRequest) -> AgentResult` and is plugged in with `register_agent(route, agent)`; the router and graph don't change. If an agent raises or returns something other than an `AgentResult`, the route is kept and the reply is "Sorry, I couldn't finish answering that just now. Please try again in a moment." with no sources.
- **Guardrail**: every agent builds its system prompt with `build_system_prompt`, which carries the shared education-not-advice rule. Advice-seeking questions get an educational redirect. The guardrail node appends the disclaimer to every reply: *Educational information only, not financial, tax, or investment advice.*

`ask` raises only for an empty question or malformed history (`ValueError`, before any LLM call) and a missing `OPENAI_API_KEY` (`ConfigurationError`). The graph and the OpenAI HTTP client are built per call, so repeated `asyncio.run(ask(...))` calls (as Streamlit makes them) are safe.

## Knowledge base

`scripts/build_knowledge_base.py` downloads the curated articles in `knowledge_base/articles/`. See the script's docstring for usage.
