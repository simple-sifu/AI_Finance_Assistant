---
title: 'Market Analysis agent'
type: 'feature'
created: '2026-10-09'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '55452dbed85fa01c6fb36179164ef068565910f8'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/stories/1-project-skeleton-and-alpha-vantage-client.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The `market` route still returns a stub. CAP-4 needs a ticker question to return a quote plus a plain-language explanation of what its figures mean, and CAP-9 needs that to keep working when Alpha Vantage is out of quota (cache or mock data).

**Approach:** A `MarketAnalysisAgent` finds the ticker(s) in the question, fetches each quote through the story 1 `MarketDataClient`, renders the figures deterministically in Python, and has the LLM explain them in plain language. Register it in `install_real_agents()`; its replies pass through the existing advice review like any real agent's.

## Boundaries & Constraints

**Always:** Quote figures shown to the user come from the `Quote` object, formatted by Python, never retyped by the LLM. Every reply says where the data came from and how fresh it is (`live`, `cache`, `stale_cache`, `mock`); mock quotes are clearly labelled as demo data, not real-time prices, with their trading day. Advice-seeking questions open with `ADVICE_REDIRECT`, then explain the figures without a buy/sell/hold verdict. The LLM system prompt is built with `build_system_prompt`.

**Never:** Changing the router, guardrail, or review behavior, or the `market_data` fallback chain (only the 1 s spacing below is added). Price predictions, targets, or buy/sell/hold opinions. Fetching historical series, fundamentals, or news. Inventing a quote for a symbol with none.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Ticker quote | "What's the price of AAPL?" (live) | Figures block (price, change and % vs previous close, open/high/low, volume, trading day, source) then a short plain-language explanation of those figures | N/A |
| Quota exhausted | Same, budget at zero, mock available | Same shape, labelled as demo data from the mock trading day | client falls back; no error |
| Advice-seeking | "Should I buy NVDA?" (`seeks_advice=True`) | Opens with `ADVICE_REDIRECT`, shows the figures, explains them, no verdict | N/A |
| Unknown symbol | "Quote for APPL" (live says not found) | Says no quote exists for APPL, asks to check the ticker; no LLM call | `SymbolNotFoundError` caught |
| Nothing available | Symbol with no live/cache/mock data | Says the quote is unavailable right now, try again later; no LLM call | `QuoteUnavailableError` caught |
| No ticker | "How is the market doing?" / no symbol found | Asks for a ticker, with examples from the mock symbols; no quote fetch | N/A |
| Several tickers | "Compare VOO and QQQ" | Figures block per ticker (in question order), then one explanation comparing them; more than 3 tickers → first 3 quoted and the reply says so | per-ticker errors shown inline for that ticker; others still answered |
| Burst | Two live fetches back to back | Second live call waits until ≥1 s after the first instead of tripping Alpha Vantage's burst limit | N/A |
| Explanation fails | OpenAI error after quotes fetched | Graph's existing "couldn't finish answering" reply | caught by `_agent_node` |

**Decisions (human, 2026-10-09):** Tickers are found by LLM structured extraction, so company names resolve ("What's Apple trading at?" → AAPL); extracted symbols are validated with `normalize_symbol` and a wrong guess surfaces as the "no quote found" reply. A question may name up to 3 tickers, fetched one at a time. To make that safe, `CallBudget` now spaces live calls at least 1 s apart by waiting (without blocking the event loop), pulled forward from story 7's deferred item.

</frozen-after-approval>

## Code Map

- `src/finance_assistant/market_data/budget.py` -- `CallBudget.try_acquire()` (sync, non-blocking; minute/day/pause). Add the 1 s minimum spacing between live calls so callers wait for the slot instead of falling back; existing limits and tests unchanged. `client.py:161` is the only caller; it holds the per-symbol lock while waiting, which is fine.
- `src/finance_assistant/market_data/client.py` -- `get_client()`/`MarketDataClient.get_quote(symbol)` (:140): fresh cache → live (budget) → stale cache → mock; raises `ValueError` (bad symbol), `SymbolNotFoundError`, `QuoteUnavailableError`. `normalize_symbol` validates tickers. Reuse; do not change.
- `src/finance_assistant/market_data/models.py` -- `Quote` (price, open, high, low, previous_close, change, change_percent in percent units, volume, latest_trading_day, source, fetched_at).
- `src/finance_assistant/market_data/mock.py` -- `mock_symbols()` (SPY, VOO, VTI, QQQ, BND, AAPL, MSFT, GOOGL, AMZN, NVDA; trading day 2026-09-29).
- `src/finance_assistant/tutor/finance_qa.py` -- patterns to copy: `run` (:279) with `chat_model(settings, temperature=0.2)`, `build_system_prompt`, `message_text`, advice-redirect prepend; `_format_history`.
- `src/finance_assistant/tutor/router.py` -- `OpenAIClassifier` shows structured output via `with_structured_output` (pattern for ticker extraction if chosen). Do not change routing.
- `src/finance_assistant/tutor/agents.py` -- `install_real_agents()` (:79) registers real agents; `_STUBS["market"]` stays for tests.
- `src/finance_assistant/tutor/__init__.py` -- export the new agent; importing stays light.
- `src/finance_assistant/tutor/models.py` -- `AgentRequest`, `AgentResult`, `Source`.
- `tests/tutor/test_tax_education.py`, `tests/tutor/test_finance_qa.py` -- `FakeLLM` + respx on `COMPLETIONS_URL`, `real_agents` fixture, `ask` with `FakeClassifier`/`AllowAllReviewer`; `tests/market_data/test_client.py` for building a `MarketDataClient` with fake settings/budget and respx on Alpha Vantage.

## Tasks & Acceptance

**Execution:**
- [x] `src/finance_assistant/market_data/budget.py`, `client.py`, `tests/market_data/test_budget.py`, `test_client.py` -- space live calls ≥1 s apart by awaiting the wait (fake clock in tests); close the burst-limit entry in `_bmad-output/implementation-artifacts/deferred-work.md` and drop the now-done instruction from story 7's `invoke_dev_with` in `stories.yaml`
- [x] `src/finance_assistant/tutor/market_analysis.py` (new) -- `MarketAnalysisAgent(client_provider=get_client, settings=None)`: extract up to 3 tickers (structured output), fetch quotes sequentially, render the figures block with source/freshness label, LLM explanation, all matrix replies
- [x] `src/finance_assistant/tutor/agents.py`, `tutor/__init__.py` -- register and export in `install_real_agents()`
- [x] `tests/tutor/test_market_analysis.py` (new) -- every matrix row offline (respx for OpenAI and Alpha Vantage, mock-mode client), source labels for all four `QuoteSource`s, `ask` routing to `market` with review applied
- [x] `tests/tutor/test_market_analysis_live.py` (new) -- `@pytest.mark.live`: 5 CAP-4 questions incl. a company name, a two-ticker comparison, one advice-seeking and one mock-mode; figures present, explanation passes the real advice reviewer

**Acceptance Criteria:**
- Given `install_real_agents()`, when `get_agent("market")` is called, then it returns a `MarketAnalysisAgent`; after `reset_agents()` it is the stub again.
- Given `MARKET_DATA_MODE=mock` and no Alpha Vantage key, when `ask("What's VOO trading at?")` runs with a real OpenAI key, then the route is `market`, the reply shows VOO's mock figures labelled as demo data, and the disclaimer appears once.
- Given the existing suite, when it runs after this change, then all tests pass unchanged.

## Implementation Notes

- Review fixes: spacing is 1.1 s (jitter margin). `acquire()` re-checks the pause after its wait and returns False if a rate-limit pause began meanwhile. A backward wall-clock step books the call one spacing from now instead of waiting out the jump; this clamp only applies when the last slot is further ahead than any real queue (per_minute × spacing), so concurrent callers still get distinct slots.
- Spacing: `CallBudget.try_acquire()` is unchanged (sync, no spacing) so its existing tests hold. New `reserve()` books the live call at `max(now, last slot + 1 s)` under the lock and returns the wait; new `async acquire()` awaits it with an injectable `sleep` (default `asyncio.sleep`). `MarketDataClient.get_quote` now calls `await budget.acquire()`. Concurrent callers get distinct slots (0 s, 1 s, 2 s). A slot is consumed even if the waiter is cancelled (conservative). `test_client.py`'s `make_client` injects a budget whose sleep advances the fake clock, so the suite never sleeps for real.
- "Never retyped by the LLM" is enforced, not just requested. The first live run showed the model restating every figure and getting a comparison wrong (it called VOO's range wider than QQQ's). So the explanation LLM never sees a number: Python turns each quote into word-only descriptions (direction and size of the move, relative range width, position in the range, vs the open, source in words) plus computed comparison leaders, and the prompt forbids digits. As a backstop, any explanation sentence containing a digit (outside ticker symbols) is dropped and logged; if nothing is left, a deterministic word-only explanation built from the descriptions is used.
- Move/range bands: |change| < 0.5 % small, < 2 % moderate, else large; (high-low)/price < 1 % narrow, < 3 % moderate, else wide.
- The ticker-extraction call is a plain structured-output call (like the router's classifier), not built with `build_system_prompt`; the explanation call is. Extraction sees recent history for follow-ups ("what does its volume mean?"). Extraction failures propagate to `_agent_node`'s failure reply, like explanation failures.
- Every reply (including no-ticker and error-only replies) opens with `ADVICE_REDIRECT` when `seeks_advice`; a copy the model adds is removed so it appears once.
- `sources` is `[Alpha Vantage]` when any shown quote came from Alpha Vantage (live/cache/stale), empty for mock-only replies.
- `$` is shown unless the symbol has an exchange suffix such as `.LON` (non-US listing, local currency). Prices under 1 use 4 decimals.
- The live eval runs its five CAP-4 questions against a mock-mode client (no Alpha Vantage quota spent, figures known); the end-to-end `ask` case runs with `MARKET_DATA_MODE=mock` and no AV key per the acceptance criterion.

- Post-review verification (2026-10-09): 343 offline tests pass; live market eval 6/6 on gpt-4o-mini after the review patches.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-09): layers blind (B), edge-case (E), verification-gap (V).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | B, E, V | `acquire` does not re-check the pause after sleeping; a queued waiter fires a live call during a rate-limit pause | medium | `reserve()` checks `_paused_until` only at booking; `acquire` returns True after the sleep unconditionally. Needs two concurrent `get_quote` calls (e.g. two Streamlit sessions) | patch |
| 2 | E | Wall clock stepping backward makes the reserved delay arbitrarily large while the symbol lock is held | low | `slot = max(now, last_slot + spacing)` with `last_slot` far in the future; fix is a one-line clamp | patch |
| 3 | B | Spacing is exactly 1.0 s with no margin for network jitter | low | Spacing is measured at booking on the client; arrival gap at Alpha Vantage can be under 1 s. Fix is a constant change | patch |
| 4 | E | `try_acquire` bypasses spacing and unsorts `_recent` if mixed with `reserve` | false | `try_acquire` has no production caller after this change (only `test_budget.py`); the client uses `acquire` only | rejected |
| 5 | B | A cancelled or paused waiter's slot still counts against the day quota | low | By design (conservative direction, noted by implementer); rare and releasing adds state | rejected |
| 6 | B | In mock mode a typo (APPL) says "isn't available right now, try again later" | medium | Mock/no-key is the default demo setup; `get_quote` raises `QuoteUnavailableError` there, never `SymbolNotFoundError`. Wording fix only | patch |
| 7 | B, E | "Too many tickers" text echoes raw extracted strings | low | `_too_many_text` joins raw LLM output before `normalize_symbol`/`_symbol_label` | patch |
| 8 | B, E | Dedupe and the 3-ticker cap run on raw strings; `$AAPL` is rejected as invalid and `$SPY`/`SPY` use two slots | low | `_dedupe` then `normalize_symbol`; `_SYMBOL_RE` rejects a leading `$`. Same root: validate before dedupe/cap | patch |
| 9 | B | Flat day: figures show 0.00 but the description says "a small move down" | low | `describe_quote` uses raw `change` sign; `_signed` rounds to 2 dp. Direct correction | patch |
| 10 | E | Sub-$1 quotes show change at 2 dp while price uses 4 | low | Penny stocks are rare in a beginner demo; fix adds formatting branches | rejected |
| 11 | B, E | Digit filter drops harmless sentences ("S&P 500", "Nasdaq-100", "401(k)") and every numbered-list item | medium | VOO/SPY explanations naturally say "S&P 500"; the prompt does not forbid lists. Explanations get silently thinned or replaced by the fallback | patch |
| 12 | B | Sentence splitter breaks on "e.g."/"U.S." leaving fragments when a digit is present | low | Needs an abbreviation and a digit in one sentence; fix adds splitter logic | rejected |
| 13 | E | Explanation mentioning a missing ticker that contains a digit is dropped | low | Requires a digit ticker that failed lookup; rare | rejected |
| 14 | B | Fallback explanation omits the demo-data caveat | low | `fallback_explanation` skips the source line; figures block still labels mock. One added sentence | patch |
| 15 | E | Comparison leaders computed across different trading days (live + mock mix) without saying so | low | Mixed sources occur when the budget runs out mid-question, plausible with 25/day; one added description line | patch |
| 16 | V | No test that a curly-apostrophe model redirect is removed | low | Only the straight-apostrophe case is tested | patch |
| 17 | B, V | No test that a ticker-extraction failure gives the graph's failure reply | low | Only the explanation failure path is tested | patch |
| 18 | V | `sources` rule untested for mixed live + mock quotes | low | `any`→`all` passes every current test | patch |
| 19 | B | Real-clock timing test may flake and leaves the ticker task un-awaited | low | 5 ms ticks with `ticks >= 3` on a loaded runner; cancelled task not awaited | patch |
| 20 | B | `test_acquire_books_concurrent_callers_into_distinct_slots` calls sync `reserve()` only | low | Covered once #1 adds a concurrent client test | rejected |
| 21 | B, E | Spec says `client.py` unchanged and "tests pass unchanged", but `client.py` and the `make_client` fixture changed | low | `client.py` change is the one-line `acquire` swap the frozen decision requires; the fixture only injects a fake sleep. Fix would edit the spec | rejected |
| 22 | B | Live eval runs on mock data only | low | Deliberate: no Alpha Vantage quota spent; live path covered offline with respx | rejected |
| 23 | B | Live eval reads only the last paragraph as the explanation | low | Test-only; the assertions still require figures and a reviewed reply | rejected |
| 24 | B | Import order in `tutor/__init__.py`, long docstring line in `client.py` | low | No linter configured; the docstring rewrap is a direct correction | patch (docstring only) |
| 25 | B | `market_analysis.py` imports private `_format_history` from `finance_qa` | low | No named harm beyond style; moving it touches story 3 code | rejected |

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass offline, live deselected
- `uv run pytest -q -m live tests/tutor/test_market_analysis_live.py` -- expected: all pass
