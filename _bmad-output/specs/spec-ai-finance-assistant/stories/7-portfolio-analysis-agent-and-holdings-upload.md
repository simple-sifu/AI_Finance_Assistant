---
title: 'Portfolio Analysis agent and holdings upload'
type: 'feature'
created: '2026-10-09'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: 'eb7b6040dd63f4aeae811aff99ce5e19d118ef67'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/stories/5-market-analysis-agent.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The `portfolio` route still returns a stub, and there is no way to give the tutor a portfolio. CAP-3 needs a user to upload their holdings and learn, in plain language, how diversified and concentrated they are.

**Approach:** A pure `portfolio` module parses a holdings CSV and computes the analysis deterministically; `ask()` gains an optional session-only `portfolio`; a `PortfolioAnalysisAgent` takes the uploaded portfolio (or holdings typed in the question), prices share rows through the story 1 client, renders the holdings table and metrics in Python, and has the LLM explain them in words only (story 5 pattern). Register it in `install_real_agents()`; replies pass the existing advice review.

**Decisions (human, 2026-10-09):** CSV columns are `ticker` plus `shares` or `value` (market value in $) per row; a value is used as given, shares are priced via quotes. Holdings come from an upload passed to `ask(portfolio=...)` (UI wiring is story 9) or, when none is uploaded, from holdings typed in the question, extracted by the LLM. A small bundled fund list marks broad and narrow funds so a broad index fund is not treated as single-company risk; unknown tickers are treated as single companies and the reply says so.

## Boundaries & Constraints

**Always:** All arithmetic in Python; the LLM never computes or retypes a figure (word-only description, digit backstop and fallback reused from story 5). Every price says its source (live, cache, stale, demo); unpriceable rows are listed and excluded, never guessed. Uploaded portfolios stay in memory for the call only. Advice-seeking questions open with `ADVICE_REDIRECT`. The analysis names a concentration-risk finding and a diversification level (CAP-3).

**Never:** Recommending buys, sells, rebalancing, target allocations, or specific funds. Sector/fundamental data beyond the bundled fund list. Persisting holdings. Changing the router, guardrail, review, or market-data behavior.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Sample upload | CSV of 6 rows mixing `shares` and `value`, incl. VOO and BND, one stock at 30 % | Holdings table (ticker, value, weight, type, price source), metrics, "concentration risk" naming the 30 % stock, diversification level, then explanation | N/A |
| Broad fund only | 100 % VOO | Not flagged as single-company risk; explained as one fund holding many companies; level high | N/A |
| Typed holdings | "I have 10 AAPL, 5 MSFT and $8,000 in VTI" (no upload) | Same analysis from extracted holdings | N/A |
| Upload wins | Upload present and question also names holdings | Analysis of the upload; reply says so | N/A |
| No holdings | "How diversified is my portfolio?" with no upload or holdings in the text | Explains how to upload (CSV format with an example) or type holdings; no quote fetch, no explanation call | N/A |
| Bad CSV | Missing `ticker` column, row with both/neither of shares/value, negative or non-numeric amount, > 50 rows, > 1 MB | `HoldingsFormatError` listing each problem by row; the agent never sees a bad file | parse-time error |
| Unpriceable row | Shares of a ticker with no live/cache/mock quote, or unknown symbol | Row listed as not priced and excluded from weights; others analysed | per-row, inline |
| Nothing priced | Every row unpriceable | Says no holding could be valued; no explanation call | N/A |
| Advice-seeking | "Should I sell some AAPL?" with upload | Opens with `ADVICE_REDIRECT`, analysis, no sell/rebalance verdict | N/A |
| Explanation fails | OpenAI error after the math | Graph's "couldn't finish answering" reply | caught by `_agent_node` |

</frozen-after-approval>

## Code Map

- `src/finance_assistant/tutor/models.py` -- `AgentRequest` (:52): add `portfolio: Portfolio | None = None` (import type only; keep the module light).
- `src/finance_assistant/tutor/graph.py` -- `TutorState` (:58), `_request` (:69), `ask` (:155): thread an optional keyword `portfolio` through state into `AgentRequest`; validate its type in `ask`. No other behavior change.
- `src/finance_assistant/tutor/market_analysis.py` -- reuse `source_label`, `strip_numeric_sentences`, the extraction/explanation/fallback pattern, `ALPHA_VANTAGE_SOURCE`, advice-redirect placement. Do not change its behavior.
- `src/finance_assistant/market_data/client.py` -- `get_client().get_quote` (spaced live calls, fallback chain, `SymbolNotFoundError`, `QuoteUnavailableError`); `normalize_symbol`. Do not change.
- `src/finance_assistant/goals.py` -- example of a pure, UI-callable module with `Decimal` money and a problems list for validation.
- `src/finance_assistant/tutor/agents.py`, `tutor/__init__.py` -- register and export; `_STUBS["portfolio"]` stays.
- `tests/tutor/test_market_analysis.py` -- respx/`FakeOpenAI` and mock-mode client patterns; `tests/market_data/` fake clock/sleep.

## Tasks & Acceptance

**Execution:**
- [x] `src/finance_assistant/portfolio.py` (new) + bundled `funds.json` -- `parse_holdings_csv(data: str | bytes) -> Portfolio` (header aliases case-insensitive: ticker/symbol, shares/quantity, value/market value; `$` and commas allowed; duplicate tickers merged; ≤ 50 rows, ≤ 1 MB), `HoldingsFormatError`, `analyze(priced) -> PortfolioAnalysis` (weights, top holding, top-3 company share, fund breakdown by category, concentration flags, diversification level); ~20 common funds with category (broad US stock, international stock, bond, narrow/sector)
- [x] `src/finance_assistant/tutor/models.py`, `graph.py` -- optional `portfolio` on `AgentRequest`, `TutorState`, `ask`
- [x] `src/finance_assistant/tutor/portfolio_analysis.py` (new) -- `PortfolioAnalysisAgent(client_provider=get_client, settings=None)`: holdings source, typed-holdings extraction, sequential pricing, rendered table/metrics/findings, word-only explanation, all matrix rows
- [x] `src/finance_assistant/tutor/agents.py`, `tutor/__init__.py` -- register and export the agent, `Portfolio`, `parse_holdings_csv`
- [x] `tests/test_portfolio.py` (new) -- parser rows of the matrix, aliases, merging, limits; analysis thresholds at their edges; a sample CSV at `tests/data/sample_holdings.csv`
- [x] `tests/tutor/test_portfolio_analysis.py` (new) -- every agent matrix row offline; `ask(portfolio=...)` routing to `portfolio` with review applied
- [x] `tests/tutor/test_portfolio_analysis_live.py` (new) -- `@pytest.mark.live`, mock-mode quotes: sample upload, broad-fund-only, typed holdings, advice-seeking; findings present and replies pass the real reviewer

**Acceptance Criteria:**
- Given `tests/data/sample_holdings.csv` and a real OpenAI key in mock market mode, when `ask("How diversified is my portfolio?", portfolio=parse_holdings_csv(...))` runs, then the route is `portfolio`, the reply names the concentration risk and a diversification level, and the disclaimer appears once.
- Given `ask()` called without `portfolio`, when any existing test runs, then behavior is unchanged and the full suite passes.

## Design Notes

Findings rules (deterministic; thresholds are constants):
- Concentration risk: any single company (not a listed fund) ≥ 20 % of value; the three largest companies together ≥ 50 %; or one narrow/sector fund ≥ 40 %.
- Diversification level: **low** if any concentration flag; **high** if broad stock/international/bond funds are ≥ 50 % of value and no flag; otherwise **moderate**.
- Weights use priced value only; excluded rows are listed. Bond share and stock share come from fund categories plus companies (companies count as stock).

## Implementation Notes

- Weights and thresholds are compared as shown (rounded to 0.1 %), so a holding displayed at 20.0 % is always flagged and one at 19.9 % never is. The top-companies flag needs at least two companies (one company is already covered by the single-company rule).
- A zero amount is rejected like a negative one ("must be more than 0"). Merged duplicates sum shares with shares and values with values; a merged holding with both is valued as shares × price + value. If its shares cannot be priced the whole ticker is excluded (never part-counted).
- QQQ is on the fund list as narrow/sector (Nasdaq-100, technology-heavy), so 40 %+ QQQ is concentration risk.
- "Upload wins": whenever an upload is present the reply opens with a note that the uploaded file is used and typed holdings are used only without one; no extraction call is made.
- Typed holdings: if the model returns both shares and value for one mention, the dollar value is used as given. Invalid typed entries (bad ticker, no or non-positive amount) are listed under "Not valued" and the rest analysed.
- The explanation sees a word-only description (categories, size bands, flags, level, bond/stock split, excluded/unknown/demo notes); `strip_numeric_sentences` masks tickers; the deterministic fallback is built from the same description.
- `ask(portfolio=...)` raises `TypeError` for a non-`Portfolio`. The existing finance-QA test that used `portfolio` as an example stub route now uses `news`.
- Review fixes: fund list extended to 35 (common mutual funds, IWM/SCHD/VUG/VTV as narrow); amounts over 1e12 rejected per row/typed holding (and merged totals over it); invalid typed tickers shown as a sanitized label; market-data client resolved once per question; merged excluded rows show shares + given value; bond/stock presence from rows; a description line when one kind of holding is ≥ 90 %.
- Verification (2026-10-09): 467 offline tests pass; live portfolio eval 5/5 on the configured OpenAI model.

- Post-review verification (2026-10-09): 489 offline tests pass; live portfolio eval 5/5 on gpt-4o-mini after the review patches.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-09): layers blind (B), edge-case (E), verification-gap (V).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | B | Common funds (VFIAX, FZROX, FSKAX, SWTSX, VTIAX, VBTLX, SCHD, VUG, IWM) are missing, so a large holding in one is flagged as single-company risk | medium | `funds.json` has 23 tickers; unlisted funds fall to "company". Adding entries in existing categories is data only | patch |
| 2 | B | In demo mode a listed fund with shares (VXUS, IVV, AGG…) can never be priced, yet the reply says "no quote is available for it right now" | medium | `mock_quotes.json` has 10 symbols; the wording implies retrying helps. Wording fix only | patch |
| 3 | B, E | Typed holdings past 50 are sliced off silently; blanks consume slots | low | Needs the model to extract more than 50 typed holdings; fix adds listing logic | rejected |
| 4 | B, E | An invalid extracted ticker's raw text goes into the markdown list and the explanation prompt | low | `_Excluded(raw.strip()[:12], …)`; `|`/markdown/digits break the table or the word-only description. Sanitize like story 5's `_symbol_label` | patch |
| 5 | B, E | An excluded merged holding's detail omits its given dollar value | low | Detail is only "N shares"; direct correction | patch |
| 6 | B | `client_provider()` is called per holding instead of once per question | low | Docstring says per question; resolving once is a direct correction | patch |
| 7 | V, B | No agent test for merged shares+value pricing or whole-ticker exclusion | low | Only parser-level tests touch merged holdings | patch |
| 8 | V | No test for a typed holding with both shares and value (value wins) | low | Every extraction payload has one of them null | patch |
| 9 | V, B | No agent test for `cache`/`stale_cache` source notes or the stale line in the description | low | Only demo and live are exercised | patch |
| 10 | B | No test that a holdings-extraction failure gives the graph's failure reply | low | The only failure test fails `_explain` on an upload | patch |
| 11 | B | An upload is silently ignored when the router picks another route | medium | `ask(portfolio=)` doesn't inform the classifier; router changes are out of scope here; story 9 can pre-scope the Portfolio tab | defer |
| 12 | B | Separately rounded weights/shares may not total exactly 100 % | low | Standard display rounding; fix adds reconciliation logic | rejected |
| 13 | B | 100 % BND or 100 % VXUS rates "high" without noting it is one asset class | low | Design Notes count bond/international funds as broad; a description line when one category is almost everything is a small addition | patch |
| 14 | B | Results aren't returned for the UI; "never stored" relies on no checkpointer | low | Story 9 concern; no checkpointer exists | rejected |
| 15 | B | Extraction schema uses `float` for amounts; redundant temp in `_number` | low | Converted to `Decimal` on entry; cosmetic | rejected |
| 16 | E | Huge amounts (1e30 in CSV or typed) raise `decimal.InvalidOperation` and the whole reply fails | low | Reproduced: `parse_holdings_csv("ticker,value\nVOO,1e30")` is accepted; cents quantizing overflows later. A size bound in parsing/typed validation is a direct correction | patch |
| 17 | E | Typed shares valid but value ≤ 0 excludes the holding | low | Needs the model to extract a non-positive value alongside shares; fix adds branches | rejected |
| 18 | E | A tiny bond holding rounds `bond_share` to 0.0 so the description says "no bond funds" | low | Description keys on the rounded share; use presence instead. Direct correction | patch |
| 19 | E | Multi-line quoted CSV cells report the record's last line number | low | Rare in holdings exports; fix adds line tracking | rejected |
| 20 | E | Extraction may take holdings from earlier turns | low | Intended: follow-ups use history (`test_typed_holdings_use_history_for_follow_ups`) | rejected |
| 21 | V | Live test parses the sample CSV at import | low | Harmless; reads one small file | rejected |

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass offline, live deselected
- `uv run pytest -q -m live tests/tutor/test_portfolio_analysis_live.py` -- expected: all pass
