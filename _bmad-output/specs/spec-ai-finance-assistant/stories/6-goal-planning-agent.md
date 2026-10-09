---
title: 'Goal Planning agent'
type: 'feature'
created: '2026-10-09'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: 'c85ca791d0c55ca37d0e8284c453c748b8f7a3ea'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/stories/5-market-analysis-agent.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The `goal_planning` route still returns a stub. CAP-5 needs a user to learn how much to save each month to reach a target amount by a date, with an arithmetically correct monthly amount and the compound-interest math shown.

**Approach:** A `GoalPlanningAgent` extracts the goal's inputs (target, horizon, annual rate, optional current savings) with one structured-output LLM call, computes the monthly amount deterministically in a pure Python function, renders the math (formula, values substituted, totals) in Python, and has the LLM only explain the computed result in words. Register it in `install_real_agents()`; replies pass through the existing advice review.

## Boundaries & Constraints

**Always:** All arithmetic happens in Python (`decimal` or float with correct rounding to cents); the LLM never computes and never retypes a figure the user sees (same word-only explanation and digit backstop as story 5). Model: monthly compounding at annual rate / 12, deposits at the end of each month, `FV = PV·(1+r)^n + PMT·((1+r)^n − 1)/r`, solved for `PMT`; `r = 0` → `PMT = (FV − PV)/n`. Horizons given as a date are converted to whole months from today in Python. The computation function is public and UI-callable (story 9 Goals tab). The explanation system prompt is built with `build_system_prompt`.

**Never:** Recommending a rate of return, an investment, an account, or whether the goal is realistic for this user. Inflation adjustment, taxes, fees, irregular deposits, or withdrawal planning. Changing the router, guardrail, or review behavior.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Full inputs | "How much do I need to save monthly to reach $50,000 in 5 years at 5%?" | Monthly amount ($735.23), the formula with values substituted, total deposited vs interest earned, then a plain-language explanation | N/A |
| With current savings | "...I already have $10,000 saved" | `PV` grows at the rate and reduces the monthly amount; shown in the math | N/A |
| Target date | "$20k by June 2030 at 4%" | Months computed from today's date; the month count is shown | N/A |
| Zero rate | "...at 0%" | `PMT = (FV − PV)/n`; no interest earned | N/A |
| Already covered | `PV` grown alone reaches the target | Says no monthly saving is needed at that rate, shows the growth math | N/A |
| Invalid inputs | Negative/zero target, horizon in the past or > 100 years, rate < 0 or > 50 % | Says which value is out of range and asks again; no explanation call | validation in Python |
| Rate missing | "How much should I save to have $30k in 4 years?" | Math shown at illustrative rates 0 %, 4 % and 7 %, labelled as examples, not predictions or suggestions; invites the user to give their own rate | N/A |
| Target or horizon missing | "I want to save for a house" | Asks for exactly the missing values (and the rate if also missing); no math, no explanation call | N/A |
| Advice-flagged, computable | "How much should I save each month to reach $50,000 in 5 years at 5%?" (`seeks_advice=True`) | No advice redirect; the computed answer as in "Full inputs" | N/A |
| Advice-flagged, not computable | "How much should I invest in VOO each month?" | Opens with `ADVICE_REDIRECT`, then asks for a target, horizon and rate to show the savings math | N/A |
| Explanation fails | OpenAI error after the math is computed | Graph's existing "couldn't finish answering" reply | caught by `_agent_node` |

**Decisions (human, 2026-10-09):** When only the rate is missing, show the math at illustrative annual rates of 0 %, 4 % and 7 %, clearly labelled as examples (not predictions or suggestions), and invite the user's own rate; when the target or horizon is missing, ask for exactly what is missing and compute nothing. Goal Planning does not open with `ADVICE_REDIRECT` when it computes from the user's own numbers (doing math with numbers the user gave is allowed by the shared rule), and does open with it when it cannot compute; the advice review still runs on every reply.

</frozen-after-approval>

## Code Map

- `src/finance_assistant/tutor/market_analysis.py` -- patterns to reuse: structured-output extraction (`_extract_tickers`), deterministic rendering, word-only description for the LLM, `strip_numeric_sentences` + deterministic fallback, advice-redirect placement and removal of a model-added copy. Import its helpers rather than copying them; do not change its behavior.
- `src/finance_assistant/tutor/guardrail.py` -- `ADVICE_REDIRECT`, `build_system_prompt` (shared rule already says "You may calculate with numbers the user gives you"). Do not change.
- `src/finance_assistant/tutor/router.py` -- `ROUTER_SYSTEM_PROMPT` routes saving-toward-a-target questions to `goal_planning`; `seeks_advice_keywords` flags "how much should I save". Do not change.
- `src/finance_assistant/tutor/agents.py` -- `install_real_agents()` registers real agents; `_STUBS["goal_planning"]` stays for tests.
- `src/finance_assistant/tutor/__init__.py` -- export the agent and the computation function; importing stays light.
- `src/finance_assistant/tutor/finance_qa.py` -- `_format_history`, `message_text`.
- `tests/tutor/test_market_analysis.py` -- `FakeOpenAI`/respx pattern for a structured call plus an explanation call, `ask` with `FakeClassifier`/`AllowAllReviewer`.

## Tasks & Acceptance

**Execution:**
- [x] `src/finance_assistant/goals.py` (new) -- pure `monthly_savings(target, months, annual_rate, current=0)` returning a result dataclass (monthly amount, months, monthly rate, growth factor, future value of current savings, total deposited, interest earned) plus input validation and `months_until(date, today)`; no LLM imports
- [x] `src/finance_assistant/tutor/goal_planning.py` (new) -- `GoalPlanningAgent(settings=None, today=date.today)`: extraction, validation replies, rendered math, word-only explanation, all matrix rows
- [x] `src/finance_assistant/tutor/agents.py`, `tutor/__init__.py` -- register and export
- [x] `tests/test_goals.py` (new) -- hand-checked values (incl. $50,000 / 60 months / 5 % → $735.23), zero rate, current savings, already-covered, validation bounds, date → months
- [x] `tests/tutor/test_goal_planning.py` (new) -- every matrix row offline (respx), `ask` routing to `goal_planning` with review applied
- [x] `tests/tutor/test_goal_planning_live.py` (new) -- `@pytest.mark.live`: 5 CAP-5 questions incl. a target date, current savings, and an advice-flagged phrasing; computed amount present and correct, reply passes the real advice reviewer

**Acceptance Criteria:**
- Given `install_real_agents()`, when `get_agent("goal_planning")` is called, then it returns a `GoalPlanningAgent`; after `reset_agents()` it is the stub again.
- Given a real key, when `ask("How much do I need to save each month to reach $50,000 in 5 years at 5%?")` runs, then the route is `goal_planning`, the reply contains $735.23 and the substituted formula, and the disclaimer appears once.
- Given the existing suite, when it runs after this change, then all tests pass.

## Implementation Notes

- `annual_rate` in `goals.monthly_savings` is a percentage (`5` = 5 %). Money is `Decimal`, quantized inside the 40-digit context. PMT is rounded up to the next cent (ROUND_CEILING) so the plan never falls short; other amounts half-up. `ending_balance` is what the rounded PMT actually reaches; the reply states how far above the target it ends. `interest_earned = ending_balance − PV − deposits`. `already_covered` is stored on the plan from `PV·(1+r)^n ≥ FV` before rounding, so a sub-cent PMT becomes $0.01, not "covered". The inputs block says the rate is a yearly rate compounded monthly, not an APY.
- Dates: month-only ("by June 2030") resolves to the last day of that month, year-only to Dec 31; `months_until` counts whole months and treats an end-of-month target as a full month (Oct 9 2026 → June 2030 = 44). The reply shows the month count and "from today (…) to …".
- A date wins over a duration in the extraction: the first live run showed the model converting "by June 2030" into 72 months itself (it does not know today's date). The prompt now forbids that, and Python ignores a duration when a date is present.
- Explanation backstop extends story 5's: besides digits, sentences with spelled-out number words (two…ninety, hundred, thousand, million, billion) are dropped (`strip_figure_sentences`), because the live model wrote "fifty thousand dollars"/"five years". "one", "twelve" (rate ÷ twelve) and "zero" are allowed. Fallback is the word-only description.
- Rate missing: example rates 0/4/7 % with a label, the word-only explanation, then an invitation for the user's rate. Extraction sees history and is told to ignore numbers from the tutor's replies (example rates).
- Live eval: 5 CAP-5 questions + end-to-end `ask` pass on gpt-4o-mini; extra manual check: rate-missing, missing-everything and past-date replies via `ask` behave per the matrix. Observation (not changed, router is out of scope): the real router sends "How much should I invest in VOO each month?" to `portfolio`, not `goal_planning`, so the "advice-flagged, not computable" row is only reachable when routed here; it is covered offline.
- Streamlit renders `$…$` as LaTeX; the math lines contain several `$` amounts (as story 5's figures do), so the UI (story 9) must escape dollars when rendering.

- Post-review verification (2026-10-09): 417 offline tests pass; live goal eval 7/7 on gpt-4o-mini after the review patches.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-09): layers blind (B), edge-case (E), verification-gap (V).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | B, E | `already_covered` is `monthly_amount == 0`, so a positive PMT under half a cent reports "your current savings alone reach the target" with $0 saved | medium | Reproduced: `monthly_savings(500, 1200, 7)` → PMT 0.00, `already_covered` True, ending 0.00 | patch (with #2) |
| 2 | B | Half-up rounding makes "Monthly savings needed" fall short of the target | medium | Reproduced: 0 % / $50,000 / 60 months → $833.33 × 60 = $49,999.80. Rounding PMT up to the next cent always reaches the target and is still a correct rounding to cents; $735.23 is unchanged | patch |
| 3 | E | The rounding note says "differs slightly" even when the difference is large | low | Once #2 rounds a tiny PMT up to $0.01, a long horizon can end far above the target (e.g. $1,000 / 1200 months / 50 %); show the actual difference | patch |
| 4 | E | Very large amounts raise `decimal.InvalidOperation` from the public function | low | Reproduced with target 1e27, current 1e7, 1200 months, 50 %: `_cents` runs outside the 40-digit context. Direct correction: quantize inside the context | patch |
| 5 | B | A non-finite number from the model crashes instead of a validation reply | low | Needs OpenAI structured output to emit `1e309`; not seen in practice, and the graph's failure reply still applies | rejected |
| 6 | B, E | A month or day without a year ("by June") is silently dropped; the user gets the generic horizon prompt | low | The reply still asks for the horizon; a specific hint adds branches | rejected |
| 7 | B, E | Number-word backstop lets "one year"/"twelve months" through and drops harmless "two sources of growth" sentences | low | A restated "one year" is the user's own correct figure; over-dropping only thins the explanation, with the fallback as a floor | rejected |
| 8 | B | The illustrative description always says "at the zero rate it all comes from deposits", false when current savings exist | low | The explanation model is fed a statement the shown math contradicts; one conditional sentence | patch |
| 9 | B | No live case for the rate-missing (example rates) reply, the one most likely to be flagged as advice | low | Only a manual check is recorded; adding a live case is test-only | patch |
| 10 | B | The live test cannot tell when the model's explanation was replaced by the fallback | low | Test-only signal; the offline suite covers the fallback path | rejected |
| 11 | B | The reply does not say the rate is treated as nominal annual compounded monthly (not APY) | low | Savings accounts quote APY; a one-line note in the inputs block is a direct addition | patch |
| 12 | B | Math rendering (`format_plan` etc.) lives in the tutor module, so the Goals tab must import the LLM stack to reuse it | low | The computation the spec requires to be UI-callable is in `goals.py`; langchain is a dependency anyway. Revisit in story 9 | rejected |
| 13 | B | `$` pairs in the math may render as LaTeX in Streamlit `st.markdown` | medium | Affects story 5 replies too; rendering belongs to story 9 | defer |
| 14 | B | "Today" is the server's local date; the live test captures today at import | low | Single-process demo; month count can be off by one only near midnight | rejected |
| 15 | B | `test_validation_edges_are_inclusive` has an assertion-free call; unused `monkeypatch` param in a live test | low | Direct corrections | patch |
| 16 | B, E | `format_illustrative` hardcodes "three" rates and the join breaks for a single rate | low | `ILLUSTRATIVE_RATES` is a fixed human decision (0/4/7 %) | rejected |
| 17 | E | A tiny non-zero rate shows r = 0 and (1.000000 − 1) in the substitution | low | Needs a rate like 0.0000001 %; fix adds formatting branches | rejected |
| 18 | V | No test checks the rendered "Ending balance" line and the rounding note | low | The full-inputs test compares `format_plan` with itself | patch |
| 19 | V | The 0 % already-covered rendering branch is never exercised | low | Only the 5 % already-covered case renders math in tests | patch |

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass offline, live deselected
- `uv run pytest -q -m live tests/tutor/test_goal_planning_live.py` -- expected: all pass
