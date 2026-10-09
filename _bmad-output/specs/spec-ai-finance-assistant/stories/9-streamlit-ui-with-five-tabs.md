---
title: 'Streamlit UI with five tabs'
type: 'feature'
created: '2026-10-09'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '423c00fd291237a58d320605dfc9030c7c01d406'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** All six agents work only from Python; there is no UI. CAP-10 needs a user to work in five areas (Chat, Portfolio, Markets, Goals, Knowledge) and reach every agent end to end from the app, which story 11 then deploys.

**Approach:** One Streamlit app (`st.tabs`) calling the existing `ask()` per interaction with `asyncio.run`, session-only state in `st.session_state`, and `install_real_agents()` once at startup. A tab that belongs to one agent sends its questions to that agent's route directly instead of through the router, while advice detection still runs. Every reply shows its text (with the disclaimer `ask` adds) and its sources as clickable links.

**Decisions (human, 2026-10-09):** The four non-Chat tabs are focused panels scoped to their agents. **Chat** reaches all six agents through the router. **Portfolio**: CSV upload (format hint `ticker` + `shares` or `value`), holdings preview, question box → Portfolio agent with the upload (closes story 7's off-route deferred item). **Markets**: ticker/question box → Market agent, plus a news question box → News agent. **Goals**: form (target $, years, expected annual rate %, current savings $) whose submit asks the Goal agent with those values in words. **Knowledge**: list of the knowledge-base articles (title, source, link) plus a question box routed by the router (Finance Q&A or Tax Education; other routes answered as usual).

## Boundaries & Constraints

**Always:** Every answer comes from `ask()`, so routing, review and the disclaimer stay as they are. Uploaded files and chat history live only in the browser session. `$` in agent text is escaped so Streamlit never renders it as LaTeX. Every error the tutor can raise (`ConfigurationError` for a missing OpenAI key, `HoldingsFormatError`, `ValueError`) is shown as a readable message, never a traceback. The app starts and every tab renders with no API keys set.

**Never:** Changing agent, router, guardrail, review or market-data behavior (a new scoped classifier is additive). Persisting anything to disk. User accounts. A separate frontend or backend.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Chat any agent | Chat tab, "What is compound interest?" then "What about Roth?" | Replies routed by the router, with history; sources listed as links under each reply | N/A |
| Scoped tab | Question asked in a single-agent tab | Answered by that tab's agent even if the router would pick another; advice-seeking still gets the redirect | N/A |
| Upload | Portfolio tab, valid `sample_holdings.csv` | Holdings preview, then the Portfolio agent's analysis | N/A |
| Bad upload | CSV missing `ticker`, or a bad row | Each `HoldingsFormatError` problem listed by row; no question sent | caught |
| Dollar amounts | Goal reply with several `$` amounts | Amounts display literally, no LaTeX | escaped |
| No OpenAI key | Any question | "OPENAI_API_KEY is not set…" message in the tab | `ConfigurationError` caught |
| Agent failure | Agent raises | `ask`'s "couldn't finish answering" reply shown | handled by `ask` |
| Empty input | Submit with nothing typed | Nothing sent; no LLM call | N/A |

</frozen-after-approval>

## Code Map

- `src/finance_assistant/tutor/graph.py` -- `ask(question, history, *, portfolio, classifier, reviewer)`: the UI's only entry to the agents; pass a scoped classifier for single-agent tabs. Do not change.
- `src/finance_assistant/tutor/router.py` -- `Classifier` protocol, `OpenAIClassifier`, `classify_question` (never raises; failures → `unavailable`). A scoped classifier wraps `OpenAIClassifier`, keeps its `seeks_advice`, and replaces the route (keeps `unavailable`). The graph also ORs `seeks_advice_keywords`.
- `src/finance_assistant/tutor/agents.py` -- `install_real_agents()` once per process (`st.cache_resource`).
- `src/finance_assistant/tutor/models.py` -- `ChatTurn`, `TutorReply(text, route, seeks_advice, sources)`, `Source(title, url)`.
- `src/finance_assistant/portfolio.py` -- `parse_holdings_csv(bytes)`, `HoldingsFormatError.problems` (each prints its row). `tests/data/sample_holdings.csv` for the upload check.
- `src/finance_assistant/knowledge/articles.py` -- `load_articles()` (`Article` title/source/url/category) for a Knowledge article list.
- `src/finance_assistant/config.py` -- `ConfigurationError`, `get_settings()`.
- Deferred items closed here (`_bmad-output/implementation-artifacts/deferred-work.md`): story 6 `$`-as-LaTeX; story 7 upload ignored off-route (resolved by scoping the Portfolio tab).
- `pyproject.toml` -- add `streamlit`; tests use `streamlit.testing.v1.AppTest` with fake classifier/agents, offline.

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml`, `uv.lock` -- add `streamlit` dependency
- [x] `src/finance_assistant/tutor/router.py`, `tutor/__init__.py` -- additive `ScopedClassifier(route, inner)`; unit tests in `tests/tutor/test_router.py`
- [x] `src/finance_assistant/ui/` (new) -- pure helpers (`escape_markdown_dollars`, source links, error-to-message mapping, history ↔ `ChatTurn`) and the Streamlit app `app.py` with the five tabs per the Decisions
- [x] `app.py` (repo root) -- thin entry so `uv run streamlit run app.py` works
- [x] `tests/ui/` (new) -- helpers unit-tested; `AppTest` renders all five tabs with no keys, and drives each matrix row with fake classifier/agents and no network
- [x] `README.md` -- how to run the app; close the two deferred items in `deferred-work.md`

**Acceptance Criteria:**
- Given no API keys, when `uv run streamlit run app.py` starts, then all five tabs render without error.
- Given real keys, when one question per agent is asked from the UI, then each agent answers end to end with the disclaimer, and cited agents show clickable sources.
- Given the existing suite, when it runs after this change, then all tests pass.

## Implementation Notes

- `ScopedClassifier(route, inner)` (router.py) accepts only the six agent routes. It keeps `inner`'s `seeks_advice` and fixes the route; provider exceptions propagate so `classify_question` still routes them to `unavailable`. Unparseable inner output keeps the scoped route with `seeks_advice=False` (the graph's keyword backstop still applies) instead of `clarify`, so a scoped tab always reaches its agent.
- `ui/app.py` seams for tests: `router_classifier()` (OpenAI classifier; raises `ConfigurationError` with no key, caught and shown), `advice_reviewer()` (None = `ask`'s default), and `install_real_agents` via `st.cache_resource`. AppTest patches these; no app code reads test state.
- Each tab keeps its own session-only conversation; Chat, Portfolio, Markets and Knowledge pass their history for follow-ups. The Goals form sends each submit without history, since every submit is a full set of values.
- A bad upload shows each `HoldingsFormatError` problem and blocks the question ("nothing was sent"). With no upload, the Portfolio question still goes to the Portfolio agent (it handles typed holdings).
- `$` escaping skips inline code spans and fenced blocks, where an escape would show a backslash.
- Live check (2026-10-09, real keys, through AppTest): one question per agent from the UI (Chat compound interest + Roth follow-up, Portfolio upload of `sample_holdings.csv`, Markets AAPL, News, Goals form, Knowledge advice prompt); every reply carried the disclaimer, cited agents listed link sources, `$` amounts were escaped. `uv run streamlit run app.py` started and `/_stcore/health` returned ok.

- Post-review verification (2026-10-09): 615 offline tests pass; the app starts headless with no API keys (`/_stcore/health` ok). The manual browser click-through is still to do.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-09): layers blind (B), edge-case (E), verification-gap (V).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | V, B | No test that a second Goals submit sends no history | low | `use_history=False` at `goals_tab`; flipping it passes every test (single submit only) | patch |
| 2 | V, B, E | A tab's error is cleared only by a later successful ask; the Portfolio "fix the problems… nothing was sent" error stays after the file is fixed or removed; clearing is untested | medium | `_errors().pop` runs only in `_ask`; `portfolio_tab` never re-checks the stored bad-upload error | patch |
| 3 | B, E | A question that raises (e.g. missing OpenAI key) vanishes: not added to history and the form clears | low | `_ask` returns before appending; `clear_on_submit=True`. Direct fix: show the question in the error | patch |
| 4 | B, E | Portfolio history isn't reset when the upload changes; no tab can clear its conversation | low | Analysis is recomputed from the current upload each time; history only colors the explanation; fix adds upload-id state | rejected |
| 5 | E | `escape_markdown_dollars` skips a `$` after an escaped backslash (`\\$`), incl. link titles after `_LINK_TEXT_SPECIALS` doubles a backslash | low | `(?<!\\)\$` ignores backslash parity; one-regex fix | patch |
| 6 | B, E | Code detection misses `~~~`, indented, double-backtick and unclosed fences | low | Agent replies don't produce code blocks; fix adds Markdown parsing | rejected |
| 7 | B | `$` inside Markdown link targets in agent text gets a backslash | low | Agents emit `[n]` citations, not Markdown links; sources render via `source_links`, which doesn't escape URLs | rejected |
| 8 | B | `ValueError` branch shows raw exception text | false | Reachable `ValueError`s are the UI-guarded empty question and malformed history; classifier/agent errors are caught inside `ask`; `ScopedClassifier`'s check is a programmer error | rejected |
| 9 | B | History sent back includes the disclaimer and has no size cap | false | Router caps history at 6 turns, agents at 4 clipped turns; `apply_guardrail` removes a model-written disclaimer, so it can't double | rejected |
| 10 | B | No `.streamlit/config.toml`: 200 MB upload default, usage telemetry on; CSV re-parsed every rerun | low | Parser rejects > 1 MB but Streamlit buffers the whole upload first; telemetry contradicts session-only data. Config file is trivial; re-parse cost of ≤ 1 MB is negligible (that part rejected) | patch |
| 11 | B | `$`-as-LaTeX item closed without a real browser render | low | AppTest checks the string only; Streamlit documents `\$` as a literal dollar. Covered by the spec's manual browser check | rejected |
| 12 | B | Matrix rows only partly tested (Portfolio with no upload, whitespace-only input, other tabs' missing key, article-load failure) | low | Each matrix row has a passing test; the no-upload Portfolio send and whitespace input are cheap and pin stated choices | patch (those two) |
| 13 | B | README doesn't say what Markets/News do without `ALPHA_VANTAGE_API_KEY`/`TAVILY_API_KEY`; `TAVILY_API_KEY` missing from its table | low | README table lists 5 variables, no Tavily (story 8 gap now user-facing via the app docs) | patch |
| 14 | B | `ScopedClassifier(route: str)` with `type: ignore`s | low | No type checker configured; runtime check rejects bad routes; no named harm | rejected |
| 15 | B | Render test hard-codes one knowledge-base entry | low | Any article edit breaks a tab-render test; direct fix to check the first loaded article | patch |
| 16 | E | `install_real_agents` raising at startup gives a traceback | false | Agent constructors only store providers; heavy resources load lazily inside `run`, where `ask` catches failures | rejected |

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass offline, live deselected

**Manual checks (if no CLI):**
- `uv run streamlit run app.py`: ask one question per agent (incl. upload of `tests/data/sample_holdings.csv` and a Goals form submit); each answers with the disclaimer, sources are links, and `$` amounts display literally.
