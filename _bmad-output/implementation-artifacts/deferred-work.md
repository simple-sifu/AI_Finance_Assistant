- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/1-project-skeleton-and-alpha-vantage-client.md`
  summary: A live empty `{"Global Quote": {}}` overwrites any stale real quote for that symbol and raises SymbolNotFoundError for a full TTL, even for bundled symbols.
  evidence: Unverified (maybe-false); would be medium. It is settled by observing whether Alpha Vantage ever returns an empty Global Quote for a valid ticker. If it does, keep the stale entry separate from the not-found marker.
- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/1-project-skeleton-and-alpha-vantage-client.md`
  summary: An Alpha Vantage `Information` reply is always treated as a rate limit, so an invalid API key could look like a quota hit and silently serve mock data.
  evidence: Unverified (maybe-false); would be medium. It is settled by calling GLOBAL_QUOTE with a deliberately wrong key and checking whether the reply uses `Information` or `Error Message`. If it uses `Information`, distinguish on message text (changes the frozen matrix row, so it needs the human).
  resolution: CLOSED 2026-10-01. Verified by manual test: with `ALPHA_VANTAGE_API_KEY=WRONGKEY123`, GLOBAL_QUOTE returned a valid live SPY quote. Alpha Vantage does not reject unknown free-tier keys, so a bad key cannot be mistaken for a rate limit.
- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/1-project-skeleton-and-alpha-vantage-client.md`
  summary: Alpha Vantage also enforces a 1-request-per-second burst limit that CallBudget does not model. Two live calls within one second get an `Information` reply, which pauses all live calls for 60 s and serves stale or mock data. Address in story 7 (Portfolio Analysis), which fetches quotes for several holdings in a row.
  evidence: Verified 2026-10-01. A GLOBAL_QUOTE request sent less than 1 s after another returned `{"Information": "...Please consider spreading out your free API requests more sparingly (1 request per second)..."}`. Likely fix: have CallBudget space live calls at least 1 s apart (wait, or fall back) so a burst never triggers the 60 s pause.
  resolution: CLOSED 2026-10-09 in story 5 (Market Analysis, which fetches up to 3 tickers in a row). `CallBudget.acquire()` books each live call at least 1 s after the previous one and awaits the wait with `asyncio.sleep` (the event loop keeps running); `MarketDataClient.get_quote` uses it instead of falling back. Minute/day/pause limits are unchanged. Covered by the spacing tests in `tests/market_data/test_budget.py` and `test_back_to_back_live_calls_are_spaced_one_second_apart`.
- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/2-langgraph-router-with-stub-agents-and-advice-guardrail.md`
  summary: An exception from a registered agent's `run()` propagates out of `ask` uncaught, so the UI gets a traceback with no reply and no disclaimer. Decide the agent-failure reply in story 3, the first real agent.
  evidence: Unreachable today (stubs never raise); would be medium once real agents call OpenAI, Alpha Vantage, or news APIs. Settle by choosing the user-facing failure message and catching in `graph._agent_node`.
  resolution: CLOSED 2026-10-01 in story 2 (human decision). `_agent_node` catches run() exceptions and non-AgentResult returns and replies "Sorry, I couldn't finish answering that just now. Please try again in a moment." with the route kept and the disclaimer applied; covered by test_agent_failure_gets_failure_reply.
- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/2-langgraph-router-with-stub-agents-and-advice-guardrail.md`
  summary: Any classifier failure (OpenAI outage, 401 bad key, 429 rate limit, timeout) routes to `clarify`, so the user sees "I'm not sure what you're asking… rephrase" for a valid question.
  evidence: Verified, medium. Behavior is mandated by story 2's frozen I/O matrix, so changing it needs a human decision. Likely fix: distinguish provider/config errors and reply "the tutor is temporarily unavailable" instead.
  resolution: CLOSED 2026-10-01 in story 2 (human renegotiated the frozen matrix). Classifier exceptions route to `unavailable` ("Sorry, the tutor is temporarily unavailable. Please try again in a moment."); 401/403 also logs an ERROR naming OPENAI_API_KEY; unparseable output still routes to `clarify`. Covered by the provider-error tests.

- source_spec: `_bmad-output/implementation-artifacts/spec-advice-guardrail-hardening.md`
  summary: The advice reviewer sees only the latest question and the reply, not conversation history, so context-dependent follow-up advice ("Yes, go with the first one") may pass review.
  evidence: Unverified (medium if true; review findings #5 and #26). Settle it with live review-eval cases for follow-up replies; if they get through, pass recent history to `AdviceReviewer.gives_advice`.

- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/3-knowledge-base-faiss-index-and-finance-qa-agent.md`
  summary: In the Docker image (story 11), the first Finance Q&A question may download the ~90 MB embedding model unless the Hugging Face cache from `build_index.py` is kept in the image (or `HF_HUB_OFFLINE` is set).
  evidence: Unverified (medium if true; story 3 review #13). Settle it when writing the Dockerfile by running the image offline and asking one Finance Q&A question.
- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/3-knowledge-base-faiss-index-and-finance-qa-agent.md`
  summary: `knowledge/articles.py` finds `knowledge_base/` via `Path(__file__).parents[3]`, which breaks if the package is installed non-editable.
  evidence: Unverified (medium if true; story 3 review #23). Settle it in story 11; if the image installs non-editable, add a configurable path and fail clearly when the directory is missing.

- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/6-goal-planning-agent.md`
  summary: Agent replies contain several `$` amounts (story 5 figures, story 6 math lines), which Streamlit `st.markdown` may render as LaTeX between pairs of `$`.
  evidence: Medium, unverified in this app (story 6 review #13). Settle it in story 9 by rendering a Goal Planning reply in the Chat tab; if the math garbles, escape `$` as `\$` when displaying agent text.
  resolution: CLOSED 2026-10-09 in story 9. The UI escapes every unescaped `$` outside code spans as `\$` before `st.markdown` (`ui.helpers.escape_markdown_dollars`); covered by `tests/ui/test_helpers.py` and the Goals/Markets AppTest cases, and checked live with a Goals form submit.

- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/7-portfolio-analysis-agent-and-holdings-upload.md`
  summary: An uploaded portfolio passed to `ask(portfolio=...)` is silently ignored when the router picks a route other than `portfolio` (e.g. "Is my mix too risky?" → finance_qa).
  evidence: Medium (story 7 review #11). The classifier is never told a portfolio is attached. Settle it in story 9: either pre-scope the Portfolio tab to the `portfolio` route, or tell the router an upload is present.
  resolution: CLOSED 2026-10-09 in story 9. The Portfolio tab sends its questions with `ScopedClassifier("portfolio", ...)`, so an upload always reaches the Portfolio agent (advice flag kept). The Chat tab has no upload. Covered by `test_portfolio_upload_preview_then_analysis` ("Is my mix too risky?").

- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/8-news-synthesizer-agent.md`
  summary: Citing agents (Finance Q&A, Tax Education, News Synthesizer) enforce "every factual sentence is cited" only through the prompt; an uncited closing sentence can reach the user.
  evidence: `apply_citations` only requires at least one valid citation; story 8 live replies sometimes end with one uncited sentence. Fixing it needs a sentence-level check that tells factual sentences from connective ones, shared by all three agents.

- source_spec: `_bmad-output/specs/spec-ai-finance-assistant/stories/8-news-synthesizer-agent.md`
  summary: The router sends "Should I buy Tesla after this week's news?" to `market` instead of `news`; add it to the story 10 routing eval and adjust the router prompt if it keeps misrouting.
  evidence: Manual walkthrough check on 2026-10-09 returned route `market` (quote figures, no news) for the spec's own news advice example; story 8 may not change the router.
