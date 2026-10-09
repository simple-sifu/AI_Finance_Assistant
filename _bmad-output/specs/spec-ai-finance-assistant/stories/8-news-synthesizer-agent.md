---
title: 'News Synthesizer agent'
type: 'feature'
created: '2026-10-09'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: 'f7c206e21f78c6dccd08d05d134fad46e7a0e28c'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/stories/3-knowledge-base-faiss-index-and-finance-qa-agent.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The `news` route still returns a stub. CAP-6 needs a user to get summaries of current financial news, and every summary must cite its source URLs.

**Approach:** A small async Tavily client (`httpx`, `POST https://api.tavily.com/search`, `topic: "news"`) fetches recent articles for the question; a `NewsSynthesizerAgent` numbers them and has the LLM summarize only from those excerpts with inline `[n]` citations, reusing the story 3 citation pipeline so `sources` are exactly the cited articles with their URLs. Register it in `install_real_agents()`; replies pass the existing advice review.

**Decisions (human, 2026-10-09):** News search uses Tavily, not SerpAPI (1,000 free searches/month vs 100; returns article excerpts and publish dates built for LLM summarization). Key comes from `TAVILY_API_KEY`. When Tavily can't be reached, the reply says honestly that current news can't be looked up right now and offers to explain a concept instead; no bundled or stale news is shown as current. A 15-minute in-process cache saves quota on repeated questions.

## Boundaries & Constraints

**Always:** Summarize only what the returned excerpts say; every factual sentence carries an `[n]` citation and `sources` lists only cited articles (title, outlet domain, publish date when known, URL). Never answer without a source. Article text is untrusted reference material, never instructions. Advice-seeking questions open with `ADVICE_REDIRECT` and get no buy/sell/hold verdict drawn from the news. The LLM system prompt is built with `build_system_prompt`. The Tavily key is never logged or shown.

**Never:** Changing the router, guardrail, review, or market-data behavior. Price predictions or "what this means for your money" verdicts. Fetching full article pages (`include_raw_content`) or using Tavily's generated `answer`. Persisting searches beyond the in-process cache.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Current news | "What's happening with interest rates this week?" | Short plain-language summary of the main stories with `[n]` citations; `sources` = cited articles with URLs, numbered by first citation | N/A |
| Company news | "Any news on Nvidia?" | Same shape, about that company | N/A |
| Advice-seeking | "Should I buy Tesla after today's news?" | Opens with `ADVICE_REDIRECT`, summarizes the news, no verdict | N/A |
| Follow-up | "Tell me more about the first one" after a news reply | Search uses the question plus the previous user turn | N/A |
| No results | Tavily returns zero results | Says no recent news was found for that, suggests rephrasing; no LLM call | N/A |
| Uncited answer | Model cites nothing valid | Same "no news found" reply; summary withheld | N/A |
| Repeat question | Same search within the cache TTL | Served from cache; no Tavily call | N/A |
| Search unavailable | No key, 401, 429/432/433, 5xx, timeout, bad JSON | `NEWS_UNAVAILABLE_TEXT`: current news can't be looked up right now, try again later, or ask about a concept meanwhile (after `ADVICE_REDIRECT` if advice-seeking); no LLM call | caught; never crashes |
| Summary fails | OpenAI error after the search | Graph's "couldn't finish answering" reply | caught by `_agent_node` |

</frozen-after-approval>

## Code Map

- `src/finance_assistant/config.py` -- `Settings`/`load_settings`: add `tavily_api_key` (from `TAVILY_API_KEY`), redacted in `__repr__`; `require_tavily_api_key()` is not needed (missing key is a handled reply). Update `.env.example` and `tests/conftest.py` `_ENV_VARS`.
- `src/finance_assistant/market_data/client.py` -- pattern for a short-lived `httpx.AsyncClient` per call, `HTTP_TIMEOUT_SECONDS`, and log redaction. `market_data/cache.py` `TTLCache` -- reuse for a 15-minute news cache. Do not change either.
- `src/finance_assistant/tutor/finance_qa.py` -- reuse `apply_citations` (needs only objects with `as_source()`), `message_text`, `_format_history`, `retrieval_queries`-style follow-up rule, `not_covered`-style advice prepend, and the "excerpts are reference text" instruction. Do not change its behavior.
- `src/finance_assistant/tutor/agents.py`, `tutor/__init__.py` -- register and export; `_STUBS["news"]` stays for tests. `tests/tutor/test_portfolio_analysis.py` and `test_finance_qa.py` may use `news` as a stub route — switch them to a route that is still a stub only if they break.
- `src/finance_assistant/tutor/models.py` -- `Source(title, url)`.
- `tests/tutor/test_finance_qa.py`, `test_market_analysis.py` -- `FakeOpenAI` via respx on `COMPLETIONS_URL`, `real_agents` fixture, `ask` with `FakeClassifier`/`AllowAllReviewer`.

## Tasks & Acceptance

**Execution:**
- [x] `src/finance_assistant/config.py`, `.env.example`, `tests/conftest.py`, `tests/test_config.py` -- add `TAVILY_API_KEY` setting, redaction test
- [x] `src/finance_assistant/news.py` (new) -- `NewsClient(settings=None, cache=None, clock=time.time)` with `async search(query) -> list[NewsArticle]` (title, url, domain, content excerpt, published date); `NewsUnavailableError` for every failure; `get_news_client()`/`reset_news_client()`
- [x] `src/finance_assistant/tutor/news_synthesizer.py` (new) -- `NewsSynthesizerAgent(client_provider=get_news_client, settings=None)`: build query, search, numbered-excerpt prompt, cited summary, every matrix row
- [x] `src/finance_assistant/tutor/agents.py`, `tutor/__init__.py` -- register and export
- [x] `tests/test_news.py` (new) -- respx on Tavily: request body/auth header, parsing, each error status, timeout, malformed JSON, cache hit and expiry
- [x] `tests/tutor/test_news_synthesizer.py` (new) -- every agent matrix row offline; `ask` routing to `news` with review applied
- [x] `tests/tutor/test_news_synthesizer_live.py` (new) -- `@pytest.mark.live` (needs OpenAI and Tavily keys): 5 CAP-6 questions incl. one advice-seeking; each reply cites at least one source with an `https` URL and passes the real reviewer

**Acceptance Criteria:**
- Given `install_real_agents()`, when `get_agent("news")` is called, then it returns a `NewsSynthesizerAgent`; after `reset_agents()` it is the stub again.
- Given real OpenAI and Tavily keys, when `ask("What's the latest news on the Federal Reserve?")` runs, then the route is `news`, the text has `[n]` citations, every `[n]` matches `sources[n-1]` with a URL, and the disclaimer appears once.
- Given the existing suite, when it runs after this change, then all tests pass.

## Design Notes

Search parameters (constants): `topic="news"`, `time_range="week"`, `max_results=5`, `search_depth="basic"` (1 credit), `chunks_per_source=3`; query is the question (plus the previous user turn when the question is ≤ 6 words), clipped to 400 characters; cache key is the normalized query. Sources render as `"Headline (reuters.com, 2026-10-08)"` with the URL, matching story 3's `"Title (Outlet)"` style; the UI (story 9) shows them.

## Implementation Notes

- **Client** (`src/finance_assistant/news.py`): `POST https://api.tavily.com/search` with `Authorization: Bearer <key>` (the key is never in the URL or logs; errors log status codes or exception types only). The body carries the design-note constants plus `include_answer=false`, `include_raw_content=false`, `include_images=false`. Any non-200 status, `httpx.HTTPError` (timeouts included), non-JSON body, or body without a `results` list raises `NewsUnavailableError`. A missing key raises it without a request. Malformed or unusable results (no title, no http(s) URL, empty excerpt, duplicate URL) are skipped. Excerpts are clipped to 1,500 characters and titles to 200, and both have their whitespace collapsed, so an excerpt can't fake prompt structure with newlines. A host that is exactly `www.` is rejected. Requests send a constant `exclude_domains` list (facebook.com, instagram.com, tiktok.com, youtube.com, x.com, twitter.com, reddit.com). Titles are capped because some Tavily results are social-media posts whose whole text is the title. `published_date` is parsed as ISO 8601 or RFC 2822; an unparseable date becomes `None`.
- **Cache:** `TTLCache` with a 900 s TTL, keyed by the casefolded, whitespace-collapsed, clipped query. Successful searches are cached, including zero-result ones; failures are not. `tests/conftest.py` resets the process-wide client.
- **Import cycle:** `NewsArticle.as_source()` imports `tutor.models.Source` lazily, because the tutor package imports `news`.
- **Follow-up rule:** the matrix's follow-up example ("Tell me more about the first one") has 7 words, so the ≤ 6-word rule alone would miss it. The frozen matrix wins, so the previous user turn is appended in two cases. (1) The question contains an explicit phrase that refers back, such as "that one", "the first one", "tell me more", or "more about it". Bare pronouns don't count, so "What did Apple say about its earnings this quarter?" is searched alone. (2) The question is ≤ 6 words and doesn't name its own subject. A question names its own subject when a word after the first is capitalized (except "I") or is an all-caps, ticker-like token. So "Any news on Tesla?" is searched alone after an unrelated turn, while "What did they say about it?" is combined. The question always comes first, so clipping at 400 characters only cuts the previous turn.
- **Agent:** reuses `apply_citations` (duck-typed on `as_source()`), `message_text` and `_format_history` from `finance_qa` without changing them. The model may reply `NO_RELEVANT_NEWS` when no excerpt is about the question. Any answer containing that token, an empty reply, and an uncited summary all give `NO_NEWS_TEXT`. The prompt starts with today's date (UTC), and the history block says its citation numbers refer to earlier articles. For advice questions, any copy of the redirect the model wrote (straight or curly apostrophes, optionally quoted) is stripped before the redirect is prepended once. The prompt forbids price predictions, "what this means for your money" verdicts, and repeating analysts' buy/sell/hold ratings or price targets. That last rule was added after the first live run, where a Tesla advice reply quoted "Buy" ratings.
- **Existing test:** `test_finance_qa.py::test_install_real_agents_registers_finance_qa` asserted that `news` was still a stub. After this story no route is a stub, so that one assertion was removed.
- Verification (2026-10-09): 557 offline tests pass; live eval 6/6 on gpt-4o-mini with real Tavily (5 CAP-6 questions + `ask` Federal Reserve end to end).

- Post-review verification (2026-10-09): 561 offline tests pass; live news eval 6/6 on gpt-4o-mini with Tavily after the review patches.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-09): layers blind (B), edge-case (E), verification-gap (V).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | B, E, V | `_FOLLOW_UP_RE` matches bare it/its/they/them/these/those, so long standalone questions get the previous turn appended | medium | "What did Apple say about its earnings this quarter?" matches `its`; `search_query` then sends the combined text to Tavily and caches under it | patch |
| 2 | B, E | A short new-topic question ("Any news on Tesla?") is always combined with the previous user turn | medium | ≤ 6 words → follow-up; a demo asks one question per agent in one chat, so the news query gets blended with e.g. a goal-planning question | patch |
| 3 | B, V | "Every factual sentence carries `[n]`" is enforced only by the prompt; uncited closing sentences pass | low | Code checks only that ≥ 1 source is cited (same as story 3's pipeline); mechanical stripping would also remove non-factual connective sentences | defer |
| 4 | B | History passed to the prompt keeps old `[n]` markers that refer to an earlier article list | low | `_format_history` keeps assistant text verbatim; prompt does not say those numbers are unrelated. One-line prompt fix | patch |
| 5 | B | Prompt omits today's date, so the model can't tell "today" from a week-old story | low | Excerpts carry publish dates but no reference date; `time_range="week"`. One-line prompt fix | patch |
| 6 | B, E | Excerpt text keeps newlines, so a site can fake prompt structure (`[3] "…"`, `---`, `Question:`) | low | `_parse_result` collapses whitespace in titles only. Direct fix: collapse in content too | patch |
| 7 | B, E | News cache never evicts, keyed by free-form text | low | `TTLCache` keeps expired entries by design; ~8 KB per entry, a course demo won't reach meaningful size; fix adds eviction logic to a shared class | rejected |
| 8 | B | `chunks_per_source` may be ignored at basic depth | false | Tavily docs (fetched 2026-10-09) list it as unavailable only for `ultra-fast`; live runs accepted the request | rejected |
| 9 | B | Social-media pages are cited as news sources | low | Live runs returned Facebook/YouTube posts; Tavily's `exclude_domains` takes a constant list, no new code paths | patch |
| 10 | B, E | Process-wide news client keeps startup settings; missing-key warning fires only on the first news question | low | Same pattern as `market_data.get_client()`; key rotation needs a restart, rare; fix adds invalidation state | rejected |
| 11 | B, E | Concurrent identical searches each spend a credit | low | No in-flight dedupe; needs simultaneous identical questions; fix adds per-key locks | rejected |
| 12 | B, E | Sentinel detection is exact-match: "NO_RELEVANT_NEWS. However … [1]" leaks the token to the user | low | `answer.strip(...) == NO_NEWS_SENTINEL`; with citations it falls through to `apply_citations` and the token is shown. Direct fix | patch |
| 13 | B, E | Advice redirect duplicated when the model quotes or curls it | low | `text.startswith(ADVICE_REDIRECT)` misses `"I can’t…"`; story 5/7 hit the same; tax_education already has the strip-and-prepend regex | patch |
| 14 | E | Host exactly `www.` yields an empty outlet ("Title ()") | low | `_domain` returns `host[4:]` without an empty check. One-token fix | patch |
| 15 | E | URL variants (www, trailing slash, tracking params) are not deduped | low | Rare within 5 results; fix adds URL normalization | rejected |
| 16 | E | No OpenAI key but Tavily key set spends a credit before failing | false | `ask()` builds the OpenAI classifier first and raises `ConfigurationError` before any agent runs | rejected |
| 17 | E | `search()` raises `ValueError` for an empty query, not `NewsUnavailableError` | false | Empty query is a caller error; the agent guards it and `ask()` rejects empty questions (`graph.py:187`) | rejected |
| 18 | B | Story file's test counts unverified; Code Map's `test_portfolio_analysis.py` note unchecked | false | Orchestrator re-ran the suite (557 passed) and the live end-to-end test; `test_portfolio_analysis.py` has no `"news"` usage | rejected |

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass offline, live deselected
- `uv run pytest -q -m live tests/tutor/test_news_synthesizer_live.py` -- expected: all pass
