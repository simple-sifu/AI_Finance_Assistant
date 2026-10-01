---
title: 'LangGraph router with six stub agents and shared advice guardrail'
type: 'feature'
created: '2026-10-01'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '8b738900f3b35fa60c0e9c028f8a5c2386c58f6f'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/stack.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Users ask everything in one chat (CAP-1), but nothing yet decides which of the six agents answers. There is also no shared rule keeping every answer on the education side of the advice line (CAP-8). Stories 3–8 each need a fixed slot and contract to plug a real agent into.

**Approach:** Build a LangGraph graph: an LLM router classifies the latest question into one of six agents, or `clarify`, and flags advice-seeking. Provider errors give an `unavailable` reply, and agent crashes give a failure reply. The graph runs that agent, and a final guardrail node applies the shared disclaimer. Six stub agents implement the shared agent contract and return placeholder answers, so stories 3–8 only swap stubs for real agents.

**Decisions (human, 2026-10-01):**
- Disclaimer: a one-line footer on **every** reply, including `clarify`: "*Educational information only, not financial, tax, or investment advice.*"
- Advice-seeking: route to the matching agent as usual. The reply opens with a short educational redirect (e.g. "I can't tell you what to buy, but here's how to think about it…") and then educates. Stubs return a fixed redirect line in place of education.
- Failure replies (human, 2026-10-01, renegotiated after review; supersedes "classifier failure → clarify"): provider/LLM errors no longer look like "please rephrase". They get a "temporarily unavailable" reply, and agent crashes get a "couldn't finish answering" reply, so a real agent failing never surfaces a traceback.
- Default `OPENAI_MODEL`: `gpt-4o-mini` (human, 2026-10-01; supersedes `gpt-6-luna`). The course OpenAI project can access only gpt-4o, gpt-4o-mini and gpt-5 for chat. `gpt-6-luna` returned 403 `model_not_found` in a live test, and `gpt-4o-mini` routed all five live test questions correctly. It is overridable via env.

## Boundaries & Constraints

**Always:**
- Public entry: `async def ask(question: str, history: list[ChatTurn] | None = None) -> TutorReply`. The reply carries `text`, `route`, `seeks_advice`, and `sources: list[Source]` (empty for stubs).
- Routes: `finance_qa`, `portfolio`, `market`, `goal_planning`, `news`, `tax_education`, `clarify`, `unavailable` (`clarify` and `unavailable` are answered by the tutor itself, not agents).
- One agent contract (protocol plus registry keyed by route). Stories 3–8 replace a stub by registering a real agent, without editing the router or graph.
- The router sees the latest question plus up to the last 6 history turns, so follow-ups like "what about Roth?" route correctly.
- The education-not-advice rule is one shared system-prompt text every agent imports. The guardrail node, not the agents, adds the disclaimer.
- Unparseable classifier output routes to `clarify`; classifier exceptions route to `unavailable`; agent exceptions produce the agent-failure reply. `ask` never raises for any of them (only for empty input, malformed history, or a missing key).
- Safe across `asyncio.run()` calls (Streamlit): no LLM or HTTP client is cached across event loops; build them per `ask` call.
- `OPENAI_API_KEY` and `OPENAI_MODEL` are read through `Settings`. A missing key raises a clear configuration error when `ask` is called.
- Tests never call OpenAI: the classifier is injectable, and tests use a fake.

**Never:** No real agent logic, RAG, market calls, or UI (stories 3–9). No evaluation question set (story 10). No conversation persistence.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Clear topic | "What is compound interest?" | `route=finance_qa`; stub answer; disclaimer applied | N/A |
| Each agent | One representative question per route | Matching route and stub | N/A |
| Follow-up | history about IRAs + "what about Roth?" | `route=tax_education` | N/A |
| Advice-seeking | "What stock should I buy?" | `seeks_advice=true`; reply is an educational redirect, never a recommendation; disclaimer applied | N/A |
| Ambiguous / off-topic | "hi", "what's the weather?" | `route=clarify`; reply lists what the tutor can help with | N/A |
| Classifier bad output | classifier returns an invalid route or wrong type | `route=clarify` | Logged warning |
| Classifier/provider error | classifier raises (OpenAI outage, timeout, 429, 401 bad key, network) | `route=unavailable`; reply: "Sorry, the tutor is temporarily unavailable. Please try again in a moment."; disclaimer applied | Warning with exception type only; a 401/403 logs an error naming `OPENAI_API_KEY` (never the key) |
| Agent failure | the routed agent's `run()` raises or returns a non-`AgentResult` | `route` stays the chosen agent; reply: "Sorry, I couldn't finish answering that just now. Please try again in a moment."; `sources=[]`; disclaimer applied; `ask` does not raise | Warning with route and exception type only |
| Empty input | "" or whitespace | `ValueError` before any LLM call | N/A |
| No API key | `OPENAI_API_KEY` unset | Configuration error naming the variable | Key never logged |

</frozen-after-approval>

## Code Map

- `src/finance_assistant/config.py` -- `Settings`/`load_settings`; add the OpenAI fields, following the existing validation and key-hiding `__repr__` pattern.
- `src/finance_assistant/market_data/__init__.py` -- example of the public-API `__all__` pattern to follow. Market data is not used in this story.
- `tests/conftest.py` -- `_isolated_config` env-var list and `_no_network` guard; extend for the new env vars.
- `pyproject.toml` -- add `langgraph`, `langchain-core`, `langchain-openai` (current: 1.2.x, 1.6.x, 1.6.x).

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml`, `uv.lock`, `.env.example`, `README.md` -- add deps; document `OPENAI_API_KEY` and `OPENAI_MODEL`.
- [x] `src/finance_assistant/config.py`, `tests/conftest.py`, `tests/test_config.py` -- OpenAI settings with a hidden key; isolate the new vars in tests.
- [x] `src/finance_assistant/tutor/models.py` -- `Route`, `ChatTurn`, `Source`, `AgentResult`, `TutorReply`.
- [x] `src/finance_assistant/tutor/guardrail.py` -- shared education-not-advice system-prompt text, disclaimer text, and the guardrail function.
- [x] `src/finance_assistant/tutor/agents.py` -- agent protocol, registry, six stub agents, and the deterministic `clarify` responder.
- [x] `src/finance_assistant/tutor/router.py` -- classifier protocol plus an OpenAI implementation with structured output (`route`, `seeks_advice`).
- [x] `src/finance_assistant/tutor/graph.py`, `tutor/__init__.py` -- LangGraph graph (router → agent → guardrail) and the public `ask`.
- [x] `tests/tutor/` -- every I/O matrix row with a fake classifier; consecutive `asyncio.run(ask(...))` calls.

**Acceptance Criteria:**
- Given the dev extra and no `.env`, when `pytest` runs, then all tests pass offline, including story 1's.
- Given a test registers a custom agent for `market`, when a market question is asked, then that agent answers without router or graph edits.
- Given two consecutive `asyncio.run(ask(...))` calls, when both run, then both succeed.

## Implementation Notes

## Spec Change Log

## Review Triage Log

| # | Source | Finding | Verdict | Evidence / route |
|---|---|---|---|---|
| 1 | blind | OpenAI outage, 401 or 429 routes to `clarify` ("please rephrase") | medium | Real: misleading reply when OpenAI fails. But the frozen matrix mandates classifier failure → `clarify`, so the fix edits this spec. Rejected for this story; surfaced to the human and recorded in deferred-work. |
| 2 | blind | Real-classifier tests depend on the catch-all swallowing the network-guard error, so accidental network use passes silently | medium | Verified: `test_key_never_logged_or_in_error` and `test_consecutive_asyncio_run_calls_with_real_classifier` assert `clarify` caused by the guard's `RuntimeError`. **patch** (respx-mock the endpoint) |
| 3 | blind | "never in error" test doesn't check reply text or exceptions | low | Same test as #2; fixed in the same patch. **patch** (grouped with #2) |
| 4 | blind | Agent exceptions propagate out of `ask` | maybe-false | Unreachable in this story: stubs never raise. Becomes reachable in story 3, where the failure reply is a behavior choice. Would be medium. **defer** |
| 5 | blind | Advice redirect relies on agents following the prompt | low | Matches the human decision (agents open with the redirect; guardrail enforces the disclaimer only). Rejected. |
| 6 | blind | `register_agent` accepts a synchronous `run` | low | Verified: `runtime_checkable` checks only that the attribute exists, so it fails on the first routed question. Direct check. **patch** |
| 7 | blind | Latest question length is unbounded | low | Unlikely in a demo; the fix adds a limit parameter. Rejected. |
| 8 | blind | Disclaimer dedupe only strips an exact trailing copy | low | Violates the "footer exactly once" invariant if an agent's text contains it mid-text. Direct correction. **patch** |
| 9 | blind | History content can forge turns in the router prompt | low | Only affects routing of the user's own question; there is no other user's data. Rejected. |
| 10 | blind | `pydantic` imported but not declared | low | Verified: `router.py` imports it; it arrives only transitively. One-line dependency. **patch** |
| 11 | blind | Frozen dataclasses hold mutable `list` sources | low | Changing to tuple alters public types; no named harm. Rejected. |
| 12 | blind | No overall latency deadline | low | Per-call 30 s timeout + 1 retry; acceptable for a demo. Rejected. |
| 13 | blind | Default model literal repeated in tests/docs | low | `test_config` intentionally pins the human-chosen default. Rejected. |
| 14 | edge | Agent `run()` raising propagates (same as #4) | maybe-false | Grouped with #4. **defer** |
| 15 | edge | `AgentResult` with non-str text crashes the guardrail | low | Programmer error failing loudly is correct behavior. Rejected. |
| 16 | edge | Sync `run` registers (same as #6) | low | **patch** (grouped with #6) |
| 17 | edge | Disclaimer mid-text or repeated (same as #8) | low | **patch** (grouped with #8) |
| 18 | edge | `chat_model(**kwargs)` collides on `timeout`/`max_retries` | low | Verified: passing `timeout=` raises "multiple values". Stories 3–8 will tune these. Direct merge fix. **patch** |
| 19 | edge | Whitespace-only `openai_api_key` via direct `Settings(...)` | low | `load_settings` strips; only direct construction is affected. Rejected. |
| 20 | edge | Non-iterable history raises `TypeError`, not `ValueError` | low | Loud failure on caller error. Rejected. |
| 21 | edge | Long question → context error → `clarify` (same as #7) | low | Rejected. |
| 22 | edge | `pydantic` undeclared (same as #10) | low | **patch** (grouped with #10) |
| 23 | edge | Tests rely on the socket guard plus retries (same as #2) | medium | **patch** (grouped with #2) |
| 24 | verification-gap | Invalid route with `seeks_advice=True` loses the redirect untested | medium | Pre-verified by mutation: all tutor tests still passed. **patch** |
| 25 | verification-gap | Non-`ChatTurn` history `ValueError` untested; bad `ChatTurn` role untested | medium | Pre-verified: removing the check silently degrades to `clarify`. **patch** |
| 26 | verification-gap | 500-char history clipping untested | low | Pre-verified; the filed disposition was defer, but it is a trivial test in a file already being patched. **patch** |
| 27 | verification-gap (other) | `test_consecutive_asyncio_run_calls_with_real_classifier` docstring overstates what it proves | low | Same root as #2. **patch** (grouped with #2) |
| 28 | verification-gap (other) | Empty `openai_model` rejection untested | low | Unreachable via `load_settings`. Rejected. |

## Verification

**Commands:**
- `uv run --extra dev pytest -q` -- expected: all tests pass offline.
