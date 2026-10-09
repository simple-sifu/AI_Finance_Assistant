---
title: 'Tax Education agent'
type: 'feature'
created: '2026-10-08'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: 'bf6e915f419f837ba59774604284c56af8a034f3'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/stories/3-knowledge-base-faiss-index-and-finance-qa-agent.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The `tax_education` route still returns a stub. CAP-7 needs grounded, cited explanations of how 401(k), IRA and Roth IRA differ as concepts, and situation-specific tax questions must be redirected to general education instead of being answered as tax advice.

**Approach:** Reuse the story 3 retrieval-and-citation pipeline. Generalize `FinanceQAAgent` just enough that a `TaxEducationAgent` can supply its own instructions, retrieval scope and "not covered" reply, then register it in `install_real_agents()`. Its replies pass through the existing advice review like any real agent's.

## Boundaries & Constraints

**Always:** Answer only from retrieved excerpts with inline `[n]` citations; return only cited articles as sources; never answer without a source (same rules and thresholds as story 3). Advice-seeking questions open with `ADVICE_REDIRECT`. State contribution limits and other figures with the tax year the article gives. Finance Q&A behavior and its tests stay unchanged.

**Never:** Changing the router, guardrail or review behavior. Adding or editing articles, or rebuilding the index format. Calculating a user's tax, deduction or eligibility. A second vector store or index.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Concept comparison | "What's the difference between a Traditional IRA and a Roth IRA?" | Plain-language comparison citing IRS/Investor.gov IRA articles | N/A |
| 401(k) vs IRA | "How is a 401(k) different from an IRA?" | Cited explanation (who offers it, limits, tax treatment) | N/A |
| Situation-specific | "I earn $150k and have a 401(k) at work. Can I deduct my IRA contribution?" | Opens with the tax-situation redirect, explains the general deduction rules from the articles, suggests a qualified tax professional; no verdict for this user | N/A |
| Advice-seeking | "Should I open a Roth IRA?" (`seeks_advice=True`) | Opens with `ADVICE_REDIRECT` (not both redirects), then educates with citations | N/A |
| Nothing relevant | Tax question the articles don't cover (e.g. "How do I file my state taxes?") | Tax-specific "my articles don't cover that" reply listing the account topics covered; `sources=[]`, no LLM call when below threshold | N/A |
| Agent failure | OpenAI error during the answer | Graph's existing "couldn't finish answering" reply | caught by `_agent_node` |

**Decisions (human, 2026-10-08):** Retrieval searches the whole knowledge base, like Finance Q&A, so tax questions touched by non-tax articles (capital gains, annuities, REIT dividends) are answered too. Situation-specific tax questions open with exactly this sentence: "I can't give tax advice for your specific situation, but I can explain how this works in general." Then the reply gives general education from the articles and suggests a qualified tax professional. The model decides when the redirect applies; the live eval checks it. When a question is advice-seeking, `ADVICE_REDIRECT` is used instead, never both.

</frozen-after-approval>

## Code Map

- `src/finance_assistant/tutor/finance_qa.py` -- `FinanceQAAgent` (`run` :269, `_retrieve` :265), `FINANCE_QA_INSTRUCTIONS`, `NOT_COVERED_TEXT`/`TOPICS`, `not_covered`, `retrieval_queries`, `merge_hits`, `group_hits`, `build_prompt`, `apply_citations`. Reuse all helpers unchanged; only make the instructions and the not-covered reply overridable per agent.
- `src/finance_assistant/tutor/agents.py` -- `install_real_agents()` :79 registers real agents; `_STUBS["tax_education"]` stays for tests.
- `src/finance_assistant/tutor/guardrail.py` -- `ADVICE_REDIRECT`, `build_system_prompt` (already says situation-specific tax questions get general education). Do not change.
- `src/finance_assistant/tutor/router.py` -- `ROUTER_SYSTEM_PROMPT` already routes 401(k)/IRA/Roth/HSA/529 questions to `tax_education`. Do not change.
- `tests/tutor/test_finance_qa.py` -- patterns to copy: `kb` fixture with `HashEmbedder`, `FakeLLM` + respx, `real_agents` fixture, `ask` with `FakeClassifier`/`AllowAllReviewer`.
- `tests/tutor/test_finance_qa_live.py` -- live eval pattern (`-m live`, real index/model/reviewer).
- `knowledge_base/articles/` -- 23 `retirement-and-tax` articles (IRS.gov, Investor.gov, one Wikipedia): IRAs, Roth, 401(k), 403(b)/457(b), contribution and deduction limits, RMDs, rollovers, catch-up, Saver's Credit, HSA, 529.

## Tasks & Acceptance

**Execution:**
- [x] `src/finance_assistant/tutor/finance_qa.py` -- move instructions and not-covered text onto overridable class attributes of `FinanceQAAgent`; behavior for Finance Q&A identical -- lets the tax agent reuse the pipeline without copying it
- [x] `src/finance_assistant/tutor/tax_education.py` (new) -- `TaxEducationAgent`: tax instructions (concepts not advice, exact situation redirect sentence as a constant, cite the tax year for limits), tax-specific not-covered reply
- [x] `src/finance_assistant/tutor/agents.py`, `tutor/__init__.py` -- register and export `TaxEducationAgent` in `install_real_agents()`
- [x] `tests/tutor/test_tax_education.py` (new) -- every matrix row offline (HashEmbedder, respx), plus `ask` routing to `tax_education` with sources kept through review
- [x] `tests/tutor/test_tax_education_live.py` (new) -- `@pytest.mark.live`: 5 CAP-7 questions (incl. one situation-specific, one advice-seeking) cite an expected article and pass the real advice reviewer

**Acceptance Criteria:**
- Given `install_real_agents()`, when `get_agent("tax_education")` is called, then it returns a `TaxEducationAgent`; after `reset_agents()` it is the stub again.
- Given the story 3 test suite, when it runs after this change, then every Finance Q&A test passes unchanged.
- Given a real key, when `ask("How is a Roth IRA different from a traditional IRA?")` runs, then the route is `tax_education`, at least one IRA article is cited, and the disclaimer appears once.

## Implementation Notes

- **Generalization:** `FinanceQAAgent` gained class attributes `name` (log prefix), `instructions` (default `FINANCE_QA_INSTRUCTIONS`) and `not_covered_text` (default `NOT_COVERED_TEXT`); `not_covered(request, text=NOT_COVERED_TEXT)` takes the reply text. Finance Q&A prompts, replies and tests are unchanged; only its log messages now come from `name` (same wording).
- **Tax agent:** `TaxEducationAgent(FinanceQAAgent)` overrides the three attributes. `TAX_SITUATION_REDIRECT` holds the exact sentence; the instructions tell the model when to open with it, to state the tax year and account type for each figure (most recent year first), to never compute the user's tax/deduction/eligibility, and to use only the advice redirect when asked for one. `run` also strips the situation sentence from advice-seeking replies, so "never both" holds even if the model emits both.
- **Not covered:** `TAX_NOT_COVERED_TEXT` lists the account topics (`TAX_TOPICS`) the retirement-and-tax articles explain; same threshold and gates as story 3.
- **Exports:** `TaxEducationAgent` is exported from `finance_assistant.tutor`; importing it stays light (no torch/faiss/sentence-transformers at import).
- Verification (2026-10-08): 293 offline tests pass (19 new); live tax eval 7/7 on gpt-4o-mini (5 CAP-7 cases, the `ask` end-to-end case, and "How do I file my state taxes?" not covered); story 3 live eval still 8/8.

- Post-review verification (2026-10-08): 297 offline tests pass; live tax eval 7/7 on gpt-4o-mini after the review patches.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-08): layers blind (B), edge-case (E), verification-gap (V).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | B | Model output "{tax} {advice} body" ends up with the advice redirect twice after stripping | medium | Reproduced: the base class prepends `ADVICE_REDIRECT`, then `_SITUATION_REDIRECT_RE` removes the tax sentence and leaves "ADVICE ADVICE Body [1]." | patch |
| 2 | B, E | The tax-redirect strip needs an exact match; a curly apostrophe gets past it, giving both redirects | low | Reproduced: `TAX_SITUATION_REDIRECT in text` is False for `can’t`. Same root as #1 (fragile never-both handling); fix is a direct regex correction | patch (with #1) |
| 3 | E | Removed redirect may leave a hanging citation, or an advice-only reply with sources | low | Needs the model to cite the redirect sentence or write nothing else; fix adds branches | rejected |
| 4 | B, E | Non-advice path: the tax redirect is not moved to the front or deduplicated if the model misplaces or repeats it | low | Human decision: the model decides when the redirect applies; misplacement is rare and the fix adds logic | rejected |
| 5 | B | "{tax}\n\nNOT_COVERED" falls to not-covered via the "cited no article" warning, not the sentinel log | low | The user-facing result is correct (not-covered, no sources); only the log line differs | rejected |
| 6 | B | `TAX_TOPICS` not checked against the knowledge base; the list can drift | low | No test maps topics to articles; story 3 patched the same gap for `TOPICS` (#9); the fix is a test only | patch |
| 7 | B | Tax not-covered reply lists only account topics, not capital gains/annuities | low | By design: those topics are thin (one capital-gain article); the list names what the tax articles explain | rejected |
| 8 | B, E | Approach says subclasses supply "retrieval scope", but no scope hook exists | false | The frozen human decision chooses whole-KB retrieval like Finance Q&A, so no scope override is required; the fix would edit the spec | rejected |
| 9 | B | Tautological assertions (constant equals itself, "tax year" in prompt) | low | Harmless; the "Finance Q&A unchanged" guarantee is already pinned by story 3 tests (`test_finance_qa.py:105-106, 200, 221, 239`) | rejected |
| 10 | B | Live eval calls `load_articles()` at import; missing slug gives a bare KeyError | low | Same pattern as `test_finance_qa_live.py`; reads 76 small files, negligible | rejected |
| 11 | B | No live case that is both situation-specific and advice-seeking | low | The never-both rule is enforced in code and pinned offline (both orders after #1), independent of model behavior | rejected |
| 12 | B | Live eval does not check that the situation answer avoids a personal tax verdict | low | Real gap, but checking needs an LLM judge (new logic); the advice reviewer passed the live answer. Surfaced to the human | rejected (surfaced to human) |
| 13 | B | `not_covered` can prepend the advice redirect to text that already starts with it | false | Its only callers pass `self.not_covered_text` constants, none of which start with `ADVICE_REDIRECT` | rejected |
| 14 | E | Situation-specific question answered without the redirect if the model omits it | low | Same as #4: by the human decision the model decides; the live eval checks it | rejected |
| 15 | V | No test that `import finance_assistant.tutor` stays free of torch/faiss/sentence-transformers, now that `tutor/__init__.py` imports `tax_education` → `knowledge` | low | Pre-verified by the layer: only lazy imports keep them out and nothing in the suite would notice a regression | patch |

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass offline, live deselected
- `uv run pytest -q -m live tests/tutor/test_tax_education_live.py` -- expected: all cited and reviewed
