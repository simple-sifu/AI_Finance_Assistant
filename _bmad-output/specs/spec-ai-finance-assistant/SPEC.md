---
id: SPEC-ai-finance-assistant
companions:
  - stack.md
sources:
  - ../../planning-artifacts/briefs/brief-AIFinanceAssistant-2026-09-29/brief.md
  - ../../planning-artifacts/briefs/brief-AIFinanceAssistant-2026-09-29/addendum.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# AIFinanceAssistant

## Why

**Mandate plus pain.** This is a course assignment due the week of 2026-09-29, built to learn multi-agent orchestration (LangGraph) and RAG end to end.

The product addresses a real pain. Adults who want to invest leave money in low-yield savings because jargon, choice paralysis, and fear of mistakes (often inherited from a relative's loss) block them. Human advisors are expensive and have minimums. Robo-advisors don't teach, and search returns definitions without building confidence.

The product is a patient, grounded tutor that teaches and never advises.

## Capabilities

- **CAP-1**
  - **intent:** A user asks any question in one chat, and it reaches the right one of the six specialist agents.
  - **success:** At least 5 representative questions per agent route to the correct agent. Ambiguous questions get a clarifying or general answer, not an error.
- **CAP-2**
  - **intent:** A user gets beginner-level explanations of investing concepts (e.g., compound interest, ETFs vs mutual funds), grounded in the curated knowledge base.
  - **success:** Answers cite the knowledge-base articles they draw on.
- **CAP-3**
  - **intent:** A user uploads their holdings and learns how diversified and concentrated they are, explained in plain language.
  - **success:** A sample upload yields an analysis that names concentration risk and diversification level.
- **CAP-4**
  - **intent:** A user sees live quotes for a ticker, with an explanation of what the numbers mean.
  - **success:** A ticker query returns a quote plus a plain-language explanation of its figures.
- **CAP-5**
  - **intent:** A user learns how much to save monthly to reach a target amount by a date.
  - **success:** For a given target, horizon, and rate, the monthly amount is arithmetically correct and the compound-interest math is shown.
- **CAP-6**
  - **intent:** A user gets summaries of current financial news.
  - **success:** Every summary cites its source URLs.
- **CAP-7**
  - **intent:** A user learns how 401(k), IRA, and Roth IRA differ as concepts.
  - **success:** Answers are grounded and cited. Situation-specific tax questions are redirected to education, not answered as tax advice.
- **CAP-8**
  - **intent:** Every agent stays on the education side of the education/advice line.
  - **success:** Advice-seeking prompts ("What should I buy?", "How much should I put in X?") get an educational redirect and a disclaimer, never a recommendation.
- **CAP-9**
  - **intent:** Market data keeps working when the Alpha Vantage quota is exhausted or the API is down.
  - **success:** The full demo runs with the day's quota at zero, served from cache or mock data.
- **CAP-10**
  - **intent:** A user works in five areas (Chat, Portfolio, Markets, Goals, Knowledge) and can reach all six agents.
  - **success:** Each agent is demoable end to end from the UI.
- **CAP-11**
  - **intent:** The app is reachable on the public internet.
  - **success:** A grader opens a URL and completes the demo against the deployed container.

## Constraints

- All six agents are mandatory (assignment requirement).
- Deadline: the week of 2026-09-29. Build the market-data cache and mock fallback first, then the router, then the agents.
- Alpha Vantage free tier allows 25 requests/day and 5/minute. Quotes are cached with a 30-minute TTL, and mock data is used as the fallback.
- The app never recommends securities, allocations, amounts, or actions. It may explain concepts, describe common approaches, and calculate with the user's own inputs. Answers carry a disclaimer.
- The stack and hosting are fixed; see `stack.md`.

## Non-goals

- Brokerage connections or moving money.
- User accounts or saved history.
- Personalized recommendations.
- Guided first-visit onboarding, a pretend-money simulator, or a market-crash replay (parked for after the assignment).
- A separately deployed frontend and backend.

## Success signal

- In a live demo against the deployed URL, with the Alpha Vantage quota exhausted, the grader asks one question per agent. Each is routed correctly and answered, with citations where applicable. "What should I buy?" gets an educational redirect with a disclaimer.

## Assumptions

- The stack in `stack.md` is required by the assignment (not confirmed).
- Uploaded portfolios and chat history are session-only and not persisted (this follows from having no accounts).

## Open Questions

- Where do the 50–100 knowledge-base articles come from: written by you, or curated from public sources? And what are the licensing terms?
- What is the portfolio upload format: CSV with which columns (ticker + shares, or ticker + value)?
- News search: Tavily or SerpAPI?
- What is the disclaimer wording, and where does it appear (every answer, or a persistent banner)?
- Do the Portfolio, Markets, Goals, and Knowledge tabs have dedicated forms or views, or are they chat pre-scoped to one agent?
