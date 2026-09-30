---
title: "Product Brief: AIFinanceAssistant"
status: final
created: 2026-09-29
updated: 2026-09-30
---

# Product Brief: AIFinanceAssistant

## Executive Summary

AIFinanceAssistant is a conversational tutor for adults who want to start investing but feel shut out of it. A user asks questions in plain language, in one chat. Behind that chat, six specialist AI agents answer: they explain concepts, analyze an uploaded portfolio, interpret live market data, plan savings goals, summarize financial news, and explain retirement account types. Answers draw on a curated knowledge base, so they rest on vetted content rather than on whatever the model improvises.

The product teaches; it does not advise. It explains how investing works and runs the math on the user's own numbers, but it never tells anyone what to buy, sell, or how much to put in. That line is a real regulatory concern in fintech, and the product enforces it in every answer.

This is a course assignment, due the week of 2026-09-29. Its purpose is to learn multi-agent orchestration (LangGraph) and retrieval-augmented generation (RAG) end to end, in a product that is realistic but deliberately small on infrastructure.

## The Problem

Millions of people leave money in low-yield savings accounts even though they want it to grow. What stops them is not a lack of money. Three things stand in the way:

- **Jargon.** Terms like ETF, expense ratio, and Roth read as an insider language.
- **Choice paralysis.** There are thousands of funds and no obvious place to start.
- **Fear of mistakes.** This is often inherited, for example from a relative who lost money. Without the vocabulary, the beginner can't tell why the relative lost it, so "investing" as a whole feels dangerous.

The current options fail beginners in different ways. Human advisors are expensive and often require account minimums. Robo-advisors put money into products without teaching the concepts behind them. Search engines return definitions, and definitions don't build confidence. What's missing is a patient, always-available explainer that meets the beginner at their level and grounds every answer in trustworthy sources.

## The Solution

A single chat interface backed by six specialist agents. A **LangGraph router** classifies each question and sends it to the right agent:

| Agent | What it does for the beginner |
|---|---|
| **Finance Q&A** | Explains concepts (compound interest, ETFs vs mutual funds), grounded by RAG over a curated knowledge base of 50–100 articles |
| **Portfolio Analysis** | Reviews uploaded holdings for diversification and concentration risk, and explains what the results mean |
| **Market Analysis** | Pulls live quotes (Alpha Vantage) and explains what the numbers mean |
| **Goal Planning** | Calculates the monthly saving needed to reach a target, and shows the compound-interest math |
| **News Synthesizer** | Summarizes current financial news (Tavily or SerpAPI), with sources cited |
| **Tax Education** | Explains 401(k) vs IRA vs Roth IRA as concepts, never as tax advice for the user's situation |

The UI has five areas: **Chat, Portfolio, Markets, Goals, and Knowledge**.

**Education, not advice.** Every agent follows one rule. It may explain concepts, describe how people commonly approach something, and calculate with the user's own inputs. It must not recommend specific securities, allocations, or actions. Answers include a disclaimer, and requests for advice are redirected to education. `[ASSUMPTION: exact wording and placement of disclaimers to be settled in the PRD]`

## Who This Serves

- **The beginner investor (primary).** An adult who is employed and has savings, for example $5,000 sitting in a savings account. They want it to grow but are intimidated. Success for them means understanding a concept well enough to explain it back, and feeling less afraid of the next step.
- **The course evaluator (secondary, and the real audience this week).** They need to see correct multi-agent routing, grounded RAG answers, working external integrations, and a demo that doesn't break.

## Success Criteria

1. **Routing works.** A test set of representative questions, at least 5 per agent, reaches the correct agent, and ambiguous questions are handled gracefully.
2. **Answers are grounded.** Finance Q&A and Tax Education answers cite knowledge-base articles. News answers cite their sources.
3. **The demo survives API limits.** Alpha Vantage allows 25 requests/day and 5/minute. With a 30-minute TTL cache and mock-data fallback, the full demo runs even when the quota is exhausted or an API is down.
4. **The advice line holds.** Prompts like "What should I buy?" get an educational redirect and a disclaimer, never a recommendation.
5. **All six agents are reachable from the UI and demoable end to end** by the deadline.

## Scope

**In (v1, this week):**
- the six agents
- the LangGraph router
- the RAG knowledge base (50–100 articles; FAISS; all-MiniLM-L6-v2 embeddings)
- OpenAI as the LLM
- Alpha Vantage and Tavily/SerpAPI integrations, with caching and mock fallbacks
- portfolio file upload
- the five UI areas
- disclaimers
- async Python throughout
- one working deployment

`[ASSUMPTION: the stack listed here is required by the assignment]`

**Out (explicitly):**
- real brokerage connections or money movement
- user accounts and saved history
- personalized recommendations
- the brainstormed guided-onboarding journey, investing simulator, and "time machine" crash replay (parked for a later version)

## Key Risks

| Risk | Mitigation |
|---|---|
| Alpha Vantage's 25 requests/day cap kills the live demo (the biggest risk) | Build the cache and mock fallback **first**, before any agent logic depends on live data |
| PyTorch + FAISS are too heavy for small or serverless hosting (size limits, slow cold starts) | Run as a long-running container on AWS Lightsail or EC2 with ≥2 GB RAM and CPU-only torch; see addendum |
| Answers drift into advice | A shared system-prompt rule for all agents, plus advice-seeking test prompts in the evaluation set |
| Scope creep against a one-week deadline | Ship the six agents first; add polish only after all success criteria pass |

## Delivery Decision

**Frontend and hosting: Streamlit in a single Docker container on AWS (Lightsail or EC2, ≥2 GB RAM).** The one-week deadline decided it. A single deployable keeps the whole week on the six agents instead of on frontend and integration work.

## Beyond the Assignment

If the project continues, the brainstorm pointed to a sharper insight. Beginners can't see a **low-risk path between a savings account and real investing**. The next version would add that path next to the chat:

- a guided first-visit experience
- a pretend-money simulator
- a replay of a real market crash and its recovery, so users can feel a drop safely
