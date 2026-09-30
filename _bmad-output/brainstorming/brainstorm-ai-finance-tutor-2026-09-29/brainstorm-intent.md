# Brainstorm Intent — AI Finance Assistant

Source: `.memlog.md` in this folder (Five Whys session, 2026-09-29, facilitator mode, ended early by choice to move to the product brief).

## Original concept (user's input brief)
- **Problem:** Adults want to invest but are blocked by jargon, choice paralysis and fear of mistakes. Human advisors are expensive and have minimums; robo-advisors push products without teaching. Money sits in low-yield savings.
- **Solution:** A multi-agent conversational tutor. Six specialist agents (Finance Q&A, Portfolio Analysis, Market Analysis, Goal Planning, News Synthesizer, Tax Education) behind one chat, with a LangGraph router and RAG over 50–100 vetted articles. It explicitly separates education from advice and carries disclaimers.
- **Purpose:** A student learning project for multi-agent patterns and RAG with low infrastructure overhead.
- **Stack in the brief:** LangGraph, LangChain, OpenAI, FAISS, sentence-transformers (all-MiniLM-L6-v2), Streamlit, async Python.
- **External services:** OpenAI; Alpha Vantage (free tier: 25 requests/day, 5/min); SerpAPI (100/month) or Tavily (1,000/month).
- **Stack the user wants:** a **Next.js + FastAPI monorepo deployed on Vercel** instead of Streamlit.

## Key discovery: sharper root cause
Five Whys chain: closes the tab → fears investing "the wrong way" → a family member lost money → can't diagnose why without the jargon → believes investing needs "finance experience" → **can't see a slow, low-risk beginner lane between a savings account and "real investing".**

The barrier is less missing information than a missing *visible path* and missing *safe experience*. Definitions alone don't create comfort.

**Beginner lane (agreed definition):** small, safe steps from savings to understood investing. The app can **reveal** on-ramps that already exist and **create** a practice shallow end.

## Experience ideas generated (the user's, grouped)
- **Lower the temperature:** name the fear up front; "nothing tonight moves real money"; no sign-up wall; pool visual (shallow, middle, deep end) marking "you're at the edge".
- **Placement questions:** where the money lives now; when it's needed (<1 year / 1–5 years / long term), which sorts cash from investable money; biggest worry, which sets the tone; "has anyone close to you been burned?"; existing 401(k) ("you may already be an investor").
- **Reveal the lane:** on-ramp ladder (HYSA → T-bills/CDs → 401(k) match → Roth IRA → broad index fund), each labelled by how scary it actually is; a 10-year comparison slider for their $5,000 that shows volatility honestly; "a slice of 500 companies, not a bet on one"; explicitly separate the deep end (single stocks, options, crypto, day trading).
- **Create the lane (practice):** a pretend-$5,000 simulator; a **time machine** that replays 2008 or 2020 and then the recovery; a "sell / hold / buy more?" moment with historical outcomes; a mock brokerage order screen; a 20-second micro-lesson per new term.
- **Tiny first step:** split into an emergency bucket and a 5+ year bucket; a $50–100 "tuition" amount; a "do nothing yet" path (moving to a HYSA counts); a checklist with one box pre-checked.
- **Trust:** a "how we make money / we never hold funds" corner; a scam-spotting card; peer stories rather than influencers; a respected "I'm not ready" button with one gentle reminder.
- **Leave-behind:** a one-page summary (buckets, position in the lane, next step); a dated next step; a pretend portfolio that persists as a reason to return.

## Open decisions for the product brief
1. **Product shape:** the designed experience is a *guided journey plus simulator*, while the original is *six agents behind a chat*. Which one is the product, and where does chat and multi-agent routing earn its place? (This matters because the learning goal is multi-agent and RAG.)
2. **Education vs advice line:** some ideas ("Next month, try $100 in an index fund", the on-ramp ladder as a sequence) drift toward personalized recommendations. Where exactly is the line?
3. **Deployment on Vercel:** Python functions are capped at ~250 MB unzipped, and sentence-transformers/PyTorch plus FAISS exceed that. There's no persistent in-memory cache, so a 30-minute TTL cache needs an external store. Duration limits affect long LangGraph runs and streaming.
4. **Data sources:** Alpha Vantage's 25 requests/day cannot feed a simulator or a 2008/2020 time machine. Historical data needs a static or bundled source; T-bill and HYSA rates need a source. Mock and cached fallbacks remain the top demo risk.
5. **Scope:** for a student project, which slice is the MVP (for example, placement → lane → time machine → a grounded Q&A agent), and which of the six agents are v2?
6. **Persistence:** pretend portfolios, progress and email-later all imply accounts and storage, which were not in the original brief.
