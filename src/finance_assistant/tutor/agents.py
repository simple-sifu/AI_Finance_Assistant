"""The agent contract, the route-keyed registry, six stub agents, and ``clarify``.

Stories 3-8 swap a stub for a real agent with ``register_agent(route, agent)``;
the router and graph look agents up at run time, so they need no edits.

Real agents that call an LLM must build their system prompt with
``guardrail.build_system_prompt`` and must not cache LLM/HTTP clients across
calls (see ``tutor.llm.chat_model``).
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .guardrail import ADVICE_REDIRECT
from .models import AGENT_ROUTES, AgentRequest, AgentResult, Route


@runtime_checkable
class Agent(Protocol):
    """One specialist agent. ``run`` returns the answer without the disclaimer."""

    async def run(self, request: AgentRequest) -> AgentResult: ...


@dataclass(frozen=True)
class StubAgent:
    """Placeholder agent: a fixed answer until the real agent is registered."""

    name: str
    topic: str

    async def run(self, request: AgentRequest) -> AgentResult:
        if request.seeks_advice:
            return AgentResult(text=ADVICE_REDIRECT)
        return AgentResult(
            text=(
                f"[{self.name} — coming soon] This agent will explain {self.topic}. "
                "For now this is a placeholder answer."
            )
        )


_STUBS: dict[Route, StubAgent] = {
    "finance_qa": StubAgent("Finance Q&A", "general investing concepts"),
    "portfolio": StubAgent("Portfolio Analysis", "how diversified your holdings are"),
    "market": StubAgent("Market Analysis", "live quotes and what the numbers mean"),
    "goal_planning": StubAgent("Goal Planning", "how much to save to reach a goal"),
    "news": StubAgent("News Synthesizer", "current financial news, with sources"),
    "tax_education": StubAgent("Tax Education", "401(k), IRA, and Roth IRA concepts"),
}

_registry: dict[Route, Agent] = dict(_STUBS)
_registry_lock = threading.Lock()


def register_agent(route: Route, agent: Agent) -> None:
    """Make ``agent`` answer questions routed to ``route`` (one of the six agent routes)."""
    if route not in AGENT_ROUTES:
        raise ValueError(f"route must be one of {list(AGENT_ROUTES)}, got {route!r}")
    if not isinstance(agent, Agent) or not inspect.iscoroutinefunction(agent.run):
        raise TypeError("agent must have an async run(request) -> AgentResult method")
    with _registry_lock:
        _registry[route] = agent


def get_agent(route: Route) -> Agent:
    """Return the agent registered for ``route``."""
    with _registry_lock:
        try:
            return _registry[route]
        except KeyError:
            raise ValueError(f"no agent for route {route!r}") from None


def install_real_agents() -> None:
    """Register the real agents built so far in place of their stubs.

    Opt-in: the app calls this at startup, while tests keep the offline stubs
    unless they call it. Stories 4-8 add their agents here. Importing an agent is
    cheap; its heavy resources (e.g. the embedding model) load on first use.
    """
    from .finance_qa import FinanceQAAgent
    from .goal_planning import GoalPlanningAgent
    from .market_analysis import MarketAnalysisAgent
    from .portfolio_analysis import PortfolioAnalysisAgent
    from .tax_education import TaxEducationAgent

    register_agent("finance_qa", FinanceQAAgent())
    register_agent("portfolio", PortfolioAnalysisAgent())
    register_agent("market", MarketAnalysisAgent())
    register_agent("goal_planning", GoalPlanningAgent())
    register_agent("tax_education", TaxEducationAgent())


def reset_agents() -> None:
    """Restore the six stubs (for tests)."""
    with _registry_lock:
        _registry.clear()
        _registry.update(_STUBS)


CLARIFY_TEXT = """\
I'm not sure what you're asking. I'm a finance tutor, and I can help with:
- **Investing concepts**: compound interest, ETFs vs mutual funds, risk, and more
- **Your portfolio**: how diversified or concentrated your holdings are
- **Markets**: live quotes for a ticker and what the numbers mean
- **Goals**: how much to save each month to reach a target
- **News**: summaries of current financial news, with sources
- **Tax-advantaged accounts**: how 401(k), IRA, and Roth IRA differ

Could you rephrase your question around one of these?"""


def clarify(request: AgentRequest) -> AgentResult:
    """Deterministic reply for ambiguous or off-topic questions."""
    text = f"{ADVICE_REDIRECT}\n\n{CLARIFY_TEXT}" if request.seeks_advice else CLARIFY_TEXT
    return AgentResult(text=text)


UNAVAILABLE_TEXT = "Sorry, the tutor is temporarily unavailable. Please try again in a moment."
AGENT_FAILURE_TEXT = "Sorry, I couldn't finish answering that just now. Please try again in a moment."
# Follows ADVICE_REDIRECT when the guardrail's reviewer withholds a reply that gave advice.
ADVICE_REVIEW_NOTE = (
    'If you like, ask me how something works instead, such as "How does an index fund work?" '
    "and I'll explain the concepts so you can decide for yourself."
)


def unavailable(request: AgentRequest) -> AgentResult:
    """Deterministic reply when the classifier or LLM provider fails."""
    return AgentResult(text=UNAVAILABLE_TEXT)
