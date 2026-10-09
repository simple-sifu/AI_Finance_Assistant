"""The tutor: route each question to one of six agents and keep answers educational."""

from __future__ import annotations

from .agents import (
    AGENT_FAILURE_TEXT,
    UNAVAILABLE_TEXT,
    Agent,
    StubAgent,
    get_agent,
    install_real_agents,
    register_agent,
    reset_agents,
)
from .graph import ask, build_graph
from .guardrail import (
    ADVICE_REDIRECT,
    DISCLAIMER,
    EDUCATION_SYSTEM_PROMPT,
    apply_guardrail,
    build_system_prompt,
)
from .models import (
    AGENT_ROUTES,
    ROUTES,
    AgentRequest,
    AgentResult,
    ChatTurn,
    Classification,
    Route,
    Source,
    TutorReply,
)
from ..goals import SavingsPlan, monthly_savings
from ..portfolio import HoldingsFormatError, Portfolio, parse_holdings_csv
from .goal_planning import GoalPlanningAgent
from .market_analysis import MarketAnalysisAgent
from .news_synthesizer import NewsSynthesizerAgent
from .portfolio_analysis import PortfolioAnalysisAgent
from .tax_education import TaxEducationAgent
from .router import AdviceReviewer, Classifier, OpenAIAdviceReviewer, OpenAIClassifier

__all__ = [
    "ADVICE_REDIRECT",
    "AGENT_FAILURE_TEXT",
    "AGENT_ROUTES",
    "AdviceReviewer",
    "DISCLAIMER",
    "EDUCATION_SYSTEM_PROMPT",
    "GoalPlanningAgent",
    "HoldingsFormatError",
    "MarketAnalysisAgent",
    "NewsSynthesizerAgent",
    "ROUTES",
    "SavingsPlan",
    "Agent",
    "AgentRequest",
    "AgentResult",
    "ChatTurn",
    "Classification",
    "Classifier",
    "OpenAIAdviceReviewer",
    "OpenAIClassifier",
    "Portfolio",
    "PortfolioAnalysisAgent",
    "Route",
    "Source",
    "StubAgent",
    "TaxEducationAgent",
    "TutorReply",
    "UNAVAILABLE_TEXT",
    "apply_guardrail",
    "ask",
    "build_graph",
    "build_system_prompt",
    "get_agent",
    "install_real_agents",
    "monthly_savings",
    "parse_holdings_csv",
    "register_agent",
    "reset_agents",
]
