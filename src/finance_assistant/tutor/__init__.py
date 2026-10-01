"""The tutor: route each question to one of six agents and keep answers educational."""

from __future__ import annotations

from .agents import (
    AGENT_FAILURE_TEXT,
    UNAVAILABLE_TEXT,
    Agent,
    StubAgent,
    get_agent,
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
from .router import Classifier, OpenAIClassifier

__all__ = [
    "ADVICE_REDIRECT",
    "AGENT_FAILURE_TEXT",
    "AGENT_ROUTES",
    "DISCLAIMER",
    "EDUCATION_SYSTEM_PROMPT",
    "ROUTES",
    "Agent",
    "AgentRequest",
    "AgentResult",
    "ChatTurn",
    "Classification",
    "Classifier",
    "OpenAIClassifier",
    "Route",
    "Source",
    "StubAgent",
    "TutorReply",
    "UNAVAILABLE_TEXT",
    "apply_guardrail",
    "ask",
    "build_graph",
    "build_system_prompt",
    "get_agent",
    "register_agent",
    "reset_agents",
]
