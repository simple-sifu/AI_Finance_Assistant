"""Data types shared by the router, the agents, and the public ``ask`` entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

# Routes the classifier may choose: the six agents plus ``clarify``.
ClassifierRoute = Literal[
    "finance_qa",
    "portfolio",
    "market",
    "goal_planning",
    "news",
    "tax_education",
    "clarify",
]
# ``unavailable`` is set by the tutor itself when the classifier/provider fails.
Route = Literal[ClassifierRoute, "unavailable"]
ROUTES: tuple[Route, ...] = (*get_args(ClassifierRoute), "unavailable")
CLASSIFIER_ROUTES: tuple[ClassifierRoute, ...] = get_args(ClassifierRoute)
# The six specialist agents; ``clarify`` and ``unavailable`` are answered by the tutor itself.
AGENT_ROUTES: tuple[Route, ...] = tuple(r for r in ROUTES if r not in ("clarify", "unavailable"))

Role = Literal["user", "assistant"]
ROLES: tuple[Role, ...] = get_args(Role)


@dataclass(frozen=True)
class ChatTurn:
    """One earlier message in the conversation."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"ChatTurn.role must be one of {list(ROLES)}, got {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError("ChatTurn.content must be a string")


@dataclass(frozen=True)
class Source:
    """Something an answer draws on, e.g. a knowledge-base article or news URL."""

    title: str
    url: str | None = None


@dataclass(frozen=True)
class AgentRequest:
    """What an agent receives: the question, prior turns, and the router's advice flag."""

    question: str
    history: tuple[ChatTurn, ...] = ()
    seeks_advice: bool = False


@dataclass(frozen=True)
class AgentResult:
    """What an agent returns, before the guardrail adds the disclaimer."""

    text: str
    sources: list[Source] = field(default_factory=list)


@dataclass(frozen=True)
class Classification:
    """The router's decision for one question."""

    route: Route
    seeks_advice: bool = False


@dataclass(frozen=True)
class TutorReply:
    """The final answer shown to the user."""

    text: str
    route: Route
    seeks_advice: bool
    sources: list[Source] = field(default_factory=list)
