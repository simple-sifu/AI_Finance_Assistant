"""The LangGraph tutor graph (router -> one agent -> guardrail) and the public ``ask``.

Each route has its own node. Agent nodes look their agent up in the registry at
run time, so registering a real agent needs no graph edits. The graph and the
classifier (with its LLM client) are built per ``ask`` call, so nothing is tied
to a previous ``asyncio.run`` event loop.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from .agents import AGENT_FAILURE_TEXT, clarify, get_agent, unavailable
from .guardrail import apply_guardrail
from .models import (
    AGENT_ROUTES,
    ROUTES,
    AgentRequest,
    AgentResult,
    ChatTurn,
    Route,
    TutorReply,
)
from .router import Classifier, OpenAIClassifier, classify_question

logger = logging.getLogger(__name__)


class TutorState(TypedDict, total=False):
    question: str
    history: tuple[ChatTurn, ...]
    route: Route
    seeks_advice: bool
    result: AgentResult
    text: str


def _request(state: TutorState) -> AgentRequest:
    return AgentRequest(
        question=state["question"],
        history=state.get("history", ()),
        seeks_advice=state.get("seeks_advice", False),
    )


def _agent_node(route: Route):
    async def run_agent(state: TutorState) -> dict[str, Any]:
        try:
            result = await get_agent(route).run(_request(state))
            if not isinstance(result, AgentResult):
                raise TypeError("agent did not return an AgentResult")
        except Exception as exc:  # noqa: BLE001 - an agent crash must never reach the user
            logger.warning("Agent %s failed (%s)", route, type(exc).__name__)
            result = AgentResult(text=AGENT_FAILURE_TEXT)
        return {"result": result}

    run_agent.__name__ = f"run_{route}"
    return run_agent


async def _clarify_node(state: TutorState) -> dict[str, Any]:
    return {"result": clarify(_request(state))}


async def _unavailable_node(state: TutorState) -> dict[str, Any]:
    return {"result": unavailable(_request(state))}


async def _guardrail_node(state: TutorState) -> dict[str, Any]:
    return {"text": apply_guardrail(state["result"].text)}


def build_graph(classifier: Classifier):
    """Compile the tutor graph around ``classifier``."""

    async def router_node(state: TutorState) -> dict[str, Any]:
        decision = await classify_question(classifier, state["question"], state.get("history", ()))
        return {"route": decision.route, "seeks_advice": decision.seeks_advice}

    graph = StateGraph(TutorState)
    graph.add_node("router", router_node)
    for route in AGENT_ROUTES:
        graph.add_node(route, _agent_node(route))
    graph.add_node("clarify", _clarify_node)
    graph.add_node("unavailable", _unavailable_node)
    graph.add_node("guardrail", _guardrail_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s["route"], {r: r for r in ROUTES})
    for route in ROUTES:
        graph.add_edge(route, "guardrail")
    graph.add_edge("guardrail", END)
    return graph.compile()


async def ask(
    question: str,
    history: Sequence[ChatTurn] | None = None,
    *,
    classifier: Classifier | None = None,
) -> TutorReply:
    """Answer one question. Every reply carries the education-only disclaimer.

    Raises ``ValueError`` for an empty question (before any LLM call) or
    malformed history, and ``ConfigurationError`` if no classifier is given and
    ``OPENAI_API_KEY`` is unset. Nothing else raises: unparseable classifier
    output routes to ``clarify``, classifier/provider errors route to
    ``unavailable`` ("temporarily unavailable" reply), and an agent crash keeps
    its route but replies "couldn't finish answering" with no sources.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    turns = tuple(history or ())
    if not all(isinstance(t, ChatTurn) for t in turns):
        raise ValueError("history must be a list of ChatTurn")
    if classifier is None:
        classifier = OpenAIClassifier(get_settings())

    final = await build_graph(classifier).ainvoke(
        {"question": question.strip(), "history": turns}
    )
    result: AgentResult = final["result"]
    return TutorReply(
        text=final["text"],
        route=final["route"],
        seeks_advice=final["seeks_advice"],
        sources=list(result.sources),
    )
