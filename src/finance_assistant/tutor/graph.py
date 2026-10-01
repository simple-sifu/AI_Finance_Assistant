"""The LangGraph tutor graph (router -> one agent -> guardrail) and the public ``ask``.

Each route has its own node. Agent nodes look their agent up in the registry at
run time, so registering a real agent needs no graph edits. The graph and the
classifier (with its LLM client) are built per ``ask`` call, so nothing is tied
to a previous ``asyncio.run`` event loop.

The guardrail node reviews replies written by registered (non-stub) agents with
one LLM call and replaces any reply that gives personal advice. Whether a reply
is reviewed is decided here, from which node produced it (``needs_review``),
never by anything an agent returns: stub answers and the graph's own failure
text skip review, and ``clarify``/``unavailable`` are never reviewed.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from .agents import (
    ADVICE_REVIEW_NOTE,
    AGENT_FAILURE_TEXT,
    StubAgent,
    clarify,
    get_agent,
    unavailable,
)
from .guardrail import ADVICE_REDIRECT, apply_guardrail
from .models import (
    AGENT_ROUTES,
    ROUTES,
    AgentRequest,
    AgentResult,
    ChatTurn,
    Route,
    TutorReply,
)
from .router import (
    AdviceReviewer,
    Classifier,
    OpenAIAdviceReviewer,
    OpenAIClassifier,
    classify_question,
    review_reply,
    seeks_advice_keywords,
)

logger = logging.getLogger(__name__)

# What the user sees when the reviewer finds personal advice in a reply.
ADVICE_REPLACEMENT_TEXT = f"{ADVICE_REDIRECT}\n\n{ADVICE_REVIEW_NOTE}"


class TutorState(TypedDict, total=False):
    question: str
    history: tuple[ChatTurn, ...]
    route: Route
    seeks_advice: bool
    result: AgentResult
    # Set only by agent nodes: True when a registered non-stub agent wrote the reply.
    needs_review: bool
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
            agent = get_agent(route)
            result = await agent.run(_request(state))
            if not isinstance(result, AgentResult):
                raise TypeError("agent did not return an AgentResult")
        except Exception as exc:  # noqa: BLE001 - an agent crash must never reach the user
            logger.warning("Agent %s failed (%s)", route, type(exc).__name__)
            return {"result": AgentResult(text=AGENT_FAILURE_TEXT), "needs_review": False}
        # Exact type: a StubAgent subclass overriding run() is a real agent and is reviewed.
        return {"result": result, "needs_review": type(agent) is not StubAgent}

    run_agent.__name__ = f"run_{route}"
    return run_agent


async def _clarify_node(state: TutorState) -> dict[str, Any]:
    return {"result": clarify(_request(state))}


async def _unavailable_node(state: TutorState) -> dict[str, Any]:
    return {"result": unavailable(_request(state))}


class _DefaultAdviceReviewer:
    """Builds the OpenAI reviewer only when a review actually runs.

    Stub-only conversations therefore make a single completion (the router's),
    and a missing key surfaces inside ``review_reply``, which fails closed.
    """

    async def gives_advice(self, question: str, reply: str) -> bool:
        return await OpenAIAdviceReviewer(get_settings()).gives_advice(question, reply)


def build_graph(classifier: Classifier, reviewer: AdviceReviewer | None = None):
    """Compile the tutor graph around ``classifier`` and the guardrail's ``reviewer``.

    With no ``reviewer``, the OpenAI reviewer is built lazily on the first review.
    """
    active_reviewer: AdviceReviewer = reviewer if reviewer is not None else _DefaultAdviceReviewer()

    async def router_node(state: TutorState) -> dict[str, Any]:
        decision = await classify_question(classifier, state["question"], state.get("history", ()))
        seeks_advice = decision.seeks_advice
        if decision.route != "unavailable":
            seeks_advice = seeks_advice or seeks_advice_keywords(state["question"])
        return {"route": decision.route, "seeks_advice": seeks_advice}

    async def guardrail_node(state: TutorState) -> dict[str, Any]:
        result = state["result"]
        if state.get("needs_review"):
            verdict = await review_reply(active_reviewer, state["question"], result.text)
            if verdict == "advice":
                logger.warning("Reviewer flagged personal advice in the %s reply; replaced it", state["route"])
                result = AgentResult(text=ADVICE_REPLACEMENT_TEXT)
            elif verdict == "failed":
                logger.warning("Could not review the %s reply; withholding it", state["route"])
                result = AgentResult(text=AGENT_FAILURE_TEXT)
        return {"result": result, "text": apply_guardrail(result.text)}

    graph = StateGraph(TutorState)
    graph.add_node("router", router_node)
    for route in AGENT_ROUTES:
        graph.add_node(route, _agent_node(route))
    graph.add_node("clarify", _clarify_node)
    graph.add_node("unavailable", _unavailable_node)
    graph.add_node("guardrail", guardrail_node)

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
    reviewer: AdviceReviewer | None = None,
) -> TutorReply:
    """Answer one question. Every reply carries the education-only disclaimer.

    Raises ``ValueError`` for an empty question (before any LLM call) or
    malformed history, and ``ConfigurationError`` if no classifier is given and
    ``OPENAI_API_KEY`` is unset. Nothing else raises: unparseable classifier
    output routes to ``clarify``, classifier/provider errors route to
    ``unavailable`` ("temporarily unavailable" reply), and an agent crash keeps
    its route but replies "couldn't finish answering" with no sources.

    Replies from registered non-stub agents are reviewed by ``reviewer``
    (default: the OpenAI reviewer, built only when a review runs). A reply that
    gives personal advice is replaced by the advice redirect; a failed review
    fails closed with the "couldn't finish answering" reply. Either way the
    route is unchanged and sources are dropped.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    turns = tuple(history or ())
    if not all(isinstance(t, ChatTurn) for t in turns):
        raise ValueError("history must be a list of ChatTurn")
    if classifier is None:
        classifier = OpenAIClassifier(get_settings())

    final = await build_graph(classifier, reviewer).ainvoke(
        {"question": question.strip(), "history": turns}
    )
    result: AgentResult = final["result"]
    return TutorReply(
        text=final["text"],
        route=final["route"],
        seeks_advice=final["seeks_advice"],
        sources=list(result.sources),
    )
