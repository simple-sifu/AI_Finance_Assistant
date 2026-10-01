"""The router: classify the latest question into a route and flag advice-seeking.

The classifier is injectable (tests use a fake). ``classify_question`` wraps any
classifier so it never raises: invalid or unparseable output becomes ``clarify``,
and any other exception (outage, timeout, 429, bad key, network) becomes
``unavailable``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from .llm import chat_model
from .models import CLASSIFIER_ROUTES, ChatTurn, Classification, ClassifierRoute

logger = logging.getLogger(__name__)

# The router sees the latest question plus at most this many earlier turns.
MAX_HISTORY_TURNS = 6
# Long earlier answers are clipped in the router prompt; routing needs only the gist.
_HISTORY_CHARS_PER_TURN = 500


class Classifier(Protocol):
    """Decides the route for a question given recent history."""

    async def classify(self, question: str, history: Sequence[ChatTurn]) -> Classification: ...


ROUTER_SYSTEM_PROMPT = """\
You route questions for a beginner personal-finance tutor. Pick exactly one route \
for the user's LATEST question, using earlier turns only to resolve follow-ups \
(e.g. "what about Roth?" after a question about IRAs is tax_education).

Routes:
- finance_qa: general investing and personal-finance concepts (compound interest, \
ETFs vs mutual funds, stocks, bonds, risk, diversification, fees, index funds).
- portfolio: the user's own holdings — analyzing diversification or concentration \
of a portfolio they have or upload.
- market: a specific ticker or the market right now — quotes, prices, what a \
stock's figures mean.
- goal_planning: saving toward a target amount by a date, monthly savings needed, \
projecting growth with compound interest.
- news: current or recent financial news, headlines, what happened in markets today.
- tax_education: 401(k), IRA, Roth IRA, 403(b), 529, HSA, contribution limits, \
tax-advantaged accounts, how investments are taxed.
- clarify: greetings, off-topic questions (weather, sports, coding), or questions \
too vague to route.

seeks_advice is true when the user asks what THEY should buy, sell, hold, or \
invest in, how much to put somewhere, or for a personal recommendation \
(e.g. "What stock should I buy?", "Should I put $5,000 in VOO?"). Asking how \
something works is not advice-seeking. Route advice-seeking questions to the \
matching topic as usual."""


class _RouterDecision(BaseModel):
    route: ClassifierRoute = Field(description="The single best route for the latest question.")
    seeks_advice: bool = Field(description="True if the user asks for a personal recommendation.")


def _format_history(history: Sequence[ChatTurn]) -> str:
    lines = []
    for turn in history:
        content = turn.content.strip()
        if len(content) > _HISTORY_CHARS_PER_TURN:
            content = content[:_HISTORY_CHARS_PER_TURN] + "…"
        lines.append(f"{turn.role.capitalize()}: {content}")
    return "\n".join(lines)


class OpenAIClassifier:
    """Classifier backed by an OpenAI model with structured output.

    Holds only Settings; the model and its HTTP client are built per call so
    nothing is shared across event loops.
    """

    def __init__(self, settings: Settings) -> None:
        settings.require_openai_api_key()
        self._settings = settings

    async def classify(self, question: str, history: Sequence[ChatTurn]) -> Classification:
        prompt = f"Latest question: {question}"
        if history:
            prompt = f"Conversation so far:\n{_format_history(history)}\n\n{prompt}"
        async with chat_model(self._settings) as model:
            structured = model.with_structured_output(_RouterDecision)
            decision = await structured.ainvoke(
                [SystemMessage(ROUTER_SYSTEM_PROMPT), HumanMessage(prompt)]
            )
        if not isinstance(decision, _RouterDecision):
            raise ValueError(f"unexpected classifier output type: {type(decision).__name__}")
        return Classification(route=decision.route, seeks_advice=decision.seeks_advice)


def recent_history(history: Sequence[ChatTurn]) -> tuple[ChatTurn, ...]:
    """The last ``MAX_HISTORY_TURNS`` turns the router is allowed to see."""
    return tuple(history[-MAX_HISTORY_TURNS:]) if MAX_HISTORY_TURNS else ()


# Structured-output parsing/validation failures: the model answered, but badly.
_BAD_OUTPUT_ERRORS = (OutputParserException, ValidationError, ValueError)
_AUTH_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError)


async def classify_question(
    classifier: Classifier, question: str, history: Sequence[ChatTurn]
) -> Classification:
    """Classify with ``classifier``; never raises.

    Invalid or unparseable output routes to ``clarify``; any other exception
    routes to ``unavailable``. Only exception types are logged, never messages,
    because provider errors can echo request details such as the key.
    """
    try:
        result = await classifier.classify(question, recent_history(history))
    except _BAD_OUTPUT_ERRORS as exc:
        logger.warning("Router classifier output unparseable (%s); routing to clarify", type(exc).__name__)
        return Classification(route="clarify")
    except _AUTH_ERRORS as exc:
        logger.error(
            "OpenAI rejected the request (%s); check that OPENAI_API_KEY is valid. "
            "Routing to unavailable",
            type(exc).__name__,
        )
        return Classification(route="unavailable")
    except Exception as exc:  # noqa: BLE001 - provider/LLM failure means "unavailable"
        logger.warning("Router classifier failed (%s); routing to unavailable", type(exc).__name__)
        return Classification(route="unavailable")
    route = getattr(result, "route", None)
    seeks_advice = getattr(result, "seeks_advice", None)
    if route not in CLASSIFIER_ROUTES or not isinstance(seeks_advice, bool):
        logger.warning("Router classifier returned an invalid result (route=%r); routing to clarify", route)
        return Classification(route="clarify", seeks_advice=seeks_advice is True)
    return Classification(route=route, seeks_advice=seeks_advice)
