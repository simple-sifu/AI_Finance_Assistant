"""Live end-to-end advice eval (CAP-8; opt in: ``uv run pytest -m live``, needs OPENAI_API_KEY).

Sends one advice prompt per agent (``end_to_end`` cases in ``eval_cases``)
through ``ask()`` with the real router, real agents and real advice reviewer,
and checks that every reply opens with the educational redirect and ends with
the disclaimer. Market questions fall back to mock data and news questions to
the canned "unavailable" reply when their keys are missing, so only
OPENAI_API_KEY is required. Deselected by default; settings load in a fixture.
"""

from __future__ import annotations

import pytest

from finance_assistant import config
from finance_assistant.knowledge import reset_index
from finance_assistant.market_data import reset_client
from finance_assistant.news import reset_news_client
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    DISCLAIMER,
    ask,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.graph import ADVICE_REPLACEMENT_TEXT

from .eval_cases import CASES

pytestmark = pytest.mark.live

END_TO_END = [c for c in CASES if c.end_to_end]


@pytest.fixture(autouse=True)
def _no_network():
    """Override the global guard: this eval must reach the real APIs."""
    yield


@pytest.fixture(autouse=True)
def _isolated_knowledge():
    """Override the global isolation: use the real index directory and embedding model."""
    reset_index()
    yield
    reset_index()


@pytest.fixture(autouse=True)
def _isolated_config():
    """Override the global isolation so the real env/.env keys are used, with the real agents."""
    config.reset_settings()
    reset_client()
    reset_news_client()
    install_real_agents()
    yield
    config.reset_settings()
    reset_client()
    reset_news_client()
    reset_agents()


@pytest.fixture
def live_settings() -> config.Settings:
    settings = config.load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    return settings


@pytest.mark.parametrize("case", END_TO_END, ids=lambda c: c.question)
async def test_advice_prompt_gets_redirect_and_disclaimer(
    live_settings: config.Settings, case, capsys: pytest.CaptureFixture[str]  # type: ignore[no-untyped-def]
) -> None:
    reply = await ask(case.question, case.history)
    replaced = reply.text.startswith(ADVICE_REPLACEMENT_TEXT)
    # Without these keys the market and news agents answer from mock data or a canned reply.
    data = (
        f"alpha_vantage={'live' if live_settings.alpha_vantage_api_key else 'mock'} "
        f"tavily={'live' if live_settings.tavily_api_key else 'unavailable'}"
    )
    with capsys.disabled():
        print(
            f"\n--- {case.question}\nroute={reply.route} seeks_advice={reply.seeks_advice} {data}"
            f"{' (reviewer replaced the reply)' if replaced else ''}\n{reply.text}"
        )
    assert reply.route != "unavailable", f"provider error (not a guardrail miss) for: {case.question}"
    assert not reply.text.startswith(AGENT_FAILURE_TEXT), (
        f"agent or review failure (not a guardrail miss) for: {case.question}"
    )
    assert reply.route in case.routes
    assert reply.seeks_advice
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert reply.text.endswith("\n\n" + DISCLAIMER)
    assert reply.text.count(DISCLAIMER) == 1
