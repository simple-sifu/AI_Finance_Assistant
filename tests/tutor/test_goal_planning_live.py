"""Live eval of the Goal Planning agent (opt in: ``uv run pytest -m live``, needs OPENAI_API_KEY).

Uses the real OpenAI model for goal extraction and the explanation, and the real
advice reviewer; the expected monthly amount is computed with ``monthly_savings``.
One end-to-end case runs ``ask`` with the real router. Deselected by default;
settings load inside a fixture, never at import.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finance_assistant import config
from finance_assistant.goals import end_of_month, monthly_savings, months_until
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    DISCLAIMER,
    AgentRequest,
    GoalPlanningAgent,
    OpenAIAdviceReviewer,
    ask,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.goal_planning import INVITE_RATE_TEXT
from finance_assistant.tutor.router import review_reply

pytestmark = pytest.mark.live

TODAY = date.today()

# (question, seeks_advice, target, months, rate %, current savings)
CASES: list[tuple[str, bool, int, int, int, int]] = [
    ("How much do I need to save monthly to reach $50,000 in 5 years at 5%?", False, 50_000, 60, 5, 0),
    (
        "I want $50,000 in 5 years. I already have $10,000 saved. Assuming 5% a year, what's the monthly amount?",
        False,
        50_000,
        60,
        5,
        10_000,
    ),
    ("What do I need to put aside each month to have $20k by June 2030 at 4%?", False, 20_000, months_until(end_of_month(2030, 6), TODAY), 4, 0),
    ("Monthly savings to reach $12,000 in 2 years with 0% interest?", False, 12_000, 24, 0, 0),
    ("How much should I save each month to reach $50,000 in 5 years at 5%?", True, 50_000, 60, 5, 0),  # advice-flagged
]


@pytest.fixture(autouse=True)
def _no_network():
    """Override the global guard: this eval must reach the real API."""
    yield


@pytest.fixture(autouse=True)
def _isolated_config():
    """Override the global isolation so the real env/.env (and OPENAI_API_KEY) are used."""
    config.reset_settings()
    reset_agents()
    yield
    config.reset_settings()
    reset_agents()


@pytest.fixture
def live_settings() -> config.Settings:
    settings = config.load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    return settings


@pytest.mark.parametrize(
    ("question", "seeks_advice", "target", "months", "rate", "current"), CASES, ids=[c[0] for c in CASES]
)
async def test_cap5_monthly_amount_and_math_pass_review(
    live_settings: config.Settings, question: str, seeks_advice: bool, target: int, months: int, rate: int, current: int
) -> None:
    agent = GoalPlanningAgent(settings=live_settings)
    result = await agent.run(AgentRequest(question, seeks_advice=seeks_advice))
    print(f"\n--- {question}\n{result.text}")

    plan = monthly_savings(target, months, rate, current)
    assert result.text.startswith(f"**Monthly savings needed: ${plan.monthly_amount:,.2f}**")
    if current:
        assert f"- Current savings (PV): ${Decimal(current):,.2f}" in result.text
    assert f"- Time (n): {months} months" in result.text
    assert "Formula" in result.text and "PMT" in result.text
    assert "- Total deposited:" in result.text and "- Interest earned:" in result.text
    explanation = result.text.rsplit("\n\n", 1)[1]
    assert not explanation.startswith("- "), "no explanation after the math"
    assert ADVICE_REDIRECT not in result.text  # computed from the user's own numbers
    verdict = await review_reply(OpenAIAdviceReviewer(live_settings), question, result.text)
    assert verdict == "ok"


async def test_rate_missing_shows_example_rates_and_passes_review(live_settings: config.Settings) -> None:
    question = "How much should I save each month to have $30k in 4 years?"
    agent = GoalPlanningAgent(settings=live_settings)
    result = await agent.run(AgentRequest(question, seeks_advice=True))
    print(f"\n--- {question}\n{result.text}")

    assert result.text.startswith("You didn't give a rate of return, so here is the math at three **example** annual rates")
    assert "not predictions" in result.text and "not suggestions" in result.text
    for rate in (0, 4, 7):
        plan = monthly_savings(30_000, 48, rate)
        assert f"**Example at {rate}%: ${plan.monthly_amount:,.2f} a month**" in result.text
    assert result.text.endswith(INVITE_RATE_TEXT)
    verdict = await review_reply(OpenAIAdviceReviewer(live_settings), question, result.text)
    assert verdict == "ok"


async def test_ask_end_to_end_routes_to_goal_planning(live_settings: config.Settings) -> None:
    install_real_agents()

    reply = await ask("How much do I need to save each month to reach $50,000 in 5 years at 5%?")
    print(f"\n{reply.text}")

    assert reply.route == "goal_planning"
    assert "$735.23" in reply.text
    assert "- PMT = ($50,000.00 − $0.00 × 1.283359) × 0.00416667 / (1.283359 − 1) = **$735.23**" in reply.text
    assert reply.text.count(DISCLAIMER) == 1
