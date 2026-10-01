"""Live eval of the real router prompt (opt in: ``uv run pytest -m live``, needs OPENAI_API_KEY).

Deselected by default (``addopts = -m "not live"``). Settings load inside a
fixture, never at import, so a malformed .env cannot break default collection.
"""

from __future__ import annotations

import asyncio

import pytest

from finance_assistant import config
from finance_assistant.tutor import OpenAIClassifier, reset_agents
from finance_assistant.tutor.router import classify_question, seeks_advice_keywords

pytestmark = pytest.mark.live

MIN_ROUTE_ACCURACY = 0.90
MAX_ADVICE_FALSE_POSITIVES = 1

# Advice questions with more than one defensible route: only the advice flag must be right.
_ANY_TOPIC = ("market", "portfolio", "finance_qa")

# (question, expected route or acceptable routes, expected seeks_advice)
CASES: list[tuple[str, str | tuple[str, ...], bool]] = [
    ("What is compound interest?", "finance_qa", False),
    ("What's the difference between an ETF and a mutual fund?", "finance_qa", False),
    ("Why do people say diversification lowers risk?", "finance_qa", False),
    ("Should I buy AAPL at today's price?", "market", True),
    ("How diversified is my portfolio of 60% AAPL and 40% MSFT?", "portfolio", False),
    ("Should I sell some of my Apple shares to rebalance my portfolio?", "portfolio", True),
    ("What is the current price of AAPL?", "market", False),
    ("What does a P/E ratio of 30 mean for NVDA?", "market", False),
    ("Which index fund should I invest in?", "finance_qa", True),
    ("How much do I need to save each month to have $20,000 in 3 years?", "goal_planning", False),
    ("If I save $300 a month at 6%, what will I have in 10 years?", "goal_planning", False),
    ("What happened in the stock market today?", "news", False),
    ("Summarize this week's financial headlines.", "news", False),
    ("How is a 401(k) different from an IRA?", "tax_education", False),
    ("What are the Roth IRA contribution limits?", "tax_education", False),
    ("How much should I put in my Roth IRA this year?", "tax_education", True),
    ("Would you buy Tesla right now?", "market", True),
    ("What stock should I buy?", _ANY_TOPIC, True),
    ("Should I put $5,000 in VOO?", _ANY_TOPIC, True),
    ("hi", "clarify", False),
    ("What's the weather in Chicago?", "clarify", False),
    ("Can you help me write a Python script?", "clarify", False),
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


async def test_router_prompt_eval(live_settings: config.Settings, capsys: pytest.CaptureFixture[str]) -> None:
    classifier = OpenAIClassifier(live_settings)
    limit = asyncio.Semaphore(5)

    async def run(question: str):  # type: ignore[no-untyped-def]
        async with limit:
            return await classify_question(classifier, question, ())

    results = await asyncio.gather(*(run(q) for q, _, _ in CASES))

    lines = [f"Router eval ({live_settings.openai_model}):"]
    route_hits = advice_total = advice_hits = 0
    for (question, route, advice), got in zip(CASES, results, strict=True):
        allowed = route if isinstance(route, tuple) else (route,)
        route_ok = got.route in allowed
        route_hits += route_ok
        if advice:
            advice_total += 1
            advice_hits += got.seeks_advice
        advice_ok = got.seeks_advice == advice
        status = "ok  " if route_ok and advice_ok else "MISS"
        lines.append(
            f"  {status} route={got.route:<13} (want {"|".join(allowed):<13}) "
            f"advice={got.seeks_advice!s:<5} (want {advice!s:<5}) "
            f"backstop={seeks_advice_keywords(question)!s:<5} {question}"
        )
    accuracy = route_hits / len(CASES)
    lines.append(
        f"  route accuracy {route_hits}/{len(CASES)} ({accuracy:.0%}); "
        f"advice recall (prompt only) {advice_hits}/{advice_total}"
    )
    with capsys.disabled():
        print("\n" + "\n".join(lines))

    unavailable = [q for (q, _, _), got in zip(CASES, results, strict=True) if got.route == "unavailable"]
    assert not unavailable, f"provider errors (not prompt misses) for: {unavailable}"
    assert accuracy >= MIN_ROUTE_ACCURACY
    assert advice_hits == advice_total
    false_flags = [q for (q, _, advice), got in zip(CASES, results, strict=True) if not advice and got.seeks_advice]
    assert len(false_flags) <= MAX_ADVICE_FALSE_POSITIVES, f"non-advice questions flagged as advice: {false_flags}"
