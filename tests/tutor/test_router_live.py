"""Live eval of the real router prompt (opt in: ``uv run pytest -m live``, needs OPENAI_API_KEY).

Routes every case in ``eval_cases.CASES`` (CAP-1, CAP-8). Deselected by default
(``addopts = -m "not live"``). Settings load inside a fixture, never at import,
so a malformed .env cannot break default collection.
"""

from __future__ import annotations

import asyncio

import pytest

from finance_assistant import config
from finance_assistant.tutor import OpenAIClassifier, reset_agents
from finance_assistant.tutor.models import AGENT_ROUTES
from finance_assistant.tutor.router import classify_question, seeks_advice_keywords

from .eval_cases import CASES

pytestmark = pytest.mark.live

MIN_ROUTE_ACCURACY = 0.90
# CAP-1: at least this many plain questions per agent reach that agent.
MIN_CORRECT_PER_AGENT = 5
MAX_ADVICE_FALSE_POSITIVES = 1


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

    async def run(case):  # type: ignore[no-untyped-def]
        async with limit:
            return await classify_question(classifier, case.question, case.history)

    results = await asyncio.gather(*(run(c) for c in CASES))

    lines = [f"Router eval ({live_settings.openai_model}, {len(CASES)} cases):"]
    route_hits = 0
    advice_misses: list[str] = []
    prompt_misses: list[str] = []
    false_flags: list[str] = []
    per_agent = {route: [0, 0] for route in AGENT_ROUTES}  # plain cases: [correct, total]
    for case, got in zip(CASES, results, strict=True):
        route_ok = got.route in case.routes
        route_hits += route_ok
        # The graph ORs the keyword backstop with the classifier's flag.
        flagged = got.seeks_advice or seeks_advice_keywords(case.question)
        if case.seeks_advice and not flagged:
            advice_misses.append(case.question)
        if case.seeks_advice and not got.seeks_advice:
            prompt_misses.append(case.question)
        if not case.seeks_advice and flagged:
            false_flags.append(case.question)
        if not case.seeks_advice and not case.history and len(case.routes) == 1 and case.routes[0] in per_agent:
            per_agent[case.routes[0]][0] += route_ok
            per_agent[case.routes[0]][1] += 1
        status = "ok  " if route_ok and flagged == case.seeks_advice else "MISS"
        follow_up = " (follow-up)" if case.history else ""
        lines.append(
            f"  {status} route={got.route:<13} (want {'|'.join(case.routes):<13}) "
            f"advice={flagged!s:<5} (want {case.seeks_advice!s:<5}) "
            f"prompt={got.seeks_advice!s:<5} {case.question}{follow_up}"
        )
    accuracy = route_hits / len(CASES)
    lines.append("  plain questions (no history) routed correctly per agent:")
    lines.extend(f"    {route:<13} {hits}/{total}" for route, (hits, total) in per_agent.items())
    advice_total = sum(c.seeks_advice for c in CASES)
    lines.append(
        f"  route accuracy {route_hits}/{len(CASES)} ({accuracy:.0%}); "
        f"advice flagged {advice_total - len(advice_misses)}/{advice_total} "
        f"(prompt only {advice_total - len(prompt_misses)}/{advice_total}); "
        f"plain flagged as advice {len(false_flags)}"
    )
    with capsys.disabled():
        print("\n" + "\n".join(lines))

    unavailable = [c.question for c, got in zip(CASES, results, strict=True) if got.route == "unavailable"]
    assert not unavailable, f"provider errors (not prompt misses) for: {unavailable}"
    weak = {route: f"{hits}/{total}" for route, (hits, total) in per_agent.items() if hits < MIN_CORRECT_PER_AGENT}
    assert not weak, f"agents with fewer than {MIN_CORRECT_PER_AGENT} correctly routed questions: {weak}"
    assert accuracy >= MIN_ROUTE_ACCURACY
    assert not advice_misses, f"advice prompts not flagged: {advice_misses}"
    # The router prompt alone must also catch every advice prompt (the pre-story-10 bar).
    assert not prompt_misses, f"advice prompts the router prompt missed: {prompt_misses}"
    assert len(false_flags) <= MAX_ADVICE_FALSE_POSITIVES, f"plain questions flagged as advice: {false_flags}"
