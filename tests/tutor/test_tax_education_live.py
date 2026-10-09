"""Live eval of the Tax Education agent (opt in: ``uv run pytest -m live``, needs OPENAI_API_KEY).

Uses the real FAISS index (built on first use if missing), the real
all-MiniLM-L6-v2 model, the real OpenAI model, and the real advice reviewer.
Deselected by default; settings load inside a fixture, never at import.
"""

from __future__ import annotations

import pytest

from finance_assistant import config
from finance_assistant.knowledge import load_articles, reset_index
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    DISCLAIMER,
    AgentRequest,
    OpenAIAdviceReviewer,
    TaxEducationAgent,
    ask,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.router import review_reply
from finance_assistant.tutor.tax_education import TAX_NOT_COVERED_TEXT, TAX_SITUATION_REDIRECT

pytestmark = pytest.mark.live

IRA_ARTICLES = {
    "individual-retirement-accounts-iras",
    "individual-retirement-arrangements-iras",
    "traditional-iras",
    "roth-iras",
    "traditional-and-roth-iras",
    "roth-comparison-chart",
    "ira-deduction-limits",
    "retirement-topics-ira-contribution-limits",
}
K401_ARTICLES = {
    "401-k-plans",
    "401-k-plans-irs-overview",
    "401-k-resource-guide-plan-participants-401-k-plan-overview",
    "retirement-topics-401-k-and-profit-sharing-plan-contribution-limits",
    "retirement-topics-designated-roth-account",
}

# (question, seeks_advice, situation_specific, slugs of articles of which at least one must be cited)
CASES: list[tuple[str, bool, bool, set[str]]] = [
    ("What's the difference between a Traditional IRA and a Roth IRA?", False, False, IRA_ARTICLES),
    ("How is a 401(k) different from an IRA?", False, False, IRA_ARTICLES | K401_ARTICLES),
    ("What are the IRA contribution limits?", False, False, {"retirement-topics-ira-contribution-limits", *IRA_ARTICLES}),
    (
        "I earn $150k and have a 401(k) at work. Can I deduct my IRA contribution?",
        False,
        True,
        {"ira-deduction-limits", *IRA_ARTICLES},
    ),
    ("Should I open a Roth IRA?", True, False, IRA_ARTICLES),
]
URL_BY_SLUG = {a.slug: a.url for a in load_articles()}


@pytest.fixture(autouse=True)
def _no_network():
    """Override the global guard: this eval must reach the real API."""
    yield


@pytest.fixture(autouse=True)
def _isolated_knowledge():
    """Override the global isolation: use the real index directory and embedding model."""
    reset_index()
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


@pytest.mark.parametrize(("question", "seeks_advice", "situational", "expected"), CASES, ids=[c[0] for c in CASES])
async def test_cap7_answer_is_cited_and_passes_review(
    live_settings: config.Settings, question: str, seeks_advice: bool, situational: bool, expected: set[str]
) -> None:
    agent = TaxEducationAgent(settings=live_settings)
    result = await agent.run(AgentRequest(question, seeks_advice=seeks_advice))
    print(f"\n--- {question}\n{result.text}\n" + "\n".join(f"[{i}] {s.title} {s.url}" for i, s in enumerate(result.sources, 1)))

    assert result.sources, "answer cited no article"
    urls = {s.url for s in result.sources}
    expected_urls = {URL_BY_SLUG[slug] for slug in expected}
    assert urls & expected_urls, f"expected one of {sorted(expected)}, got {urls}"
    for i in range(1, len(result.sources) + 1):
        assert f"[{i}]" in result.text
    if seeks_advice:
        assert result.text.startswith(ADVICE_REDIRECT)
        assert TAX_SITUATION_REDIRECT not in result.text
    elif situational:
        assert result.text.startswith(TAX_SITUATION_REDIRECT)
        assert "professional" in result.text.lower()
    else:
        assert not result.text.startswith((TAX_SITUATION_REDIRECT, ADVICE_REDIRECT))
    verdict = await review_reply(OpenAIAdviceReviewer(live_settings), question, result.text)
    assert verdict == "ok"


async def test_ask_roth_vs_traditional_end_to_end(live_settings: config.Settings) -> None:
    install_real_agents()
    reply = await ask("How is a Roth IRA different from a traditional IRA?")
    print(f"\n{reply.text}\n{reply.sources}")
    assert reply.route == "tax_education"
    assert reply.text.count(DISCLAIMER) == 1
    ira_urls = {URL_BY_SLUG[slug] for slug in IRA_ARTICLES}
    assert {s.url for s in reply.sources} & ira_urls


async def test_uncovered_tax_question_is_not_covered(live_settings: config.Settings) -> None:
    """Checks MIN_SCORE (and the model's NOT_COVERED reply) for a tax question the articles don't cover."""
    result = await TaxEducationAgent(settings=live_settings).run(AgentRequest("How do I file my state taxes?"))
    assert result.sources == []
    assert result.text == TAX_NOT_COVERED_TEXT
