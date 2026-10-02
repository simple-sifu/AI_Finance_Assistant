"""Live eval of the Finance Q&A agent (opt in: ``uv run pytest -m live``, needs OPENAI_API_KEY).

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
    ask,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.finance_qa import NOT_COVERED_TEXT, FinanceQAAgent
from finance_assistant.tutor.router import review_reply

pytestmark = pytest.mark.live

# (question, seeks_advice, slugs of articles of which at least one must be cited)
CASES: list[tuple[str, bool, set[str]]] = [
    ("What is compound interest?", False, {"compound-interest", "what-is-compound-interest"}),
    ("What's the difference between ETFs and mutual funds?", False, {"exchange-traded-funds-etfs", "mutual-funds", "index-funds"}),
    ("What is the rule of 72?", False, {"rule-of-72"}),
    (
        "Why does diversification reduce risk?",
        False,
        {"asset-allocation-and-diversification", "diversify-your-investments", "what-is-risk"},
    ),
    ("Which index fund should I buy?", True, {"index-funds", "mutual-funds", "exchange-traded-funds-etfs"}),
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


@pytest.mark.parametrize(("question", "seeks_advice", "expected"), CASES, ids=[c[0] for c in CASES])
async def test_cap2_answer_is_cited_and_passes_review(
    live_settings: config.Settings, question: str, seeks_advice: bool, expected: set[str]
) -> None:
    agent = FinanceQAAgent(settings=live_settings)
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
    verdict = await review_reply(OpenAIAdviceReviewer(live_settings), question, result.text)
    assert verdict == "ok"


async def test_ask_compound_interest_end_to_end(live_settings: config.Settings) -> None:
    install_real_agents()
    reply = await ask("What is compound interest?")
    assert reply.route == "finance_qa"
    assert reply.text.count(DISCLAIMER) == 1
    assert any("compound" in (s.url or "").lower() for s in reply.sources)


OFF_TOPIC = [
    "How do I refinance a mortgage on my house?",
    "How does cryptocurrency staking work?",
]


@pytest.mark.parametrize("question", OFF_TOPIC)
async def test_off_topic_question_is_not_covered(live_settings: config.Settings, question: str) -> None:
    """Checks MIN_SCORE (and the model's NOT_COVERED reply) against the real embedding model."""
    result = await FinanceQAAgent(settings=live_settings).run(AgentRequest(question))
    assert result.sources == []
    assert result.text == NOT_COVERED_TEXT
