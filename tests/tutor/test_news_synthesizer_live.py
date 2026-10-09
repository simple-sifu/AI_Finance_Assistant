"""Live eval of the News Synthesizer (opt in: ``uv run pytest -m live``; needs OPENAI_API_KEY and TAVILY_API_KEY).

Uses real Tavily news search (1 credit per question), the real OpenAI model,
and the real advice reviewer. Deselected by default; settings load inside a
fixture, never at import.
"""

from __future__ import annotations

import re

import pytest

from finance_assistant import config
from finance_assistant.news import NewsClient, reset_news_client
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    DISCLAIMER,
    AgentRequest,
    AgentResult,
    NewsSynthesizerAgent,
    OpenAIAdviceReviewer,
    ask,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.news_synthesizer import NEWS_UNAVAILABLE_TEXT, NO_NEWS_TEXT
from finance_assistant.tutor.router import review_reply

pytestmark = pytest.mark.live

# (question, seeks_advice)
CASES: list[tuple[str, bool]] = [
    ("What's happening with interest rates this week?", False),
    ("Any news on Nvidia?", False),
    ("What's the latest on inflation?", False),
    ("Summarize this week's stock market news.", False),
    ("Should I buy Tesla after today's news?", True),
]

_CITATION_RE = re.compile(r"\[(\d+)\]")


@pytest.fixture(autouse=True)
def _no_network():
    """Override the global guard: this eval must reach the real APIs."""
    yield


@pytest.fixture(autouse=True)
def _isolated_config():
    """Override the global isolation so the real env/.env keys are used."""
    config.reset_settings()
    reset_news_client()
    reset_agents()
    yield
    config.reset_settings()
    reset_news_client()
    reset_agents()


@pytest.fixture
def live_settings() -> config.Settings:
    settings = config.load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    if not settings.tavily_api_key:
        pytest.skip("TAVILY_API_KEY is not set")
    return settings


def assert_cited(text: str, sources: list) -> None:  # type: ignore[type-arg]
    assert text not in (NEWS_UNAVAILABLE_TEXT, NO_NEWS_TEXT), f"no summary: {text!r}"
    assert sources, "summary cited no article"
    cited = {int(n) for n in _CITATION_RE.findall(text)}
    assert cited == set(range(1, len(sources) + 1)), f"citations {sorted(cited)} vs {len(sources)} sources"
    for source in sources:
        assert source.url and source.url.startswith("https://"), source


def show(question: str, result: AgentResult) -> None:
    print(f"\n--- {question}\n{result.text}\n" + "\n".join(f"[{i}] {s.title} {s.url}" for i, s in enumerate(result.sources, 1)))


@pytest.mark.parametrize(("question", "seeks_advice"), CASES, ids=[c[0] for c in CASES])
async def test_cap6_summary_cites_urls_and_passes_review(
    live_settings: config.Settings, question: str, seeks_advice: bool
) -> None:
    agent = NewsSynthesizerAgent(client_provider=lambda: NewsClient(live_settings), settings=live_settings)
    result = await agent.run(AgentRequest(question, seeks_advice=seeks_advice))
    show(question, result)

    assert_cited(result.text, result.sources)
    if seeks_advice:
        assert result.text.startswith(ADVICE_REDIRECT)
        assert result.text.count(ADVICE_REDIRECT) == 1
    verdict = await review_reply(OpenAIAdviceReviewer(live_settings), question, result.text)
    assert verdict == "ok"


async def test_ask_federal_reserve_news_end_to_end(live_settings: config.Settings) -> None:
    install_real_agents()
    reply = await ask("What's the latest news on the Federal Reserve?")
    print(f"\n{reply.text}\n" + "\n".join(f"[{i}] {s.title} {s.url}" for i, s in enumerate(reply.sources, 1)))

    assert reply.route == "news"
    assert reply.text.count(DISCLAIMER) == 1
    assert_cited(reply.text.removesuffix(f"\n\n{DISCLAIMER}"), reply.sources)
