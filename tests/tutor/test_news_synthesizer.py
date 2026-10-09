"""News Synthesizer agent: every I/O-matrix row offline (respx for Tavily and OpenAI)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

import httpx
import pytest
import respx

from finance_assistant import config
from finance_assistant.config import Settings
from finance_assistant.news import TAVILY_SEARCH_URL, NewsClient
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    DISCLAIMER,
    EDUCATION_SYSTEM_PROMPT,
    AgentRequest,
    ChatTurn,
    NewsSynthesizerAgent,
    Source,
    StubAgent,
    ask,
    get_agent,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.news_synthesizer import (
    NEWS_SYNTHESIZER_INSTRUCTIONS,
    NEWS_UNAVAILABLE_TEXT,
    NO_NEWS_SENTINEL,
    NO_NEWS_TEXT,
    search_query,
)

from ..market_data.conftest import FakeClock
from ..test_news import TAVILY_KEY, result, tavily_body
from .test_ask import AllowAllReviewer, FakeClassifier
from .test_router import COMPLETIONS_URL
from .test_router import _completion as completion

SETTINGS = Settings(openai_api_key="sk-test", openai_model="gpt-4o-mini", tavily_api_key=TAVILY_KEY)

FED = result(
    title="Fed holds rates steady",
    url="https://www.reuters.com/markets/fed-holds/",
    content="The Federal Reserve held its benchmark rate steady on Wednesday.",
    published_date="Wed, 08 Oct 2026 18:30:00 GMT",
)
YIELDS = result(
    title="Treasury yields slip",
    url="https://apnews.com/article/yields-1",
    content="Ten-year Treasury yields slipped after the Fed decision.",
    published_date="2026-10-08T20:00:00Z",
)
MORTGAGE = result(
    title="Mortgage rates ease",
    url="https://www.cnbc.com/2026/10/07/mortgage-rates.html",
    content="Average 30-year mortgage rates eased for a third week.",
    published_date=None,
)
NVIDIA = result(
    title="Nvidia unveils new chip",
    url="https://www.bloomberg.com/news/nvidia-chip",
    content="Nvidia announced a new data-center chip on Tuesday.",
    published_date="Tue, 07 Oct 2026 15:00:00 GMT",
)

FED_SOURCE = Source("Fed holds rates steady (reuters.com, 2026-10-08)", "https://www.reuters.com/markets/fed-holds/")
YIELDS_SOURCE = Source("Treasury yields slip (apnews.com, 2026-10-08)", "https://apnews.com/article/yields-1")


def numbered(prompt: str) -> dict[str, int]:
    """Headline -> number, as listed in the agent's prompt."""
    return {title: int(n) for n, title in re.findall(r'^\[(\d+)\] "([^"]+)"', prompt, re.M)}


class FakeLLM:
    """respx side effect: records requests and answers with ``answer(headline -> number)``."""

    def __init__(self, answer: Callable[[dict[str, int]], str]) -> None:
        self.answer = answer
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        prompt = body["messages"][-1]["content"]
        return httpx.Response(200, json=completion(self.answer(numbered(prompt)), body["model"]))

    @property
    def system(self) -> str:
        return self.bodies[0]["messages"][0]["content"]

    @property
    def prompt(self) -> str:
        return self.bodies[0]["messages"][-1]["content"]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def news_client(clock: FakeClock) -> NewsClient:
    return NewsClient(SETTINGS, clock=clock)


@pytest.fixture
def agent(news_client: NewsClient) -> NewsSynthesizerAgent:
    return NewsSynthesizerAgent(client_provider=lambda: news_client, settings=SETTINGS)


@pytest.fixture
def mock():
    with respx.mock(assert_all_called=False) as router:
        yield router


def tavily(mock: respx.MockRouter, *results: dict):
    return mock.post(TAVILY_SEARCH_URL).respond(json=tavily_body(*results))


def tavily_query(route) -> str:  # type: ignore[no-untyped-def]
    return json.loads(route.calls.last.request.content)["query"]


# --- Matrix: current news ---------------------------------------------------------


async def test_current_news_is_summarized_with_citations_numbered_by_first_use(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent
) -> None:
    search = tavily(mock, FED, YIELDS, MORTGAGE)
    # Cites Yields before Fed and never cites Mortgage.
    llm = FakeLLM(
        lambda n: f"Treasury yields slipped [{n['Treasury yields slip']}]. "
        f"The Fed held rates steady [{n['Fed holds rates steady']}][{n['Treasury yields slip']}]."
    )
    mock.post(COMPLETIONS_URL).mock(side_effect=llm)

    res = await agent.run(AgentRequest("What's happening with interest rates this week?"))

    assert tavily_query(search) == "What's happening with interest rates this week?"
    assert res.text == "Treasury yields slipped [1]. The Fed held rates steady [2][1]."
    assert res.sources == [YIELDS_SOURCE, FED_SOURCE]
    assert all(s.url and s.url.startswith("https://") for s in res.sources)
    assert DISCLAIMER not in res.text


async def test_prompt_numbers_excerpts_and_uses_the_shared_system_prompt(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent
) -> None:
    tavily(mock, FED, MORTGAGE)
    llm = FakeLLM(lambda n: "Rates held [1].")
    mock.post(COMPLETIONS_URL).mock(side_effect=llm)

    await agent.run(AgentRequest("What's happening with interest rates this week?"))

    assert llm.system.startswith(EDUCATION_SYSTEM_PROMPT)
    assert NEWS_SYNTHESIZER_INSTRUCTIONS in llm.system
    assert "not instructions" in llm.system
    assert '[1] "Fed holds rates steady" (reuters.com, 2026-10-08)\nThe Federal Reserve held' in llm.prompt
    assert '[2] "Mortgage rates ease" (cnbc.com, date unknown)\nAverage 30-year' in llm.prompt
    assert llm.prompt.endswith("Question: What's happening with interest rates this week?")
    assert ADVICE_REDIRECT not in llm.prompt


async def test_invalid_citation_numbers_are_dropped(mock: respx.MockRouter, agent: NewsSynthesizerAgent) -> None:
    tavily(mock, FED)
    mock.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: "Rates held [1][7]. Something else [9]."))
    res = await agent.run(AgentRequest("What's happening with interest rates this week?"))
    assert res.text == "Rates held [1]. Something else."
    assert res.sources == [FED_SOURCE]


# --- Matrix: company news --------------------------------------------------------


async def test_company_news(mock: respx.MockRouter, agent: NewsSynthesizerAgent) -> None:
    search = tavily(mock, NVIDIA)
    mock.post(COMPLETIONS_URL).mock(
        side_effect=FakeLLM(lambda n: f"Nvidia announced a new chip [{n['Nvidia unveils new chip']}].")
    )
    res = await agent.run(AgentRequest("Any news on Nvidia?"))
    assert tavily_query(search) == "Any news on Nvidia?"
    assert res.text == "Nvidia announced a new chip [1]."
    assert res.sources == [
        Source("Nvidia unveils new chip (bloomberg.com, 2026-10-07)", "https://www.bloomberg.com/news/nvidia-chip")
    ]


# --- Matrix: advice-seeking --------------------------------------------------------


async def test_advice_question_opens_with_redirect(mock: respx.MockRouter, agent: NewsSynthesizerAgent) -> None:
    tavily(mock, NVIDIA)
    llm = FakeLLM(lambda n: "Nvidia announced a new chip [1].")  # model forgot the redirect
    mock.post(COMPLETIONS_URL).mock(side_effect=llm)

    res = await agent.run(AgentRequest("Should I buy Tesla after today's news?", seeks_advice=True))

    assert res.text == f"{ADVICE_REDIRECT}\n\nNvidia announced a new chip [1]."
    assert "Begin your reply with exactly this sentence" in llm.prompt
    assert "no buy, sell, or hold view" in llm.prompt


async def test_advice_redirect_is_not_duplicated(mock: respx.MockRouter, agent: NewsSynthesizerAgent) -> None:
    tavily(mock, NVIDIA)
    mock.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: f"{ADVICE_REDIRECT} Nvidia showed a chip [1]."))
    res = await agent.run(AgentRequest("Should I buy Nvidia after today's news?", seeks_advice=True))
    assert res.text.startswith(ADVICE_REDIRECT)
    assert res.text.count(ADVICE_REDIRECT) == 1


async def test_quoted_curly_apostrophe_redirect_is_not_duplicated(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent
) -> None:
    tavily(mock, NVIDIA)
    curly = "\u201c" + ADVICE_REDIRECT.replace("'", "\u2019") + "\u201d"
    mock.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: f"{curly}\n\nNvidia showed a chip [1]."))
    res = await agent.run(AgentRequest("Should I buy Nvidia after today's news?", seeks_advice=True))
    assert res.text == f"{ADVICE_REDIRECT}\n\nNvidia showed a chip [1]."


# --- Matrix: follow-up ------------------------------------------------------------


HISTORY = (
    ChatTurn("user", "What's happening with interest rates this week?"),
    ChatTurn("assistant", "The Fed held rates steady [1]."),
)


async def test_follow_up_searches_question_plus_previous_user_turn(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent
) -> None:
    search = tavily(mock, FED)
    llm = FakeLLM(lambda n: "The Fed held its rate [1].")
    mock.post(COMPLETIONS_URL).mock(side_effect=llm)

    await agent.run(AgentRequest("Tell me more about the first one", HISTORY))

    assert tavily_query(search) == (
        "Tell me more about the first one What's happening with interest rates this week?"
    )
    assert "not these excerpts):\nUser: What's happening" in llm.prompt


def test_search_query_rules() -> None:
    # Short question with no subject of its own: a follow-up.
    assert search_query(AgentRequest("What did they say about it?", HISTORY)) == (
        "What did they say about it? What's happening with interest rates this week?"
    )
    # Short question that names its own subject: searched alone.
    assert search_query(AgentRequest("And Nvidia?", HISTORY)) == "And Nvidia?"
    assert search_query(AgentRequest("Any news on Tesla?", HISTORY)) == "Any news on Tesla?"
    assert search_query(AgentRequest("what about NVDA?", HISTORY)) == "what about NVDA?"
    # "I" is not a subject.
    assert search_query(AgentRequest("Should I worry?", HISTORY)) == (
        "Should I worry? What's happening with interest rates this week?"
    )
    # Explicit referring-back phrase in a longer question: a follow-up.
    assert search_query(AgentRequest("Tell me more about the first one", HISTORY)).startswith(
        "Tell me more about the first one What's happening"
    )
    # Longer standalone question with pronouns: searched alone.
    apple = "What did Apple say about its earnings this quarter?"
    assert search_query(AgentRequest(apple, HISTORY)) == apple
    # Long, self-contained question: searched alone.
    long_q = "What did the European Central Bank decide about rates on Thursday?"
    assert search_query(AgentRequest(long_q, HISTORY)) == long_q
    # No history, or the previous turn is the same question.
    assert search_query(AgentRequest("Any news on Nvidia?")) == "Any news on Nvidia?"
    same = (ChatTurn("user", "any news on nvidia?"),)
    assert search_query(AgentRequest("Any news on Nvidia?", same)) == "Any news on Nvidia?"
    # Clipped to 400 characters, question first.
    huge = (ChatTurn("user", "x" * 1000),)
    query = search_query(AgentRequest("Tell me more about it", huge))
    assert len(query) == 400 and query.startswith("Tell me more about it x")


async def test_prompt_has_todays_date(mock: respx.MockRouter, agent: NewsSynthesizerAgent) -> None:
    from datetime import UTC, datetime

    tavily(mock, FED)
    llm = FakeLLM(lambda n: "Rates held [1].")
    mock.post(COMPLETIONS_URL).mock(side_effect=llm)
    await agent.run(AgentRequest("What's happening with interest rates this week?"))
    assert llm.prompt.startswith(f"Today's date (UTC): {datetime.now(UTC).date().isoformat()}\n\n")


# --- Matrix: no results / uncited answer --------------------------------------------


async def test_no_results_says_no_news_without_calling_the_llm(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent
) -> None:
    tavily(mock)
    llm = mock.post(COMPLETIONS_URL)
    res = await agent.run(AgentRequest("Any news on Zzyzx Widgets Corp?"))
    assert res.text == NO_NEWS_TEXT
    assert res.sources == []
    assert llm.call_count == 0


async def test_no_results_for_advice_question_keeps_redirect(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent
) -> None:
    tavily(mock)
    res = await agent.run(AgentRequest("Should I buy Zzyzx after the news?", seeks_advice=True))
    assert res.text == f"{ADVICE_REDIRECT}\n\n{NO_NEWS_TEXT}"


@pytest.mark.parametrize(
    "answer",
    [
        "The Fed held rates steady.",
        "Rates held [8].",
        "",
        NO_NEWS_SENTINEL,
        f"{NO_NEWS_SENTINEL}.",
        f"{NO_NEWS_SENTINEL}. However, the Fed held rates steady [1].",
    ],
)
async def test_uncited_or_sentinel_answer_is_withheld(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent, answer: str
) -> None:
    tavily(mock, FED)
    mock.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: answer))
    res = await agent.run(AgentRequest("What's happening with interest rates this week?"))
    assert res.text == NO_NEWS_TEXT
    assert res.sources == []


# --- Matrix: repeat question (cache) -------------------------------------------------


async def test_repeat_question_is_served_from_cache(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent, clock: FakeClock
) -> None:
    search = tavily(mock, FED)
    mock.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: "Rates held [1]."))

    first = await agent.run(AgentRequest("What's happening with interest rates this week?"))
    clock.advance(14 * 60)
    second = await agent.run(AgentRequest("what's happening with  interest rates this week?"))

    assert search.call_count == 1
    assert first.sources == second.sources == [FED_SOURCE]


# --- Matrix: search unavailable --------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"detail": {"error": "Unauthorized"}}),
        httpx.Response(429, json={"detail": {"error": "rate limit"}}),
        httpx.Response(432, json={"detail": {"error": "plan limit"}}),
        httpx.Response(433, json={"detail": {"error": "pay-as-you-go limit"}}),
        httpx.Response(500),
        httpx.Response(503),
        httpx.Response(200, content=b"not json"),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("refused"),
    ],
    ids=["401", "429", "432", "433", "500", "503", "bad-json", "timeout", "connect-error"],
)
async def test_search_unavailable_is_honest_and_skips_the_llm(
    mock: respx.MockRouter,
    agent: NewsSynthesizerAgent,
    response: httpx.Response | Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    route = mock.post(TAVILY_SEARCH_URL)
    if isinstance(response, Exception):
        route.mock(side_effect=response)
    else:
        route.mock(return_value=response)
    llm = mock.post(COMPLETIONS_URL)

    with caplog.at_level(logging.DEBUG):
        res = await agent.run(AgentRequest("What's happening with interest rates this week?"))

    assert res.text == NEWS_UNAVAILABLE_TEXT
    assert res.sources == []
    assert llm.call_count == 0
    assert TAVILY_KEY not in caplog.text


async def test_no_key_is_unavailable_without_any_request(mock: respx.MockRouter) -> None:
    search = mock.post(TAVILY_SEARCH_URL)
    llm = mock.post(COMPLETIONS_URL)
    settings = Settings(openai_api_key="sk-test")
    agent = NewsSynthesizerAgent(client_provider=lambda: NewsClient(settings), settings=settings)

    res = await agent.run(AgentRequest("Any news on Nvidia?"))

    assert res.text == NEWS_UNAVAILABLE_TEXT
    assert search.call_count == 0 and llm.call_count == 0


async def test_unavailable_for_advice_question_keeps_redirect(
    mock: respx.MockRouter, agent: NewsSynthesizerAgent
) -> None:
    mock.post(TAVILY_SEARCH_URL).respond(503)
    res = await agent.run(AgentRequest("Should I buy Tesla after today's news?", seeks_advice=True))
    assert res.text == f"{ADVICE_REDIRECT}\n\n{NEWS_UNAVAILABLE_TEXT}"


def test_unavailable_text_offers_a_concept_and_says_try_later() -> None:
    assert "can't look up current news right now" in NEWS_UNAVAILABLE_TEXT
    assert "try again later" in NEWS_UNAVAILABLE_TEXT
    assert "explain a finance concept" in NEWS_UNAVAILABLE_TEXT


# --- Matrix: summary fails ---------------------------------------------------------


async def test_llm_failure_raises_from_the_agent(mock: respx.MockRouter, agent: NewsSynthesizerAgent) -> None:
    import openai

    tavily(mock, FED)
    mock.post(COMPLETIONS_URL).respond(500, json={"error": {"message": "boom"}})
    with pytest.raises(openai.APIError):
        await agent.run(AgentRequest("What's happening with interest rates this week?"))


# --- Through ask(), with install_real_agents() --------------------------------------


@pytest.fixture
def real_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", TAVILY_KEY)
    config.reset_settings()
    install_real_agents()


async def test_install_real_agents_registers_news(real_agents: None) -> None:
    assert isinstance(get_agent("news"), NewsSynthesizerAgent)
    reset_agents()
    assert type(get_agent("news")) is StubAgent


async def test_ask_routes_to_news_and_reviews_the_reply(real_agents: None, mock: respx.MockRouter) -> None:
    tavily(mock, FED, YIELDS)
    mock.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: f"The Fed held rates [{n['Fed holds rates steady']}]."))
    reviewed: list[tuple[str, str]] = []

    class RecordingReviewer(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            reviewed.append((question, reply))
            return False

    question = "What's the latest news on interest rates?"
    reply = await ask(question, classifier=FakeClassifier(), reviewer=RecordingReviewer())

    assert reply.route == "news"
    assert reply.text == f"The Fed held rates [1].\n\n{DISCLAIMER}"
    assert reply.sources == [FED_SOURCE]
    assert reviewed == [(question, "The Fed held rates [1].")]


async def test_ask_advice_reply_flagged_by_review_is_replaced(real_agents: None, mock: respx.MockRouter) -> None:
    tavily(mock, NVIDIA)
    mock.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: "Nvidia looks like a buy [1]."))

    class FlagAll(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            return True

    reply = await ask("Should I buy Nvidia after today's news?", classifier=FakeClassifier(), reviewer=FlagAll())

    assert reply.route == "news"
    assert reply.seeks_advice is True
    assert "looks like a buy" not in reply.text
    assert reply.sources == []
    assert reply.text.count(DISCLAIMER) == 1


async def test_ask_unavailable_search_is_reviewed_and_disclaimed(real_agents: None, mock: respx.MockRouter) -> None:
    mock.post(TAVILY_SEARCH_URL).respond(432)
    reply = await ask("Summarize today's market news", classifier=FakeClassifier(), reviewer=AllowAllReviewer())
    assert reply.route == "news"
    assert reply.text == f"{NEWS_UNAVAILABLE_TEXT}\n\n{DISCLAIMER}"
    assert reply.sources == []


async def test_ask_llm_failure_gives_agent_failure_text(
    real_agents: None, mock: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    tavily(mock, FED)
    mock.post(COMPLETIONS_URL).respond(500, json={"error": {"message": "secret sk-test"}})
    with caplog.at_level(logging.WARNING):
        reply = await ask("Summarize today's market news", classifier=FakeClassifier(), reviewer=AllowAllReviewer())
    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"
    assert reply.sources == []
    assert "Agent news failed" in caplog.text
    assert "sk-test" not in caplog.text and TAVILY_KEY not in caplog.text


def test_instructions_forbid_interpreting_the_news() -> None:
    assert "Do not interpret or judge the news" in NEWS_SYNTHESIZER_INSTRUCTIONS
    assert "a positive sign" in NEWS_SYNTHESIZER_INSTRUCTIONS
