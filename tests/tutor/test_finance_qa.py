"""Finance Q&A agent: every I/O-matrix row offline (fake embedder, respx-mocked OpenAI)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

import openai

import httpx
import pytest
import respx

from finance_assistant import config
from finance_assistant.config import Settings
from finance_assistant.knowledge import Chunk, Hit, KnowledgeIndex, get_index, load_articles
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    DISCLAIMER,
    EDUCATION_SYSTEM_PROMPT,
    AgentRequest,
    ChatTurn,
    Source,
    ask,
    get_agent,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor import finance_qa
from finance_assistant.tutor.finance_qa import (
    MAX_ARTICLES,
    MAX_CHUNKS_PER_ARTICLE,
    MIN_SCORE,
    NO_ANSWER_SENTINEL,
    NOT_COVERED_TEXT,
    TOPICS,
    FinanceQAAgent,
    _ArticleContext,
    apply_citations,
    group_hits,
    message_text,
    retrieval_queries,
)

from ..knowledge.fakes import HashEmbedder
from .test_ask import AllowAllReviewer, FakeClassifier
from .test_router import COMPLETIONS_URL
from .test_router import _completion as completion

SETTINGS = Settings(openai_api_key="sk-test", openai_model="gpt-4o-mini")
COMPOUND = Source("Compound interest (Wikipedia)", "https://en.wikipedia.org/wiki/Compound_interest")


@pytest.fixture(scope="module")
def kb() -> KnowledgeIndex:
    from finance_assistant.knowledge import build_index

    return build_index(HashEmbedder())


@pytest.fixture
def agent(kb: KnowledgeIndex) -> FinanceQAAgent:
    return FinanceQAAgent(settings=SETTINGS, index_provider=lambda: kb)


def numbered_articles(prompt: str) -> dict[str, int]:
    """Title -> number, as listed in the agent's prompt."""
    return {title: int(n) for n, title in re.findall(r'^\[(\d+)\] "([^"]+)"', prompt, re.M)}


class FakeLLM:
    """respx side effect: records requests and answers with ``answer(prompt_numbers)``."""

    def __init__(self, answer: Callable[[dict[str, int]], str]) -> None:
        self.answer = answer
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        prompt = body["messages"][-1]["content"]
        return httpx.Response(200, json=completion(self.answer(numbered_articles(prompt)), body["model"]))

    @property
    def system(self) -> str:
        return self.bodies[0]["messages"][0]["content"]

    @property
    def prompt(self) -> str:
        return self.bodies[0]["messages"][-1]["content"]


@respx.mock
async def test_grounded_answer_cites_the_compound_interest_article(agent: FinanceQAAgent) -> None:
    llm = FakeLLM(lambda n: f"Compound interest is interest earned on interest [{n['Compound interest']}].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent.run(AgentRequest("What is compound interest?"))

    assert result.text == "Compound interest is interest earned on interest [1]."
    assert result.sources == [COMPOUND]
    assert llm.system.startswith(EDUCATION_SYSTEM_PROMPT)
    assert "Finance Q&A agent" in llm.system
    assert DISCLAIMER not in result.text
    assert "Question: What is compound interest?" in llm.prompt
    assert "Compound interest" in numbered_articles(llm.prompt)


@respx.mock
async def test_multi_article_answer_returns_only_cited_articles_in_citation_order(agent: FinanceQAAgent) -> None:
    def answer(n: dict[str, int]) -> str:
        etf, mf = n["Exchange-Traded Funds (ETFs)"], n["Mutual Funds"]
        return f"ETFs trade on exchanges all day [{etf}]. Mutual funds price once a day [{mf}]. Both pool money [{etf}, {mf}]."

    llm = FakeLLM(answer)
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent.run(AgentRequest("ETFs vs mutual funds?"))

    assert numbered_articles(llm.prompt)["Mutual Funds"] == 1  # renumbered: ETFs were cited first
    assert result.text == "ETFs trade on exchanges all day [1]. Mutual funds price once a day [2]. Both pool money [1][2]."
    assert [s.title for s in result.sources] == ["Exchange-Traded Funds (ETFs) (Investor.gov)", "Mutual Funds (Investor.gov)"]
    assert all(s.url and s.url.startswith("https://www.investor.gov/") for s in result.sources)


@respx.mock
async def test_uncited_articles_are_not_returned(agent: FinanceQAAgent) -> None:
    llm = FakeLLM(lambda n: f"ETFs trade on exchanges all day [{n['Exchange-Traded Funds (ETFs)']}].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent.run(AgentRequest("ETFs vs mutual funds?"))

    assert len(numbered_articles(llm.prompt)) >= 2  # mutual funds was shown too
    assert result.text == "ETFs trade on exchanges all day [1]."
    assert [s.title for s in result.sources] == ["Exchange-Traded Funds (ETFs) (Investor.gov)"]


@respx.mock
async def test_advice_seeking_opens_with_redirect_then_educates(agent: FinanceQAAgent) -> None:
    llm = FakeLLM(lambda n: f"An index fund tracks a market index [{n['Index Funds']}].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent.run(AgentRequest("Which index fund should I buy?", seeks_advice=True))

    assert result.text.startswith(ADVICE_REDIRECT + "\n\n")
    assert result.text.endswith("An index fund tracks a market index [1].")
    assert [s.title for s in result.sources] == ["Index Funds (Investor.gov)"]
    assert "personal recommendation" in llm.prompt and ADVICE_REDIRECT in llm.prompt


@respx.mock
async def test_redirect_already_present_is_not_duplicated(agent: FinanceQAAgent) -> None:
    respx.post(COMPLETIONS_URL).mock(
        side_effect=FakeLLM(lambda n: f"{ADVICE_REDIRECT} Index funds track an index [{n['Index Funds']}].")
    )
    result = await agent.run(AgentRequest("Which index fund should I buy?", seeks_advice=True))
    assert result.text.count(ADVICE_REDIRECT) == 1


def _chunk(slug: str, position: int = 0, text: str | None = None) -> Chunk:
    return Chunk(
        article_slug=slug,
        title=f"Title {slug}",
        source="Investor.gov (U.S. Securities and Exchange Commission)",
        url=f"https://example.com/{slug}",
        category="concepts",
        heading="",
        text=text if text is not None else f"{slug} chunk {position}",
        position=position,
    )


class StubIndex:
    """Returns fixed hits for every query, so the relevance gate is tested on exact scores."""

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.queries.append(query)
        return self.hits[:k]


def stub_agent(*scores: float) -> FinanceQAAgent:
    index = StubIndex([Hit(_chunk(f"a{i}"), score) for i, score in enumerate(scores)])
    return FinanceQAAgent(settings=SETTINGS, index_provider=lambda: index)  # type: ignore[arg-type,return-value]


@respx.mock
async def test_nothing_relevant_says_not_covered_without_calling_the_llm() -> None:
    route = respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500))

    result = await stub_agent(MIN_SCORE - 0.01, 0.2).run(AgentRequest("How do I refinance a mortgage on my house?"))

    assert not route.called
    assert result.text == NOT_COVERED_TEXT
    assert "don't cover that yet" in result.text and "401(k)" in result.text
    assert result.sources == []


@respx.mock
async def test_hit_at_the_threshold_reaches_the_llm() -> None:
    llm = FakeLLM(lambda n: "Answer [1].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)
    result = await stub_agent(MIN_SCORE, 0.1).run(AgentRequest("A question?"))
    assert result.text == "Answer [1]."
    assert [s.url for s in result.sources] == ["https://example.com/a0"]
    assert list(numbered_articles(llm.prompt)) == ["Title a0"]  # the 0.1 hit was filtered out


@respx.mock
async def test_nothing_relevant_for_advice_question_still_redirects() -> None:
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500))
    result = await stub_agent(MIN_SCORE - 0.01).run(
        AgentRequest("Should I refinance my house mortgage now?", seeks_advice=True)
    )
    assert result.text == f"{ADVICE_REDIRECT}\n\n{NOT_COVERED_TEXT}"
    assert result.sources == []


@pytest.mark.parametrize(
    "reply",
    [
        NO_ANSWER_SENTINEL,
        f" {NO_ANSWER_SENTINEL}.\n",
        "Compound interest is interest on interest.",
        "Interest on interest [9].",
        "",
    ],
)
@respx.mock
async def test_model_declines_or_cites_nothing_valid_gives_not_covered(agent: FinanceQAAgent, reply: str) -> None:
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: reply))
    result = await agent.run(AgentRequest("What is compound interest?"))
    assert result.text == NOT_COVERED_TEXT
    assert result.sources == []


@respx.mock
async def test_llm_failure_raises_from_the_agent(agent: FinanceQAAgent) -> None:
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom sk-test"}}))
    with pytest.raises(openai.InternalServerError):
        await agent.run(AgentRequest("What is compound interest?"))


def test_retrieval_queries_add_question_plus_previous_turn_only_for_short_follow_ups() -> None:
    history = (ChatTurn("user", "What is a Roth IRA?"), ChatTurn("assistant", "A Roth IRA is..."))
    assert retrieval_queries(AgentRequest("and the limits?", history)) == [
        "and the limits?",
        "and the limits?\nWhat is a Roth IRA?",
    ]
    long_q = "How does dollar cost averaging work when markets fall?"
    assert retrieval_queries(AgentRequest(long_q, history)) == [long_q]
    assert retrieval_queries(AgentRequest("and the limits?")) == ["and the limits?"]


@respx.mock
async def test_short_new_topic_question_after_unrelated_history_retrieves_its_own_article(
    agent: FinanceQAAgent,
) -> None:
    llm = FakeLLM(lambda n: f"Divide 72 by the rate [{n['Rule of 72']}].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)
    history = (
        ChatTurn("user", "How do Roth IRA contribution limits and income limits work for a traditional IRA?"),
        ChatTurn("assistant", "Roth IRAs have income limits..."),
    )

    result = await agent.run(AgentRequest("What is the rule of 72?", history))

    assert "Rule of 72" in numbered_articles(llm.prompt)
    assert [s.title for s in result.sources] == ["Rule of 72 (Wikipedia)"]


async def test_merged_hits_keep_each_chunks_best_score() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")

    class TwoQueryIndex:
        def search(self, query: str, k: int = 5) -> list[Hit]:
            if "\n" in query:  # the combined follow-up query
                return [Hit(b, 0.9), Hit(a, 0.3)]
            return [Hit(a, 0.8), Hit(c, 0.5)]

    agent = FinanceQAAgent(settings=SETTINGS, index_provider=lambda: TwoQueryIndex())  # type: ignore[arg-type,return-value]
    hits = agent._retrieve(["q", "q\nprevious"])
    # Interleaved by rank (bare question first); "a" keeps its better score from the bare query.
    assert [(h.chunk.article_slug, h.score) for h in hits] == [("a", 0.8), ("b", 0.9), ("c", 0.5)]


def test_apply_citations_renumbers_by_first_use_and_drops_invalid() -> None:
    contexts = [_ArticleContext(str(i), f"T{i}", "S", f"https://u/{i}", ("x",)) for i in range(1, 4)]
    text, sources = apply_citations("B [3]. A [1, 3]. Bad [7]. Dup [3][3].", contexts)
    assert text == "B [1]. A [2][1]. Bad. Dup [1][1]."
    assert sources == [Source("T3 (S)", "https://u/3"), Source("T1 (S)", "https://u/1")]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("See [2-3].", "See [1][2]."),
        ("See [2\u20133].", "See [1][2]."),
        ("See [3; 2].", "See [1][2]."),
        ("See [3\u20142] and [2].", "See [1][2] and [2]."),  # reversed range: both ends
        ("See [2-9].", "See [1][2]."),  # out-of-range members dropped
    ],
)
def test_apply_citations_expands_ranges_and_separators(raw: str, expected: str) -> None:
    contexts = [_ArticleContext(str(i), f"T{i}", "S", f"https://u/{i}", ("x",)) for i in range(1, 4)]
    text, sources = apply_citations(raw, contexts)
    assert text == expected
    assert not re.search(r"\[\d+\s*[-;,\u2013\u2014]", text)
    assert len(sources) == 2


@respx.mock
async def test_answer_that_mentions_the_sentinel_word_is_kept(agent: FinanceQAAgent) -> None:
    answer = f"Fees are {NO_ANSWER_SENTINEL} by insurance, unlike deposits [1]."
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: answer))
    result = await agent.run(AgentRequest("What is compound interest?"))
    assert result.text == answer
    assert len(result.sources) == 1


def test_message_text_joins_list_content_blocks() -> None:
    blocks = [{"type": "text", "text": "Interest on "}, {"type": "image_url", "image_url": {}}, "interest [1]."]
    assert message_text(blocks) == "Interest on interest [1]."
    assert message_text("  plain  ") == "plain"
    assert message_text(None) == ""


async def test_list_form_llm_content_is_used(kb: KnowledgeIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import asynccontextmanager

    from langchain_core.messages import AIMessage

    class ListContentModel:
        async def ainvoke(self, messages):  # type: ignore[no-untyped-def]
            return AIMessage(content=[{"type": "text", "text": "Interest earns interest "}, {"type": "text", "text": "[1]."}])

    @asynccontextmanager
    async def fake_chat_model(settings, **kwargs):  # type: ignore[no-untyped-def]
        yield ListContentModel()

    monkeypatch.setattr(finance_qa, "chat_model", fake_chat_model)
    result = await FinanceQAAgent(settings=SETTINGS, index_provider=lambda: kb).run(AgentRequest("What is compound interest?"))
    assert result.text == "Interest earns interest [1]."
    assert len(result.sources) == 1


def test_topics_cover_exactly_the_article_categories() -> None:
    assert set(TOPICS) == {a.category for a in load_articles()}
    assert "tax" in TOPICS["retirement-and-tax"]


def test_group_hits_caps_orders_and_filters() -> None:
    hits = [
        Hit(_chunk("best", 5), 0.95),
        Hit(_chunk("second", 0), 0.9),
        Hit(_chunk("best", 1), 0.85),
        Hit(_chunk("best", 3), 0.8),
        Hit(_chunk("best", 0), 0.75),  # 4th chunk of "best": over MAX_CHUNKS_PER_ARTICLE
        Hit(_chunk("third", 0), 0.7),
        Hit(_chunk("low", 0), MIN_SCORE - 0.01),  # below the threshold
        Hit(_chunk("fourth", 0), 0.6),
        Hit(_chunk("fifth", 0), 0.55),  # over MAX_ARTICLES
    ]
    contexts = group_hits(hits, MIN_SCORE)
    assert MAX_ARTICLES == 4 and MAX_CHUNKS_PER_ARTICLE == 3
    assert [c.slug for c in contexts] == ["best", "second", "third", "fourth"]
    assert contexts[0].excerpts == ("best chunk 1", "best chunk 3", "best chunk 5")  # article order
    assert contexts[0].source_name == "Investor.gov"
    assert group_hits([Hit(_chunk("low"), 0.1)], MIN_SCORE) == []


@respx.mock
async def test_prompt_includes_last_four_history_turns_clipped(agent: FinanceQAAgent) -> None:
    llm = FakeLLM(lambda n: f"Interest on interest [{n['Compound interest']}].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)
    long_turn = "L" * 450
    history = (
        ChatTurn("user", "OLDEST-TURN"),
        ChatTurn("assistant", "second"),
        ChatTurn("user", "third"),
        ChatTurn("assistant", long_turn),
        ChatTurn("user", "fifth"),
    )

    await agent.run(AgentRequest("What is compound interest and how does it grow?", history))

    block = llm.prompt.split("Conversation so far (for context only):\n", 1)[1].split("\n\nQuestion:", 1)[0]
    assert block.splitlines() == ["Assistant: second", "User: third", f"Assistant: {'L' * 400}…", "User: fifth"]
    assert "OLDEST-TURN" not in llm.prompt


# --- Through ask(), with install_real_agents() ------------------------------------


@pytest.fixture
def real_agents(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config.reset_settings()
    get_index(embedder=HashEmbedder(), index_dir=tmp_path / "idx")  # process index with the fake embedder
    install_real_agents()


async def test_install_real_agents_registers_finance_qa(real_agents: None) -> None:
    assert isinstance(get_agent("finance_qa"), FinanceQAAgent)
    assert type(get_agent("news")).__name__ == "StubAgent"


async def test_reset_agents_restores_the_stub(real_agents: None) -> None:
    reset_agents()
    assert type(get_agent("finance_qa")).__name__ == "StubAgent"


@respx.mock
async def test_ask_keeps_sources_through_review_and_adds_disclaimer_once(real_agents: None) -> None:
    respx.post(COMPLETIONS_URL).mock(
        side_effect=FakeLLM(lambda n: f"Interest earns interest [{n['Compound interest']}].")
    )
    reviewed: list[tuple[str, str]] = []

    class RecordingReviewer(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            reviewed.append((question, reply))
            return False

    reply = await ask("What is compound interest?", classifier=FakeClassifier(), reviewer=RecordingReviewer())

    assert reply.route == "finance_qa"
    assert reply.sources == [COMPOUND]
    assert reply.text == f"Interest earns interest [1].\n\n{DISCLAIMER}"
    assert reviewed == [("What is compound interest?", "Interest earns interest [1].")]


@respx.mock
async def test_ask_advice_question_passes_review_with_sources(real_agents: None) -> None:
    respx.post(COMPLETIONS_URL).mock(
        side_effect=FakeLLM(lambda n: f"Index funds track an index [{n['Index Funds']}].")
    )
    reply = await ask("Which index fund should I buy?", classifier=_IndexFundClassifier(), reviewer=AllowAllReviewer())
    assert reply.seeks_advice is True
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert reply.text.count(DISCLAIMER) == 1
    assert [s.title for s in reply.sources] == ["Index Funds (Investor.gov)"]


class _IndexFundClassifier:
    async def classify(self, question, history):  # type: ignore[no-untyped-def]
        from finance_assistant.tutor import Classification

        return Classification(route="finance_qa", seeks_advice=True)


@respx.mock
async def test_ask_llm_failure_gives_agent_failure_text(real_agents: None, caplog: pytest.LogCaptureFixture) -> None:
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "secret sk-test"}}))
    with caplog.at_level(logging.WARNING):
        reply = await ask("What is compound interest?", classifier=FakeClassifier(), reviewer=AllowAllReviewer())
    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"
    assert reply.sources == []
    assert "sk-test" not in caplog.text
    assert "Agent finance_qa failed" in caplog.text


@respx.mock
async def test_ask_nothing_relevant_skips_llm_and_has_no_sources(
    real_agents: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = StubIndex([Hit(_chunk("a"), MIN_SCORE - 0.01)])
    monkeypatch.setattr(finance_qa, "get_index", lambda: index)
    install_real_agents()  # re-register so the agent picks up the stub index
    route = respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500))

    class MortgageClassifier:
        async def classify(self, question, history):  # type: ignore[no-untyped-def]
            from finance_assistant.tutor import Classification

            return Classification(route="finance_qa")

    reply = await ask("How do I refinance a mortgage on my house?", classifier=MortgageClassifier(), reviewer=AllowAllReviewer())
    assert not route.called
    assert reply.text == f"{NOT_COVERED_TEXT}\n\n{DISCLAIMER}"
    assert reply.sources == []
