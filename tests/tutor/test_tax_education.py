"""Tax Education agent: every I/O-matrix row offline (fake embedder, respx-mocked OpenAI)."""

from __future__ import annotations

import logging
import subprocess
import sys

import httpx
import pytest
import respx

from finance_assistant.knowledge import Hit, KnowledgeIndex, load_articles
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    DISCLAIMER,
    EDUCATION_SYSTEM_PROMPT,
    AgentRequest,
    Classification,
    TaxEducationAgent,
    ask,
    get_agent,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor import finance_qa
from finance_assistant.tutor.finance_qa import FINANCE_QA_INSTRUCTIONS, MIN_SCORE, NOT_COVERED_TEXT, FinanceQAAgent
from finance_assistant.tutor.tax_education import (
    TAX_EDUCATION_INSTRUCTIONS,
    TAX_NOT_COVERED_TEXT,
    TAX_SITUATION_REDIRECT,
    TAX_TOPICS,
)

from .test_ask import AllowAllReviewer, FakeClassifier
from .test_finance_qa import SETTINGS, FakeLLM, StubIndex, _chunk, kb, numbered_articles, real_agents  # noqa: F401
from .test_router import COMPLETIONS_URL

IRS = "IRS.gov"


@pytest.fixture
def agent(kb: KnowledgeIndex) -> TaxEducationAgent:  # noqa: F811
    return TaxEducationAgent(settings=SETTINGS, index_provider=lambda: kb)


def stub_agent(*scores: float) -> TaxEducationAgent:
    index = StubIndex([Hit(_chunk(f"a{i}"), score) for i, score in enumerate(scores)])
    return TaxEducationAgent(settings=SETTINGS, index_provider=lambda: index)  # type: ignore[arg-type,return-value]


# --- Matrix rows ------------------------------------------------------------------


@respx.mock
async def test_concept_comparison_cites_ira_articles(agent: TaxEducationAgent) -> None:
    def answer(n: dict[str, int]) -> str:
        trad, roth = n["Traditional IRAs"], n["Roth IRAs"]
        return f"Traditional IRA contributions may be deductible [{trad}]. Roth IRA withdrawals can be tax-free [{roth}]."

    llm = FakeLLM(answer)
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent.run(AgentRequest("What's the difference between a Traditional IRA and a Roth IRA?"))

    assert result.text == (
        "Traditional IRA contributions may be deductible [1]. Roth IRA withdrawals can be tax-free [2]."
    )
    assert [s.title for s in result.sources] == [f"Traditional IRAs ({IRS})", f"Roth IRAs ({IRS})"]
    assert all(s.url and s.url.startswith("https://www.irs.gov/") for s in result.sources)
    assert llm.system.startswith(EDUCATION_SYSTEM_PROMPT)
    assert "Tax Education agent" in llm.system and "Finance Q&A agent" not in llm.system
    assert TAX_SITUATION_REDIRECT in llm.system  # the exact sentence is given to the model
    assert "tax year" in llm.system
    assert DISCLAIMER not in result.text


@respx.mock
async def test_401k_vs_ira_returns_only_cited_articles(agent: TaxEducationAgent) -> None:
    llm = FakeLLM(lambda n: f"A 401(k) is offered by an employer [{n['401(k) Plans: IRS Overview']}].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent.run(AgentRequest("How is a 401(k) different from an IRA?"))

    assert len(numbered_articles(llm.prompt)) >= 2  # other articles were shown but not cited
    assert result.text == "A 401(k) is offered by an employer [1]."
    assert [s.title for s in result.sources] == [f"401(k) Plans: IRS Overview ({IRS})"]


@respx.mock
async def test_situation_specific_question_keeps_the_tax_redirect(agent: TaxEducationAgent) -> None:
    question = "I earn $150k and have a 401(k) at work. Can I deduct my IRA contribution?"
    body = "If you're covered by a workplace plan, the deduction may be reduced above certain incomes [{n}]. A qualified tax professional can help with your exact situation."
    llm = FakeLLM(lambda n: f"{TAX_SITUATION_REDIRECT}\n\n" + body.format(n=n["Traditional IRAs"]))
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent.run(AgentRequest(question))

    assert result.text.startswith(TAX_SITUATION_REDIRECT + "\n\n")
    assert result.text.count(TAX_SITUATION_REDIRECT) == 1
    assert ADVICE_REDIRECT not in result.text
    assert "tax professional" in result.text
    assert [s.title for s in result.sources] == [f"Traditional IRAs ({IRS})"]
    assert "personal recommendation" not in llm.prompt  # not advice-seeking: no advice redirect asked for


@respx.mock
async def test_advice_seeking_opens_with_advice_redirect_only(agent: TaxEducationAgent) -> None:
    llm = FakeLLM(lambda n: f"A Roth IRA is funded with after-tax money [{n['Roth IRAs']}].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent.run(AgentRequest("Should I open a Roth IRA?", seeks_advice=True))

    assert result.text == f"{ADVICE_REDIRECT}\n\nA Roth IRA is funded with after-tax money [1]."
    assert TAX_SITUATION_REDIRECT not in result.text
    assert [s.title for s in result.sources] == [f"Roth IRAs ({IRS})"]
    assert "personal recommendation" in llm.prompt and ADVICE_REDIRECT in llm.prompt


@pytest.mark.parametrize(
    "template",
    [
        "{advice}\n\n{tax}\n\nRoth money goes in after tax [{n}].",
        "{advice} {tax} Roth money goes in after tax [{n}].",
        "{tax}\n\nRoth money goes in after tax [{n}].",  # the agent prepends the advice redirect
        "{tax} {advice} Roth money goes in after tax [{n}].",  # tax sentence first
        "{advice}\n\n{curly_tax}\n\nRoth money goes in after tax [{n}].",  # curly apostrophe
    ],
)
@respx.mock
async def test_advice_question_never_gets_both_redirects(agent: TaxEducationAgent, template: str) -> None:
    respx.post(COMPLETIONS_URL).mock(
        side_effect=FakeLLM(
            lambda n: template.format(
                advice=ADVICE_REDIRECT,
                tax=TAX_SITUATION_REDIRECT,
                curly_tax=TAX_SITUATION_REDIRECT.replace("'", "\u2019"),
                n=n["Roth IRAs"],
            )
        )
    )
    result = await agent.run(AgentRequest("Should I open a Roth IRA?", seeks_advice=True))
    assert result.text.startswith(ADVICE_REDIRECT)
    assert result.text.count(ADVICE_REDIRECT) == 1
    assert TAX_SITUATION_REDIRECT not in result.text
    assert TAX_SITUATION_REDIRECT.replace("'", "\u2019") not in result.text
    assert result.text == f"{ADVICE_REDIRECT}\n\nRoth money goes in after tax [1]."
    assert len(result.sources) == 1


@respx.mock
async def test_nothing_relevant_gives_tax_not_covered_without_calling_the_llm(kb: KnowledgeIndex) -> None:  # noqa: F811
    route = respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500))

    on_real_kb = await TaxEducationAgent(settings=SETTINGS, index_provider=lambda: kb).run(
        AgentRequest("How do I file my state taxes?")
    )
    at_threshold = await stub_agent(MIN_SCORE - 0.01, 0.2).run(AgentRequest("How do I file my state taxes?"))

    assert not route.called
    for result in (on_real_kb, at_threshold):
        assert result.text == TAX_NOT_COVERED_TEXT
        assert result.sources == []
    assert "don't cover that tax question" in TAX_NOT_COVERED_TEXT


# Each TAX_TOPICS entry -> article slugs that explain it.
TOPIC_ARTICLES: dict[str, set[str]] = {
    "Traditional and Roth IRAs, and how they differ": {"traditional-and-roth-iras", "roth-iras", "traditional-iras"},
    "401(k) plans (including designated Roth accounts), 403(b) and 457(b) plans": {
        "401-k-plans",
        "retirement-topics-designated-roth-account",
        "403-b-and-457-b-plans",
    },
    "Contribution limits, catch-up contributions, and IRA deduction limits": {
        "retirement-topics-ira-contribution-limits",
        "retirement-topics-catch-up-contributions",
        "ira-deduction-limits",
    },
    "Rollovers, required minimum distributions (RMDs), and exceptions to the early-withdrawal tax": {
        "rollovers-of-retirement-plan-and-ira-distributions",
        "retirement-topics-required-minimum-distributions-rmds",
        "retirement-topics-exceptions-to-tax-on-early-distributions",
    },
    "The Saver's Credit": {"retirement-savings-contributions-credit-saver-s-credit"},
    "HSAs (health savings accounts) and 529 education savings plans": {"hsas-health-savings-accounts", "529-plan"},
}


def test_tax_topics_map_to_existing_retirement_and_tax_articles() -> None:
    tax_slugs = {a.slug for a in load_articles() if a.category == "retirement-and-tax"}
    assert set(TOPIC_ARTICLES) == set(TAX_TOPICS)
    for topic in TAX_TOPICS:
        assert f"- {topic}" in TAX_NOT_COVERED_TEXT
        missing = TOPIC_ARTICLES[topic] - tax_slugs
        assert not missing, f"{topic!r}: no retirement-and-tax article {sorted(missing)}"


def test_importing_the_tutor_package_stays_light() -> None:
    code = (
        "import sys, finance_assistant.tutor\n"
        "print(sorted(m for m in ('torch', 'faiss', 'sentence_transformers') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


@respx.mock
async def test_nothing_relevant_for_advice_question_still_redirects() -> None:
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500))
    result = await stub_agent(MIN_SCORE - 0.01).run(AgentRequest("Should I file my state taxes early?", seeks_advice=True))
    assert result.text == f"{ADVICE_REDIRECT}\n\n{TAX_NOT_COVERED_TEXT}"
    assert result.sources == []


@pytest.mark.parametrize("reply", ["NOT_COVERED", "A Roth IRA is funded after tax.", "Roth [9]."])
@respx.mock
async def test_model_declines_or_cites_nothing_valid_gives_tax_not_covered(agent: TaxEducationAgent, reply: str) -> None:
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: reply))
    result = await agent.run(AgentRequest("What is a Roth IRA?"))
    assert result.text == TAX_NOT_COVERED_TEXT
    assert result.sources == []


@respx.mock
async def test_hit_at_the_threshold_reaches_the_llm() -> None:
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: "Answer [1]."))
    result = await stub_agent(MIN_SCORE, 0.1).run(AgentRequest("A tax question?"))
    assert result.text == "Answer [1]."
    assert [s.url for s in result.sources] == ["https://example.com/a0"]


def test_finance_qa_keeps_its_own_instructions_and_not_covered_text() -> None:
    assert FinanceQAAgent.instructions == FINANCE_QA_INSTRUCTIONS
    assert FinanceQAAgent.not_covered_text == NOT_COVERED_TEXT
    assert TaxEducationAgent.instructions == TAX_EDUCATION_INSTRUCTIONS
    assert TaxEducationAgent.not_covered_text == TAX_NOT_COVERED_TEXT
    assert TAX_SITUATION_REDIRECT == (
        "I can't give tax advice for your specific situation, but I can explain how this works in general."
    )


# --- Through ask(), with install_real_agents() ------------------------------------


async def test_install_real_agents_registers_tax_education(real_agents: None) -> None:  # noqa: F811
    assert isinstance(get_agent("tax_education"), TaxEducationAgent)
    assert isinstance(get_agent("finance_qa"), FinanceQAAgent)
    assert not isinstance(get_agent("finance_qa"), TaxEducationAgent)
    reset_agents()
    assert type(get_agent("tax_education")).__name__ == "StubAgent"


@respx.mock
async def test_ask_routes_to_tax_education_and_keeps_sources_through_review(real_agents: None) -> None:  # noqa: F811
    llm = FakeLLM(lambda n: f"Roth IRA withdrawals can be tax-free [{n['Roth IRAs']}].")
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)
    reviewed: list[tuple[str, str]] = []

    class RecordingReviewer(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            reviewed.append((question, reply))
            return False

    question = "How is a Roth IRA different from a traditional IRA?"
    reply = await ask(question, classifier=FakeClassifier(), reviewer=RecordingReviewer())

    assert reply.route == "tax_education"
    assert [s.title for s in reply.sources] == [f"Roth IRAs ({IRS})"]
    assert reply.text == f"Roth IRA withdrawals can be tax-free [1].\n\n{DISCLAIMER}"
    assert reviewed == [(question, "Roth IRA withdrawals can be tax-free [1].")]
    assert "Tax Education agent" in llm.system


@respx.mock
async def test_ask_advice_question_passes_review_with_sources(real_agents: None) -> None:  # noqa: F811
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeLLM(lambda n: f"A Roth IRA grows tax-free [{n['Roth IRAs']}]."))
    reply = await ask("Should I open a Roth IRA?", classifier=FakeClassifier(), reviewer=AllowAllReviewer())
    assert (reply.route, reply.seeks_advice) == ("tax_education", True)
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert TAX_SITUATION_REDIRECT not in reply.text
    assert reply.text.count(DISCLAIMER) == 1
    assert [s.title for s in reply.sources] == [f"Roth IRAs ({IRS})"]


@respx.mock
async def test_ask_llm_failure_gives_agent_failure_text(
    real_agents: None,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "secret sk-test"}}))
    with caplog.at_level(logging.WARNING):
        reply = await ask("What is a Roth IRA?", classifier=FakeClassifier(), reviewer=AllowAllReviewer())
    assert reply.route == "tax_education"
    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"
    assert reply.sources == []
    assert "sk-test" not in caplog.text
    assert "Agent tax_education failed" in caplog.text


@respx.mock
async def test_ask_nothing_relevant_skips_llm_and_has_no_sources(
    real_agents: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = StubIndex([Hit(_chunk("a"), MIN_SCORE - 0.01)])
    monkeypatch.setattr(finance_qa, "get_index", lambda: index)
    install_real_agents()  # re-register so the agent picks up the stub index
    route = respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500))

    class TaxClassifier:
        async def classify(self, question, history):  # type: ignore[no-untyped-def]
            return Classification(route="tax_education")

    reply = await ask("How do I file my state taxes?", classifier=TaxClassifier(), reviewer=AllowAllReviewer())
    assert not route.called
    assert reply.route == "tax_education"
    assert reply.text == f"{TAX_NOT_COVERED_TEXT}\n\n{DISCLAIMER}"
    assert reply.sources == []
