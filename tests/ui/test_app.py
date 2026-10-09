"""The Streamlit app driven with ``AppTest``: fake classifier, fake agents, no network.

Covers every row of story 9's I/O matrix plus the five tabs rendering with no keys.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    DISCLAIMER,
    AgentRequest,
    AgentResult,
    Source,
    register_agent,
)
from finance_assistant.knowledge.articles import load_articles
from finance_assistant.ui import app as ui_app
from finance_assistant.ui.helpers import source_links

from ..tutor.test_ask import AllowAllReviewer, FakeClassifier

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app.py"
SAMPLE_CSV = ROOT / "tests" / "data" / "sample_holdings.csv"


class RecordingAgent:
    """A fake real agent: answers with ``text`` (or the redirect) and records requests."""

    def __init__(self, text: str, sources: list[Source] | None = None) -> None:
        self.text, self.sources = text, sources or []
        self.requests: list[AgentRequest] = []

    async def run(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        if request.seeks_advice:
            return AgentResult(text=ADVICE_REDIRECT)
        return AgentResult(text=self.text, sources=list(self.sources))


class FailingAgent:
    async def run(self, request: AgentRequest) -> AgentResult:
        raise RuntimeError("agent crashed")


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def classifier(monkeypatch: pytest.MonkeyPatch) -> FakeClassifier:
    """Offline router + reviewer, and no real agents installed."""
    fake = FakeClassifier()
    monkeypatch.setattr(ui_app, "router_classifier", lambda: fake)
    monkeypatch.setattr(ui_app, "advice_reviewer", AllowAllReviewer)
    monkeypatch.setattr(ui_app, "install_real_agents", lambda: None)
    return fake


@pytest.fixture
def agents(classifier: FakeClassifier) -> dict[str, RecordingAgent]:
    made = {
        "finance_qa": RecordingAgent(
            "Compound interest is interest on interest [1].",
            [Source("What Is Compound Interest?", "https://www.investor.gov/compound-interest")],
        ),
        "portfolio": RecordingAgent("Your holdings are moderately diversified."),
        "market": RecordingAgent("AAPL last traded at $190.00, up $1.50."),
        "goal_planning": RecordingAgent("Save $787.39 a month: $47,243 in, $2,757 interest."),
        "news": RecordingAgent(
            "Stocks rose today [1].", [Source("Markets rally", "https://news.example.com/rally")]
        ),
        "tax_education": RecordingAgent(
            "A Roth IRA is funded with after-tax money [1].",
            [Source("Roth IRAs", "https://www.irs.gov/roth-iras")],
        ),
    }
    for route, agent in made.items():
        register_agent(route, agent)
    return made


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def markdown_in(node) -> list[str]:  # type: ignore[no-untyped-def]
    return [m.value for m in node.markdown]


def tab(at: AppTest, name: str):  # type: ignore[no-untyped-def]
    return next(t for t in at.tabs if t.label == name)


def submit(at: AppTest, key: str, text: str) -> AppTest:
    at.text_input(key=f"{key}_question").set_value(text)
    at.button(key=f"FormSubmitter:{key}_form-Ask").click().run()
    assert not at.exception
    return at


def test_all_five_tabs_render_with_no_keys() -> None:
    at = run_app()  # real install_real_agents, real classifier factory, no keys
    assert [t.label for t in at.tabs] == ["Chat", "Portfolio", "Markets", "Goals", "Knowledge"]
    assert not at.error
    knowledge = "\n".join(markdown_in(tab(at, "Knowledge")))
    first = load_articles()[0]
    assert f"- {source_links([Source(first.title, first.url)])[0]} — " in knowledge
    assert tab(at, "Knowledge").expander[0].label.startswith("Browse the knowledge base (")
    assert len(at.file_uploader) == 1
    assert len(at.chat_input) == 1


def test_chat_routes_with_history_and_lists_source_links(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent]
) -> None:
    at = run_app()
    at.chat_input(key="chat_input").set_value("What is compound interest?").run()
    at.chat_input(key="chat_input").set_value("What about Roth?").run()
    assert not at.exception

    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    first, second = (markdown_in(m) for m in (at.chat_message[1], at.chat_message[3]))
    assert first[0].startswith("Compound interest is interest on interest")
    assert first[0].endswith(DISCLAIMER)
    assert "- [What Is Compound Interest?](https://www.investor.gov/compound-interest)" in first[1]
    assert "- [Roth IRAs](https://www.irs.gov/roth-iras)" in second[1]
    # The follow-up carried the first exchange as history.
    question, history = classifier.calls[-1]
    assert question == "What about Roth?"
    assert [t.content for t in history][0] == "What is compound interest?"
    assert agents["tax_education"].requests[0].history[0].content == "What is compound interest?"


def test_scoped_tab_uses_its_agent_even_if_router_disagrees(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent]
) -> None:
    at = submit(run_app(), "market", "What is compound interest?")  # router would pick finance_qa
    assert len(agents["market"].requests) == 1
    assert agents["finance_qa"].requests == []

    submit(at, "news", "Any stock news today?")  # router would pick market
    assert len(agents["news"].requests) == 1
    assert "- [Markets rally](https://news.example.com/rally)" in "\n".join(markdown_in(tab(at, "Markets")))


def test_scoped_tab_still_redirects_advice(classifier: FakeClassifier, agents: dict[str, RecordingAgent]) -> None:
    at = submit(run_app(), "market", "Should I buy AAPL?")
    request = agents["market"].requests[0]
    assert request.seeks_advice is True
    text = "\n".join(markdown_in(tab(at, "Markets")))
    assert ADVICE_REDIRECT.replace("$", r"\$") in text


def test_portfolio_upload_preview_then_analysis(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent]
) -> None:
    at = run_app()
    at.file_uploader(key="portfolio_upload").set_value(("holdings.csv", SAMPLE_CSV.read_bytes(), "text/csv"))
    at.run()
    assert not at.exception and not at.error
    preview = tab(at, "Portfolio").dataframe[0].value
    assert list(preview["Ticker"]) == ["VOO", "AAPL", "BND", "VXUS", "MSFT", "NVDA"]

    submit(at, "portfolio", "Is my mix too risky?")  # router would not pick portfolio
    request = agents["portfolio"].requests[0]
    assert request.portfolio is not None
    assert [h.ticker for h in request.portfolio.holdings][:2] == ["VOO", "AAPL"]
    assert "moderately diversified" in "\n".join(markdown_in(tab(at, "Portfolio")))


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"symbolx,shares\nVOO,10\n", ["Row 1: no ticker column"]),
        (b"ticker,shares\nVOO,ten\nBND,\n", ["Row 2: shares 'ten' isn't a number", "Row 3: give shares or value"]),
    ],
)
def test_bad_upload_lists_problems_and_sends_nothing(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent], data: bytes, expected: list[str]
) -> None:
    at = run_app()
    at.file_uploader(key="portfolio_upload").set_value(("bad.csv", data, "text/csv"))
    at.run()
    errors = [e.value for e in tab(at, "Portfolio").error]
    for problem in expected:
        assert any(f"- {problem}" in e for e in errors)

    submit(at, "portfolio", "How diversified is my portfolio?")
    assert classifier.calls == []
    assert agents["portfolio"].requests == []
    assert any("nothing was sent" in e.value for e in tab(at, "Portfolio").error)


def test_goals_form_sends_values_in_words_and_escapes_dollars(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent]
) -> None:
    at = run_app()
    at.number_input(key="goal_target").set_value(50000.0)
    at.number_input(key="goal_years").set_value(5)
    at.number_input(key="goal_rate").set_value(6.0)
    at.number_input(key="goal_current").set_value(2500.0)
    at.button(key="FormSubmitter:goals_form-Calculate").click().run()
    assert not at.exception

    question = agents["goal_planning"].requests[0].question
    assert question == (
        "I want to reach $50,000 in 5 years. Assume an expected annual return of 6%. "
        "I currently have $2,500 saved. How much would I need to save each month?"
    )
    shown = "\n".join(markdown_in(tab(at, "Goals")))
    assert r"Save \$787.39 a month: \$47,243 in, \$2,757 interest." in shown
    assert r"I want to reach \$50,000" in shown


def test_goals_form_without_target_sends_nothing(classifier: FakeClassifier, agents) -> None:  # type: ignore[no-untyped-def]
    at = run_app()
    at.button(key="FormSubmitter:goals_form-Calculate").click().run()
    assert classifier.calls == []
    assert any("nothing was sent" in e.value for e in tab(at, "Goals").error)


def test_market_dollar_amounts_are_escaped(classifier: FakeClassifier, agents: dict[str, RecordingAgent]) -> None:
    at = submit(run_app(), "market", "What is AAPL trading at?")
    assert r"AAPL last traded at \$190.00, up \$1.50." in "\n".join(markdown_in(tab(at, "Markets")))


def test_knowledge_question_is_routed_by_the_router(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent]
) -> None:
    at = submit(run_app(), "knowledge", "How does a Roth IRA work?")
    assert len(agents["tax_education"].requests) == 1
    assert "- [Roth IRAs](https://www.irs.gov/roth-iras)" in "\n".join(markdown_in(tab(at, "Knowledge")))


@pytest.mark.parametrize(
    "ask_in_tab",
    [
        lambda at: at.chat_input(key="chat_input").set_value("Is a $5 ETF cheap?").run(),
        lambda at: submit(at, "market", "Is a $5 ETF cheap?"),
        lambda at: submit(at, "knowledge", "Is a $5 ETF cheap?"),
    ],
)
def test_missing_openai_key_is_a_message(monkeypatch: pytest.MonkeyPatch, ask_in_tab: Callable) -> None:
    monkeypatch.setattr(ui_app, "install_real_agents", lambda: None)
    at = run_app()  # real classifier factory, no OPENAI_API_KEY
    ask_in_tab(at)
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any(e.startswith("OPENAI_API_KEY is not set") for e in errors)
    assert any(e.endswith(r"Your question: Is a \$5 ETF cheap?") for e in errors)


def test_agent_failure_shows_failure_reply(classifier: FakeClassifier) -> None:
    register_agent("market", FailingAgent())
    at = submit(run_app(), "market", "What is AAPL at?")
    shown = "\n".join(markdown_in(tab(at, "Markets")))
    assert AGENT_FAILURE_TEXT in shown
    assert DISCLAIMER in shown


def test_empty_input_sends_nothing(classifier: FakeClassifier, agents: dict[str, RecordingAgent]) -> None:
    at = run_app()
    at.chat_input(key="chat_input").set_value("   ").run()
    for key in ("portfolio", "market", "news", "knowledge"):
        submit(at, key, "")
    assert classifier.calls == []
    assert not at.chat_message
    assert not at.error


def _set_goal(at: AppTest, target, years, rate=None, current=0.0) -> AppTest:  # type: ignore[no-untyped-def]
    at.number_input(key="goal_target").set_value(target)
    at.number_input(key="goal_years").set_value(years)
    at.number_input(key="goal_rate").set_value(rate)
    at.number_input(key="goal_current").set_value(current)
    at.button(key="FormSubmitter:goals_form-Calculate").click().run()
    assert not at.exception
    return at


def test_goals_error_clears_after_a_successful_answer(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent]
) -> None:
    at = _set_goal(run_app(), None, None)
    assert tab(at, "Goals").error
    _set_goal(at, 50000.0, 5, 6.0, 2500.0)
    assert len(agents["goal_planning"].requests) == 1
    assert not tab(at, "Goals").error


def test_goals_submits_carry_no_history(classifier: FakeClassifier, agents: dict[str, RecordingAgent]) -> None:
    at = _set_goal(run_app(), 50000.0, 5, 6.0)
    _set_goal(at, 20000.0, 3)
    requests = agents["goal_planning"].requests
    assert len(requests) == 2
    assert requests[1].history == ()
    assert classifier.calls[1][1] == ()


@pytest.mark.parametrize("fix", ["good_file", "remove"])
def test_bad_upload_error_clears_when_file_is_fixed_or_removed(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent], fix: str
) -> None:
    at = run_app()
    at.file_uploader(key="portfolio_upload").set_value(("bad.csv", b"symbolx,shares\nVOO,1\n", "text/csv"))
    at.run()
    submit(at, "portfolio", "How diversified is my portfolio?")
    assert any(e.value == ui_app.BAD_UPLOAD_TEXT for e in tab(at, "Portfolio").error)

    uploader = at.file_uploader(key="portfolio_upload")
    if fix == "good_file":
        uploader.set_value(("holdings.csv", SAMPLE_CSV.read_bytes(), "text/csv"))
    else:
        uploader.clear()
    at.run()
    assert not at.exception
    assert not tab(at, "Portfolio").error
    assert agents["portfolio"].requests == []


def test_portfolio_question_without_upload_reaches_portfolio_agent(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent]
) -> None:
    submit(run_app(), "portfolio", "I hold 10 VOO and 5 AAPL. How concentrated is that?")
    requests = agents["portfolio"].requests
    assert len(requests) == 1
    assert requests[0].portfolio is None


@pytest.mark.parametrize("key", ["portfolio", "market", "news", "knowledge"])
def test_whitespace_only_form_input_sends_nothing(
    classifier: FakeClassifier, agents: dict[str, RecordingAgent], key: str
) -> None:
    at = submit(run_app(), key, "   \t ")
    assert classifier.calls == []
    assert not at.chat_message
    assert not at.error
