"""Portfolio Analysis agent: every I/O-matrix row offline (respx for OpenAI and Alpha Vantage)."""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx

from finance_assistant import config
from finance_assistant.config import Settings
from finance_assistant.market_data import MarketDataClient, Quote
from finance_assistant.market_data.budget import CallBudget
from finance_assistant.market_data.client import ALPHA_VANTAGE_URL
from finance_assistant.portfolio import Holding
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    DISCLAIMER,
    EDUCATION_SYSTEM_PROMPT,
    AgentRequest,
    ChatTurn,
    Classification,
    Portfolio,
    PortfolioAnalysisAgent,
    StubAgent,
    ask,
    get_agent,
    install_real_agents,
    parse_holdings_csv,
    reset_agents,
)
from finance_assistant.tutor.market_analysis import ALPHA_VANTAGE_SOURCE
from finance_assistant.tutor.portfolio_analysis import (
    NO_HOLDINGS_TEXT,
    PORTFOLIO_ANALYSIS_INSTRUCTIONS,
    TYPED_NOTE,
    UPLOAD_NOTE,
)

from ..market_data.conftest import TEST_API_KEY, FakeClock, global_quote_for
from .test_ask import AllowAllReviewer
from .test_router import COMPLETIONS_URL
from .test_router import _completion as completion

SETTINGS = Settings(openai_api_key="sk-test", openai_model="gpt-4o-mini")
SAMPLE = Path(__file__).parent.parent / "data" / "sample_holdings.csv"
EXPLANATION = "Diversification means spreading money out. One company is a large share here."


class FakeOpenAI:
    """respx side effect for both LLM calls: holdings extraction (structured) and the explanation."""

    def __init__(self, holdings: list[dict] | None = None, explanation: str | Callable[[str], str] = EXPLANATION) -> None:
        self.holdings = holdings or []
        self.explanation = explanation
        self.extractions: list[dict] = []
        self.explains: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            self.extractions.append(body)
            content = json.dumps({"holdings": self.holdings})
        else:
            self.explains.append(body)
            prompt = body["messages"][-1]["content"]
            content = self.explanation(prompt) if callable(self.explanation) else self.explanation
        return httpx.Response(200, json=completion(content, body["model"]))

    @property
    def explain_system(self) -> str:
        return self.explains[0]["messages"][0]["content"]

    @property
    def explain_prompt(self) -> str:
        return self.explains[0]["messages"][-1]["content"]


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


class RecordingClient:
    """Wraps a client and records each symbol fetched."""

    def __init__(self, inner: MarketDataClient) -> None:
        self.inner = inner
        self.fetched: list[str] = []

    async def get_quote(self, symbol: str) -> Quote:
        self.fetched.append(symbol)
        return await self.inner.get_quote(symbol)


def mock_client() -> RecordingClient:
    return RecordingClient(MarketDataClient(Settings(market_data_mode="mock")))


def agent_with(client) -> PortfolioAnalysisAgent:  # type: ignore[no-untyped-def]
    return PortfolioAnalysisAgent(client_provider=lambda: client, settings=SETTINGS)


def sample() -> Portfolio:
    return parse_holdings_csv(SAMPLE.read_bytes())


def table_rows(text: str) -> list[list[str]]:
    rows = [line for line in text.splitlines() if line.startswith("| ") and not line.startswith("| Ticker")]
    return [[c.strip() for c in line.strip("|").split("|")] for line in rows]


# --- Matrix: sample upload ---------------------------------------------------------


async def test_sample_upload_shows_table_metrics_findings_then_explanation(router) -> None:
    llm = FakeOpenAI()
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    client = mock_client()

    result = await agent_with(client).run(AgentRequest("How diversified is my portfolio?", portfolio=sample()))

    assert llm.extractions == []  # an upload needs no extraction
    assert client.fetched == ["VOO", "BND", "MSFT"]  # only share rows are priced, in file order
    assert result.text.startswith(UPLOAD_NOTE)
    assert table_rows(result.text) == [
        ["AAPL", "—", "—", "$7,250.00", "30.0%", "Single company", "value you gave"],
        ["VOO", "10", "$702.55", "$7,025.50", "29.1%", "Broad US stock fund", "demo"],
        ["VXUS", "—", "—", "$3,000.00", "12.4%", "International stock fund", "value you gave"],
        ["BND", "40", "$74.62", "$2,984.80", "12.4%", "Bond fund", "demo"],
        ["MSFT", "5", "$538.75", "$2,693.75", "11.2%", "Single company", "demo"],
        ["NVDA", "—", "—", "$1,200.00", "5.0%", "Single company", "value you gave"],
    ]
    assert "Total priced value: $24,154.05" in result.text
    assert "demo = a bundled sample quote for trading day 2026-09-29, not a real-time price" in result.text
    assert "- Holdings valued: 6 of 6" in result.text
    assert "- Three largest companies together: AAPL, MSFT, NVDA, 46.1%" in result.text
    assert (
        "- Concentration risk: AAPL is a single company at 30.0% of your priced value "
        "(one company at 20% or more)." in result.text
    )
    assert "- Diversification level: **low** (concentration risk was found)." in result.text
    assert "AAPL, MSFT, NVDA aren't on my list of common funds" in result.text
    assert result.text.endswith(EXPLANATION)
    assert result.sources == []  # demo quotes only
    assert DISCLAIMER not in result.text and ADVICE_REDIRECT not in result.text
    # The explanation is built with the shared rule and sees words only, never the figures.
    assert llm.explain_system.startswith(EDUCATION_SYSTEM_PROMPT)
    assert PORTFOLIO_ANALYSIS_INSTRUCTIONS.splitlines()[0] in llm.explain_system
    prompt = llm.explain_prompt
    assert "Concentration risk: AAPL, a single company" in prompt
    assert "The diversification level is low" in prompt
    assert "Question: How diversified is my portfolio?" in prompt
    description = prompt.split("Question:")[0]
    assert not any(ch.isdigit() for ch in description)


# --- Matrix: broad fund only -------------------------------------------------------


async def test_broad_fund_only_is_not_single_company_risk_and_level_high(router) -> None:
    llm = FakeOpenAI()
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    portfolio = Portfolio((Holding("VOO", value=Decimal(10000)),))

    result = await agent_with(mock_client()).run(AgentRequest("How diversified am I?", portfolio=portfolio))

    assert table_rows(result.text) == [
        ["VOO", "—", "—", "$10,000.00", "100.0%", "Broad US stock fund", "value you gave"]
    ]
    assert "- Concentration risk: none found." in result.text
    assert "- Diversification level: **high**" in result.text
    assert "Single company" not in result.text and "aren't on my list" not in result.text
    assert "isn't on my list" not in result.text
    prompt = llm.explain_prompt
    assert "VOO is a broad US stock fund: one holding that owns shares of many US companies" in prompt
    assert "A large fund holding is not single-company risk" in prompt


# --- Matrix: typed holdings ----------------------------------------------------------


async def test_typed_holdings_are_extracted_and_analysed(router) -> None:
    llm = FakeOpenAI(
        [
            {"ticker": "AAPL", "shares": 10, "value": None},
            {"ticker": "msft", "shares": 5, "value": None},
            {"ticker": "VTI", "shares": None, "value": 8000},
        ]
    )
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    client = mock_client()
    question = "I have 10 AAPL, 5 MSFT and $8,000 in VTI. How diversified is that?"

    result = await agent_with(client).run(AgentRequest(question))

    assert llm.extractions[0]["response_format"]["type"] == "json_schema"
    assert llm.extractions[0]["messages"][-1]["content"].endswith(f"Latest message: {question}")
    assert client.fetched == ["AAPL", "MSFT"]
    assert result.text.startswith(TYPED_NOTE)
    assert table_rows(result.text) == [
        ["VTI", "—", "—", "$8,000.00", "59.9%", "Broad US stock fund", "value you gave"],
        ["MSFT", "5", "$538.75", "$2,693.75", "20.2%", "Single company", "demo"],
        ["AAPL", "10", "$265.40", "$2,654.00", "19.9%", "Single company", "demo"],
    ]
    assert "- Concentration risk: MSFT is a single company at 20.2%" in result.text
    assert "- Diversification level: **low**" in result.text
    assert result.text.endswith(EXPLANATION)


async def test_typed_holdings_use_history_for_follow_ups(router) -> None:
    llm = FakeOpenAI([{"ticker": "VTI", "shares": None, "value": 500}])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    history = (ChatTurn("user", "I own $500 of VTI"), ChatTurn("assistant", "Noted."))

    await agent_with(mock_client()).run(AgentRequest("How diversified is that?", history))

    prompt = llm.extractions[0]["messages"][-1]["content"]
    assert "User: I own $500 of VTI" in prompt


async def test_bad_typed_holdings_are_listed_not_guessed(router) -> None:
    llm = FakeOpenAI(
        [
            {"ticker": "VTI", "shares": None, "value": 1000},
            {"ticker": "AAPL", "shares": -3, "value": None},
            {"ticker": "MSFT", "shares": None, "value": None},
            {"ticker": "NOT A TICKER!", "shares": 1, "value": None},
        ]
    )
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(AgentRequest("I have VTI, AAPL, MSFT and junk"))

    assert [r[0] for r in table_rows(result.text)] == ["VTI"]
    assert "- Holdings valued: 1 of 4" in result.text
    assert "- AAPL (as typed): the shares must be more than 0." in result.text
    assert "- MSFT (as typed): no number of shares or dollar value was given." in result.text
    # The raw extracted text is reduced to a safe label before it is shown or sent to the model.
    assert "- NOTATICKER (as typed): that doesn't look like a valid ticker symbol." in result.text
    assert "NOT A TICKER!" not in result.text


async def test_huge_typed_amounts_are_excluded_not_crashing(router) -> None:
    llm = FakeOpenAI(
        [
            {"ticker": "VTI", "shares": None, "value": 1000},
            {"ticker": "VOO", "shares": None, "value": 1e30},
            {"ticker": "BND", "shares": None, "value": 9e11},
            {"ticker": "BND", "shares": None, "value": 9e11},
        ]
    )
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(AgentRequest("I have VTI, VOO and BND"))

    assert [r[0] for r in table_rows(result.text)] == ["VTI"]
    assert "- VOO (as typed): the value is too large." in result.text
    assert "- BND (as typed): the combined amounts are too large." in result.text


async def test_typed_holding_with_shares_and_value_uses_the_value(router) -> None:
    llm = FakeOpenAI([{"ticker": "AAPL", "shares": 10, "value": 2000}])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    client = mock_client()

    result = await agent_with(client).run(AgentRequest("I have 10 AAPL worth $2,000"))

    assert client.fetched == []
    assert table_rows(result.text) == [["AAPL", "—", "—", "$2,000.00", "100.0%", "Single company", "value you gave"]]


# --- Matrix: upload wins -------------------------------------------------------------


async def test_upload_wins_over_holdings_in_the_question(router) -> None:
    llm = FakeOpenAI([{"ticker": "TSLA", "shares": 100, "value": None}])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    client = mock_client()

    result = await agent_with(client).run(
        AgentRequest("I have 100 TSLA, how diversified am I?", portfolio=sample())
    )

    assert llm.extractions == []
    assert "TSLA" not in client.fetched and "| TSLA" not in result.text
    assert result.text.startswith(UPLOAD_NOTE)
    assert "uses the holdings file you uploaded" in result.text


# --- Matrix: no holdings ---------------------------------------------------------------


async def test_no_holdings_explains_how_to_upload_without_fetch_or_explanation(router) -> None:
    llm = FakeOpenAI([])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    client = mock_client()

    result = await agent_with(client).run(AgentRequest("How diversified is my portfolio?"))

    assert result.text == NO_HOLDINGS_TEXT
    assert "ticker,shares,value" in result.text and "type your holdings" in result.text
    assert client.fetched == [] and llm.explains == []
    # The example in the instructions parses as a valid upload.
    example = result.text.split("```")[1].strip()
    assert len(parse_holdings_csv(example).holdings) == 3


async def test_no_holdings_for_advice_question_opens_with_redirect(router) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI([{"ticker": " ", "shares": 1, "value": None}]))
    result = await agent_with(mock_client()).run(AgentRequest("Should I rebalance?", seeks_advice=True))
    assert result.text == f"{ADVICE_REDIRECT}\n\n{NO_HOLDINGS_TEXT}"


# --- Matrix: unpriceable rows / nothing priced ------------------------------------------


async def test_unpriceable_row_is_listed_and_excluded(router) -> None:
    llm = FakeOpenAI()
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    portfolio = parse_holdings_csv("ticker,shares,value\nZZZZ,10,\nVTI,,1000\n")

    result = await agent_with(mock_client()).run(AgentRequest("How diversified?", portfolio=portfolio))

    assert [r[0] for r in table_rows(result.text)] == ["VTI"]
    assert "100.0%" in result.text  # weights use priced value only
    assert (
        "**Not valued (left out of the weights)**\n- ZZZZ (10 shares): no quote is available for it "
        "(demo data covers only a few tickers); you can give its dollar value instead." in result.text
    )
    assert "- Holdings valued: 1 of 2" in result.text
    assert "could not be valued and are left out of the weights: ZZZZ" in llm.explain_prompt


async def test_live_unknown_symbol_is_listed_and_live_source_cited(router) -> None:
    clock = FakeClock()
    llm = FakeOpenAI()
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    def respond(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["symbol"]
        if symbol == "APPL":
            return httpx.Response(200, json={"Global Quote": {}})
        return httpx.Response(200, json=global_quote_for(symbol, "250.0000"))

    av = router.get(ALPHA_VANTAGE_URL, params={"function": "GLOBAL_QUOTE"}).mock(side_effect=respond)
    client = MarketDataClient(
        Settings(alpha_vantage_api_key=TEST_API_KEY), clock=clock, budget=CallBudget(clock=clock, sleep=clock.sleep)
    )
    portfolio = parse_holdings_csv("ticker,shares\nAPPL,3\nVTI,4\n")

    result = await agent_with(client).run(AgentRequest("How diversified?", portfolio=portfolio))

    assert [c.request.url.params["symbol"] for c in av.calls] == ["APPL", "VTI"]  # one at a time
    assert clock.sleeps == [pytest.approx(1.1)]
    assert "- APPL (3 shares): no quote exists for this ticker; please check it." in result.text
    assert table_rows(result.text) == [["VTI", "4", "$250.00", "$1,000.00", "100.0%", "Broad US stock fund", "live"]]
    assert "live = fetched just now from Alpha Vantage" in result.text
    assert result.sources == [ALPHA_VANTAGE_SOURCE]


async def test_merged_shares_and_value_are_priced_together(router) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI())
    portfolio = parse_holdings_csv("ticker,shares,value\nVOO,10,\nVOO,,50\nVTI,,1000\n")

    result = await agent_with(mock_client()).run(AgentRequest("How diversified?", portfolio=portfolio))

    # 10 × $702.55 (mock) + $50 given = $7,075.50
    assert table_rows(result.text)[0] == [
        "VOO", "10", "$702.55", "$7,075.50", "87.6%", "Broad US stock fund", "demo + value you gave"
    ]


async def test_merged_ticker_with_unpriceable_shares_is_excluded_whole(router) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI())
    portfolio = parse_holdings_csv("ticker,shares,value\nZZZZ,10,\nZZZZ,,50\nVTI,,1000\n")

    result = await agent_with(mock_client()).run(AgentRequest("How diversified?", portfolio=portfolio))

    assert [r[0] for r in table_rows(result.text)] == ["VTI"]
    assert "- ZZZZ (10 shares + $50.00): no quote is available for it" in result.text
    assert "Total priced value: $1,000.00" in result.text


async def test_cache_and_stale_cache_sources_are_labelled(router) -> None:
    clock = FakeClock()
    llm = FakeOpenAI()
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    router.get(ALPHA_VANTAGE_URL, params={"function": "GLOBAL_QUOTE"}).mock(
        side_effect=lambda request: httpx.Response(
            200, json=global_quote_for(request.url.params["symbol"], "100.0000")
        )
    )
    client = MarketDataClient(
        Settings(alpha_vantage_api_key=TEST_API_KEY), clock=clock, budget=CallBudget(clock=clock, sleep=clock.sleep)
    )
    agent = agent_with(client)
    await client.get_quote("VTI")  # cached at 14:00
    clock.advance(31 * 60)
    await client.get_quote("BND")  # cached at 14:31; VTI's copy is now expired
    clock.advance(60)
    client.budget.pause(3600)  # no live calls: BND from cache, VTI from the stale copy
    portfolio = parse_holdings_csv("ticker,shares\nVTI,1\nBND,1\n")

    result = await agent.run(AgentRequest("How diversified?", portfolio=portfolio))

    cells = {r[0]: r[6] for r in table_rows(result.text)}
    assert cells == {"VTI": "stale cache", "BND": "cache"}
    assert "cache = a recent cached copy from Alpha Vantage" in result.text
    assert "stale cache = an older cached copy from Alpha Vantage" in result.text
    assert "Some share prices are older cached copies and may be out of date." in llm.explain_prompt
    assert result.sources == [ALPHA_VANTAGE_SOURCE]


async def test_presence_of_bonds_comes_from_rows_and_one_kind_is_described(router) -> None:
    llm = FakeOpenAI()
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    portfolio = parse_holdings_csv("ticker,value\nVTI,100000\nBND,1\n")  # BND rounds to 0.0 %

    await agent_with(mock_client()).run(AgentRequest("How diversified?", portfolio=portfolio))

    assert "None of the priced value is in bond funds" not in llm.explain_prompt
    assert "Most of the priced value is in stocks; a smaller part is in bond funds." in llm.explain_prompt
    assert "Almost all of the priced value is in one kind of holding: broad US stock funds." in llm.explain_prompt


async def test_all_bonds_is_described_as_one_kind(router) -> None:
    llm = FakeOpenAI()
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    portfolio = Portfolio((Holding("BND", value=Decimal(5000)),))

    result = await agent_with(mock_client()).run(AgentRequest("How diversified?", portfolio=portfolio))

    assert "- Diversification level: **high**" in result.text  # level rule unchanged
    assert "Almost all of the priced value is in one kind of holding: bond funds." in llm.explain_prompt


async def test_nothing_priced_says_so_without_explanation(router) -> None:
    llm = FakeOpenAI()
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    portfolio = parse_holdings_csv("ticker,shares\nZZZZ,1\nYYYY,2\n")

    result = await agent_with(mock_client()).run(AgentRequest("How diversified?", portfolio=portfolio))

    assert "I couldn't value any of your holdings" in result.text
    assert "- ZZZZ (1 share)" in result.text and "- YYYY (2 shares)" in result.text
    assert "Diversification level" not in result.text
    assert llm.explains == []


# --- Matrix: advice-seeking ---------------------------------------------------------------


@pytest.mark.parametrize("model_redirect", [ADVICE_REDIRECT, ADVICE_REDIRECT.replace("'", "’")])
async def test_advice_seeking_opens_with_redirect_once(router, model_redirect: str) -> None:
    llm = FakeOpenAI(explanation=f"{model_redirect} {EXPLANATION}")
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(
        AgentRequest("Should I sell some AAPL?", seeks_advice=True, portfolio=sample())
    )

    assert result.text.startswith(f"{ADVICE_REDIRECT}\n\n{UPLOAD_NOTE}")
    assert result.text.count(ADVICE_REDIRECT) == 1
    assert ADVICE_REDIRECT.replace("'", "’") not in result.text
    assert "Concentration risk: AAPL" in result.text
    assert result.text.endswith(EXPLANATION)
    assert "personal recommendation" in llm.explain_prompt


# --- The LLM never retypes a figure -------------------------------------------------------


async def test_explanation_sentences_with_figures_are_dropped(router, caplog: pytest.LogCaptureFixture) -> None:
    router.post(COMPLETIONS_URL).mock(
        side_effect=FakeOpenAI(explanation="AAPL is 30% of your money. VOO tracks the S&P 500. A fund holds many companies.")
    )
    result = await agent_with(mock_client()).run(AgentRequest("Diversified?", portfolio=sample()))
    assert result.text.endswith("VOO tracks the S&P 500. A fund holds many companies.")
    assert "dropped 1 explanation sentence" in caplog.text


async def test_explanation_made_only_of_figures_falls_back(router) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(explanation="AAPL: 30.0%."))
    result = await agent_with(mock_client()).run(AgentRequest("Diversified?", portfolio=sample()))
    explanation = result.text.rsplit("\n\n", 1)[1]
    assert explanation.startswith("Diversification means spreading money")
    assert "Concentration risk: AAPL, a single company" in explanation
    assert not any(ch.isdigit() for ch in explanation)


# --- Matrix: explanation fails, registration, ask() --------------------------------------------


class _PortfolioClassifier:
    def __init__(self, seeks_advice: bool = False) -> None:
        self.seeks_advice = seeks_advice

    async def classify(self, question, history):  # type: ignore[no-untyped-def]
        return Classification(route="portfolio", seeks_advice=self.seeks_advice)


@respx.mock
async def test_explanation_failure_gives_the_graphs_failure_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    config.reset_settings()
    install_real_agents()
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))

    reply = await ask(
        "How diversified is my portfolio?",
        portfolio=sample(),
        classifier=_PortfolioClassifier(),
        reviewer=AllowAllReviewer(),
    )

    assert reply.route == "portfolio"
    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"


@respx.mock
async def test_holdings_extraction_failure_gives_the_graphs_failure_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    config.reset_settings()
    install_real_agents()
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))

    reply = await ask(
        "I have 10 AAPL. How diversified is that?", classifier=_PortfolioClassifier(), reviewer=AllowAllReviewer()
    )

    assert reply.route == "portfolio"
    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"


async def test_install_real_agents_registers_portfolio_and_reset_restores_stub() -> None:
    install_real_agents()
    assert isinstance(get_agent("portfolio"), PortfolioAnalysisAgent)
    reset_agents()
    assert type(get_agent("portfolio")) is StubAgent


@respx.mock
async def test_ask_with_portfolio_routes_to_portfolio_with_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    config.reset_settings()
    install_real_agents()
    llm = FakeOpenAI()
    respx.post(COMPLETIONS_URL).mock(side_effect=llm)
    reviewed: list[tuple[str, str]] = []

    class RecordingReviewer(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            reviewed.append((question, reply))
            return False

    reply = await ask(
        "How diversified is my portfolio?",
        portfolio=sample(),
        classifier=_PortfolioClassifier(),
        reviewer=RecordingReviewer(),
    )

    assert reply.route == "portfolio"
    assert "Concentration risk: AAPL" in reply.text and "Diversification level: **low**" in reply.text
    assert reply.text.count(DISCLAIMER) == 1 and reply.text.endswith(DISCLAIMER)
    assert len(reviewed) == 1 and reviewed[0][1].startswith(UPLOAD_NOTE)
    assert llm.extractions == []


@respx.mock
async def test_ask_reviewer_flagging_advice_replaces_the_portfolio_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    config.reset_settings()
    install_real_agents()
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(explanation="You should sell AAPL."))

    class FlagAll(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            return True

    reply = await ask(
        "Should I sell some AAPL?", portfolio=sample(), classifier=_PortfolioClassifier(True), reviewer=FlagAll()
    )

    assert reply.route == "portfolio"
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert "You should sell" not in reply.text and "| AAPL" not in reply.text


async def test_ask_rejects_a_portfolio_of_the_wrong_type() -> None:
    with pytest.raises(TypeError, match="Portfolio"):
        await ask("How diversified?", portfolio="ticker,shares\nVOO,1\n", classifier=_PortfolioClassifier())  # type: ignore[arg-type]


async def test_other_agents_ignore_the_portfolio() -> None:
    class MarketClassifier:
        async def classify(self, question, history):  # type: ignore[no-untyped-def]
            return Classification(route="market")

    reply = await ask("What's VOO at?", portfolio=sample(), classifier=MarketClassifier(), reviewer=AllowAllReviewer())
    assert reply.route == "market" and reply.text.startswith("[Market Analysis — coming soon]")
