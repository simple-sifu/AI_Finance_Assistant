"""Market Analysis agent: every I/O-matrix row offline (respx for OpenAI and Alpha Vantage)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from datetime import UTC, date, datetime

import httpx
import pytest
import respx

from finance_assistant import config
from finance_assistant.config import Settings
from finance_assistant.market_data import MarketDataClient, Quote
from finance_assistant.market_data.budget import CallBudget
from finance_assistant.market_data.client import ALPHA_VANTAGE_URL
from finance_assistant.market_data.mock import get_mock_quote, mock_symbols
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    DISCLAIMER,
    EDUCATION_SYSTEM_PROMPT,
    AgentRequest,
    ChatTurn,
    Classification,
    MarketAnalysisAgent,
    StubAgent,
    ask,
    get_agent,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.market_analysis import (
    ALPHA_VANTAGE_SOURCE,
    MARKET_ANALYSIS_INSTRUCTIONS,
    MAX_TICKERS,
    NO_TICKER_TEXT,
    describe_comparison,
    describe_quote,
    fallback_explanation,
    format_quote,
    source_label,
    strip_numeric_sentences,
)

from ..market_data.conftest import SPY_GLOBAL_QUOTE, TEST_API_KEY, FakeClock, global_quote_for
from .test_ask import AllowAllReviewer, FakeClassifier
from .test_router import COMPLETIONS_URL
from .test_router import _completion as completion

SETTINGS = Settings(openai_api_key="sk-test", openai_model="gpt-4o-mini")
EXPLANATION = "The price sits a little below the previous close, and the day's range was narrow."


class FakeOpenAI:
    """respx side effect for both LLM calls: ticker extraction (structured) and the explanation."""

    def __init__(self, tickers: list[str], explanation: str | Callable[[str], str] = EXPLANATION) -> None:
        self.tickers = tickers
        self.explanation = explanation
        self.extractions: list[dict] = []
        self.explains: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            self.extractions.append(body)
            content = json.dumps({"tickers": self.tickers})
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
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


def mock_client() -> MarketDataClient:
    return MarketDataClient(Settings(market_data_mode="mock"))


def live_client(clock: FakeClock) -> MarketDataClient:
    budget = CallBudget(clock=clock, sleep=clock.sleep)
    return MarketDataClient(Settings(alpha_vantage_api_key=TEST_API_KEY), clock=clock, budget=budget)


def agent_with(client: MarketDataClient) -> MarketAnalysisAgent:
    return MarketAnalysisAgent(client_provider=lambda: client, settings=SETTINGS)


def av_route(router: respx.MockRouter):
    return router.get(ALPHA_VANTAGE_URL, params={"function": "GLOBAL_QUOTE"})


def av_by_symbol(responses: dict[str, dict]):
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.params["symbol"]])

    return respond


def mock_quote(symbol: str) -> Quote:
    quote = get_mock_quote(symbol, datetime(2026, 10, 9, 15, 0, tzinfo=UTC))
    assert quote is not None
    return quote


# --- Matrix: ticker quote (live) ------------------------------------------------


async def test_live_ticker_quote_shows_figures_then_explanation(router, clock: FakeClock) -> None:
    llm = FakeOpenAI(["AAPL"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    av = av_route(router).respond(json=global_quote_for("AAPL", "254.4300"))

    result = await agent_with(live_client(clock)).run(AgentRequest("What's the price of AAPL?"))

    assert av.call_count == 1
    figures, explanation = result.text.rsplit("\n\n", 1)
    assert explanation == EXPLANATION
    assert figures.splitlines() == [
        "**AAPL**",
        "- Price: $254.43",
        "- Change vs previous close ($765.61): -1.41 (-0.18%)",
        "- Open / high / low: $765.14 / $769.06 / $760.99",
        "- Volume: 61,234,567 shares",
        "- Trading day: 2026-09-29",
        "- Source: Live from Alpha Vantage, fetched 2026-09-30 14:00 UTC",
    ]
    assert result.sources == [ALPHA_VANTAGE_SOURCE]
    assert DISCLAIMER not in result.text
    # The LLM explains the Python-rendered figures; its system prompt is built with build_system_prompt.
    assert llm.explain_system.startswith(EDUCATION_SYSTEM_PROMPT)
    assert MARKET_ANALYSIS_INSTRUCTIONS.splitlines()[0] in llm.explain_system
    # The model gets the figures described in words, never the numbers themselves.
    assert "AAPL (live data from Alpha Vantage):" in llm.explain_prompt
    assert "below the previous close: a small move down" in llm.explain_prompt
    assert "254" not in llm.explain_prompt and "61,234,567" not in llm.explain_prompt
    assert "Question: What's the price of AAPL?" in llm.explain_prompt
    assert "personal recommendation" not in llm.explain_prompt
    # Extraction is structured output and sees the question.
    assert llm.extractions[0]["response_format"]["type"] == "json_schema"
    assert llm.extractions[0]["messages"][-1]["content"].endswith("Latest question: What's the price of AAPL?")


async def test_company_name_resolves_through_extraction(router) -> None:
    llm = FakeOpenAI(["aapl"])  # lower-case from the model is normalized
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(AgentRequest("What's Apple trading at?"))

    assert result.text.startswith("**AAPL** (demo data, not a real-time price)")
    assert "Apple" in llm.extractions[0]["messages"][-1]["content"]


async def test_extraction_sees_recent_history_for_follow_ups(router) -> None:
    llm = FakeOpenAI(["MSFT"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    history = (ChatTurn("user", "What's MSFT at?"), ChatTurn("assistant", "MSFT is ..."))

    await agent_with(mock_client()).run(AgentRequest("And what does its volume mean?", history))

    prompt = llm.extractions[0]["messages"][-1]["content"]
    assert "User: What's MSFT at?" in prompt
    assert prompt.endswith("Latest question: And what does its volume mean?")


# --- Matrix: quota exhausted -> mock ----------------------------------------------


async def test_quota_exhausted_serves_labelled_mock_quote(router, clock: FakeClock) -> None:
    llm = FakeOpenAI(["AAPL"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    av = av_route(router).respond(json=SPY_GLOBAL_QUOTE)
    client = live_client(clock)
    while client.budget.remaining_today:
        client.budget.try_acquire()
        clock.advance(61)

    result = await agent_with(client).run(AgentRequest("What's the price of AAPL?"))

    assert av.call_count == 0
    expected = mock_quote("AAPL")
    lines = result.text.splitlines()
    assert lines[0] == "**AAPL** (demo data, not a real-time price)"
    assert f"- Price: ${expected.price:,.2f}" in lines
    assert (
        "- Source: Demo data, not a real-time price: a bundled sample quote for trading day 2026-09-29" in lines
    )
    assert result.text.endswith(EXPLANATION)
    assert result.sources == []  # nothing came from Alpha Vantage
    assert "AAPL (demo data: a sample quote, not a current price):" in llm.explain_prompt


# --- Matrix: advice-seeking --------------------------------------------------------


@pytest.mark.parametrize("model_redirect", [ADVICE_REDIRECT, ADVICE_REDIRECT.replace("'", "\u2019")])
async def test_advice_seeking_opens_with_redirect_and_shows_figures(router, model_redirect: str) -> None:
    # The model adds the redirect itself as well (straight or curly apostrophe): it must appear once, at the top.
    llm = FakeOpenAI(["NVDA"], explanation=f"{model_redirect} {EXPLANATION}")
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(AgentRequest("Should I buy NVDA?", seeks_advice=True))

    assert result.text.startswith(f"{ADVICE_REDIRECT}\n\n**NVDA**")
    assert result.text.count(ADVICE_REDIRECT) == 1
    assert ADVICE_REDIRECT.replace("'", "\u2019") not in result.text
    assert result.text.endswith(EXPLANATION)
    assert "personal recommendation" in llm.explain_prompt


# --- Matrix: unknown symbol / nothing available ----------------------------------


async def test_unknown_symbol_says_no_quote_and_skips_the_llm_explanation(router, clock: FakeClock) -> None:
    llm = FakeOpenAI(["APPL"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    av_route(router).respond(json={"Global Quote": {}})

    result = await agent_with(live_client(clock)).run(AgentRequest("Quote for APPL"))

    assert result.text == (
        "**APPL**: I couldn't find a quote for APPL. Please check the ticker symbol and try again."
    )
    assert llm.explains == []
    assert result.sources == []


async def test_nothing_available_says_try_later_without_llm_explanation(router) -> None:
    llm = FakeOpenAI(["ZZZZ"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(AgentRequest("What's ZZZZ at?"))

    assert result.text == (
        "**ZZZZ**: A quote for ZZZZ isn't available right now. Please check the ticker symbol, or try again later."
    )
    assert llm.explains == []


async def test_invalid_extracted_symbol_is_reported_not_fetched(router) -> None:
    llm = FakeOpenAI(["NOT A TICKER!!"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(AgentRequest("Quote for not a ticker!!"))

    assert result.text == "**NOTATICKER**: That doesn't look like a valid ticker symbol. Please check it and try again."
    assert llm.explains == []


async def test_advice_seeking_error_reply_still_opens_with_redirect(router) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(["ZZZZ"]))
    result = await agent_with(mock_client()).run(AgentRequest("Should I buy ZZZZ?", seeks_advice=True))
    assert result.text.startswith(f"{ADVICE_REDIRECT}\n\n**ZZZZ**: A quote for ZZZZ isn't available")


# --- Matrix: no ticker -------------------------------------------------------------


async def test_no_ticker_asks_for_one_with_mock_examples_and_fetches_nothing(router) -> None:
    llm = FakeOpenAI([])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    fetched: list[str] = []

    class RecordingClient:
        async def get_quote(self, symbol: str) -> Quote:
            fetched.append(symbol)
            raise AssertionError("no quote fetch expected")

    agent = MarketAnalysisAgent(client_provider=lambda: RecordingClient(), settings=SETTINGS)  # type: ignore[arg-type,return-value]
    result = await agent.run(AgentRequest("How is the market doing?"))

    assert result.text == NO_TICKER_TEXT
    assert all(sym in NO_TICKER_TEXT for sym in mock_symbols())
    assert fetched == [] and llm.explains == []


async def test_no_ticker_for_advice_question_opens_with_redirect(router) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(["  "]))
    result = await agent_with(mock_client()).run(AgentRequest("What should I buy?", seeks_advice=True))
    assert result.text == f"{ADVICE_REDIRECT}\n\n{NO_TICKER_TEXT}"


# --- Matrix: several tickers ---------------------------------------------------------


async def test_two_tickers_in_question_order_then_one_comparison(router, clock: FakeClock) -> None:
    llm = FakeOpenAI(["VOO", "QQQ"], explanation="Both moved a little; QQQ's range was wider.")
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    av = av_route(router).mock(
        side_effect=av_by_symbol({"VOO": global_quote_for("VOO", "702.5500"), "QQQ": global_quote_for("QQQ", "601.1000")})
    )

    result = await agent_with(live_client(clock)).run(AgentRequest("Compare VOO and QQQ"))

    assert [c.request.url.params["symbol"] for c in av.calls] == ["VOO", "QQQ"]
    voo, qqq, explanation = result.text.split("\n\n")
    assert voo.startswith("**VOO**\n- Price: $702.55")
    assert qqq.startswith("**QQQ**\n- Price: $601.10")
    assert explanation == "Both moved a little; QQQ's range was wider."
    assert len(llm.explains) == 1
    assert "VOO (live data from Alpha Vantage):" in llm.explain_prompt
    assert "QQQ (live data from Alpha Vantage):" in llm.explain_prompt
    # Fetched one at a time, the second waiting 1 s for Alpha Vantage's burst limit.
    assert clock.sleeps == [pytest.approx(1.1)]


async def test_more_than_three_tickers_quotes_the_first_three_and_says_so(router) -> None:
    llm = FakeOpenAI(["SPY", "VOO", "VTI", "QQQ", "BND"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(AgentRequest("Quotes for SPY, VOO, VTI, QQQ and BND"))

    blocks = result.text.split("\n\n")
    assert blocks[0] == (
        f"I can quote up to {MAX_TICKERS} tickers at a time, so here are the first {MAX_TICKERS} "
        "(SPY, VOO, VTI). Ask again for QQQ, BND."
    )
    assert [b.splitlines()[0].split()[0] for b in blocks[1:4]] == ["**SPY**", "**VOO**", "**VTI**"]
    assert "**QQQ**" not in result.text and "**BND**" not in result.text
    assert blocks[4] == EXPLANATION


async def test_duplicate_tickers_are_quoted_once(router) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(["SPY", "spy", "VOO"]))
    result = await agent_with(mock_client()).run(AgentRequest("SPY vs spy vs VOO"))
    assert result.text.count("**SPY**") == 1 and result.text.count("**VOO**") == 1
    assert "up to" not in result.text


async def test_per_ticker_error_is_inline_and_others_still_answered(router, clock: FakeClock) -> None:
    llm = FakeOpenAI(["VOO", "APPL", "QQQ"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    av_route(router).mock(
        side_effect=av_by_symbol(
            {"VOO": global_quote_for("VOO"), "APPL": {"Global Quote": {}}, "QQQ": global_quote_for("QQQ")}
        )
    )

    result = await agent_with(live_client(clock)).run(AgentRequest("VOO, APPL and QQQ?"))

    blocks = result.text.split("\n\n")
    assert blocks[0].startswith("**VOO**\n")
    assert blocks[1] == "**APPL**: I couldn't find a quote for APPL. Please check the ticker symbol and try again."
    assert blocks[2].startswith("**QQQ**\n")
    assert blocks[3] == EXPLANATION
    assert "**APPL**" not in llm.explain_prompt  # only real figures are explained
    assert "No quote is available for: APPL." in llm.explain_prompt


async def test_dollar_prefixed_and_duplicate_symbols_are_cleaned_before_the_cap(router) -> None:
    llm = FakeOpenAI(["$AAPL", "AAPL", "$SPY", "SPY", " voo ", "$QQQ", "$BND"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)

    result = await agent_with(mock_client()).run(AgentRequest("$AAPL $SPY VOO $QQQ $BND"))

    blocks = result.text.split("\n\n")
    assert blocks[0] == (
        f"I can quote up to {MAX_TICKERS} tickers at a time, so here are the first {MAX_TICKERS} "
        "(AAPL, SPY, VOO). Ask again for QQQ, BND."
    )
    assert [b.split()[0] for b in blocks[1:4]] == ["**AAPL**", "**SPY**", "**VOO**"]
    assert "valid ticker" not in result.text and "$AAPL" not in result.text


async def test_live_and_mock_mix_cites_alpha_vantage_and_notes_different_days(router, clock: FakeClock) -> None:
    llm = FakeOpenAI(["SPY", "VOO"])
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    responses = {"SPY": SPY_GLOBAL_QUOTE, "VOO": {"Note": "rate limited"}}  # VOO falls back to mock
    live_day = dict(SPY_GLOBAL_QUOTE["Global Quote"], **{"07. latest trading day": "2026-09-30"})
    responses["SPY"] = {"Global Quote": live_day}
    av_route(router).mock(side_effect=av_by_symbol(responses))

    result = await agent_with(live_client(clock)).run(AgentRequest("SPY and VOO?"))

    assert "- Source: Live from Alpha Vantage" in result.text
    assert "**VOO** (demo data, not a real-time price)" in result.text
    assert result.sources == [ALPHA_VANTAGE_SOURCE]
    assert "different trading days" in llm.explain_prompt


# --- Source labels for all four QuoteSources -----------------------------------------


def test_source_labels_for_every_quote_source() -> None:
    base = dataclasses.replace(mock_quote("SPY"), fetched_at=datetime(2026, 10, 9, 13, 45, tzinfo=UTC))
    labels = {s: source_label(dataclasses.replace(base, source=s)) for s in ("live", "cache", "stale_cache", "mock")}
    assert labels["live"] == "Live from Alpha Vantage, fetched 2026-10-09 13:45 UTC"
    assert labels["cache"] == "Alpha Vantage, cached copy fetched 2026-10-09 13:45 UTC"
    assert labels["stale_cache"].startswith("Alpha Vantage, older cached copy fetched 2026-10-09 13:45 UTC")
    assert "may be out of date" in labels["stale_cache"]
    assert labels["mock"] == "Demo data, not a real-time price: a bundled sample quote for trading day 2026-09-29"
    assert "demo data" not in format_quote(dataclasses.replace(base, source="live")).lower()


async def test_cached_and_stale_quotes_are_labelled_through_the_agent(router, clock: FakeClock) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(["SPY"]))
    av = av_route(router).respond(json=SPY_GLOBAL_QUOTE)
    client = live_client(clock)
    agent = agent_with(client)
    await agent.run(AgentRequest("SPY?"))  # live, now cached

    clock.advance(60)
    cached = await agent.run(AgentRequest("SPY?"))
    assert "- Source: Alpha Vantage, cached copy fetched 2026-09-30 14:00 UTC" in cached.text

    clock.advance(31 * 60)
    client.budget.pause(3600)
    stale = await agent.run(AgentRequest("SPY?"))
    assert "- Source: Alpha Vantage, older cached copy fetched 2026-09-30 14:00 UTC" in stale.text
    assert stale.sources == [ALPHA_VANTAGE_SOURCE]
    assert av.call_count == 1


def test_figures_formatting_edge_cases() -> None:
    base = mock_quote("SPY")
    flat = dataclasses.replace(base, change=-0.001, change_percent=0.0)
    assert "(0.00%)" in format_quote(flat) and ": 0.00 (" in format_quote(flat)
    penny = dataclasses.replace(base, symbol="PENNY", price=0.1234)
    assert "- Price: $0.1234" in format_quote(penny)
    foreign = dataclasses.replace(base, symbol="TSCO.LON", price=312.5)
    assert "- Price: 312.50" in format_quote(foreign)
    assert "- Trading day: 2026-09-29" in format_quote(dataclasses.replace(base, latest_trading_day=date(2026, 9, 29)))


# --- The LLM never retypes a figure ---------------------------------------------------


def test_describe_quote_uses_words_only() -> None:
    nvda = mock_quote("NVDA")  # +1.25 %, price near the day's high, above the open
    text = describe_quote(nvda)
    assert text.splitlines() == [
        "NVDA (demo data: a sample quote, not a current price):",
        "- The latest price is above the previous close: a moderate move up for one trading day.",
        "- The day's range (high minus low) was moderate relative to the price.",
        "- The latest price sits near the day's high.",
        "- The latest price is above where the day opened.",
    ]
    flat = dataclasses.replace(nvda, change=0.0, change_percent=0.0, high=nvda.price, low=nvda.price, open=nvda.price)
    assert "unchanged from the previous close" in describe_quote(flat)
    # The figures show -0.001 as 0.00, so the description must not call it a move down.
    tiny = dataclasses.replace(nvda, change=-0.001, change_percent=-0.0005)
    assert "unchanged from the previous close" in describe_quote(tiny)
    assert "move down" not in describe_quote(tiny)
    for quote in (nvda, flat, mock_quote("SPY"), mock_quote("BND")):
        assert not any(ch.isdigit() for ch in describe_quote(quote))


def test_describe_comparison_names_leaders_and_skips_ties() -> None:
    voo, qqq = mock_quote("VOO"), mock_quote("QQQ")
    text = describe_comparison([voo, qqq])
    assert "QQQ moved the most relative to its previous close." in text
    assert "QQQ had the widest day's range relative to its price." in text
    assert "The most shares changed hands in QQQ" in text
    assert describe_comparison([voo]) == ""
    other_day = dataclasses.replace(qqq, latest_trading_day=date(2026, 9, 30))
    assert describe_comparison([voo, other_day]).splitlines()[1] == (
        "- These quotes are from different trading days, so they describe different days, not the same session."
    )
    assert describe_comparison([voo, dataclasses.replace(voo, symbol="VOO2")]) == ""  # all tied


def test_strip_numeric_sentences_keeps_words_lines_and_symbols() -> None:
    text = "VOO rose 1.2% today. It was a calm day! QQQ fell.\n- A list item.\n- Another 5 item.\n\nVolume was 5,432,100."
    cleaned, dropped = strip_numeric_sentences(text, ["VOO", "QQQ"])
    assert cleaned == "It was a calm day! QQQ fell.\n- A list item."
    assert dropped == 3
    assert strip_numeric_sentences("BRK2 moved up.", ["BRK2"]) == ("BRK2 moved up.", 0)


@pytest.mark.parametrize(
    "sentence",
    [
        "VOO tracks the S&P 500.",
        "QQQ tracks the Nasdaq-100.",
        "QQQ tracks the Nasdaq 100.",
        "Many people hold index funds in a 401(k).",
    ],
)
def test_strip_numeric_sentences_keeps_names_with_digits(sentence: str) -> None:
    assert strip_numeric_sentences(sentence, ["VOO", "QQQ"]) == (sentence, 0)


def test_strip_numeric_sentences_keeps_numbered_list_items() -> None:
    text = "1. The price is above the close.\n2) The range was narrow.\n3. It closed at 194.60."
    cleaned, dropped = strip_numeric_sentences(text, [])
    assert cleaned == "1. The price is above the close.\n2) The range was narrow."
    assert dropped == 1


async def test_explanation_that_restates_numbers_is_cleaned(router, caplog: pytest.LogCaptureFixture) -> None:
    router.post(COMPLETIONS_URL).mock(
        side_effect=FakeOpenAI(["NVDA"], explanation="NVDA closed at $194.60. That is above the previous close.")
    )
    result = await agent_with(mock_client()).run(AgentRequest("NVDA?"))
    assert result.text == f"{format_quote(mock_quote('NVDA'))}\n\nThat is above the previous close."
    assert "dropped 1 explanation sentence" in caplog.text


async def test_explanation_made_only_of_numbers_falls_back_to_the_descriptions(router) -> None:
    router.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(["NVDA"], explanation="NVDA: $194.60 (+1.25%)."))
    result = await agent_with(mock_client()).run(AgentRequest("NVDA?"))
    explanation = result.text.rsplit("\n\n", 1)[1]
    assert explanation == fallback_explanation([mock_quote("NVDA")])
    assert explanation.startswith("For NVDA, the latest price is above the previous close")
    assert explanation.endswith("The NVDA quote is a sample quote (demo data), not a current price.")
    live = dataclasses.replace(mock_quote("NVDA"), source="live")
    assert "sample quote" not in fallback_explanation([live])


# --- Matrix: explanation fails -------------------------------------------------------


@respx.mock
async def test_explanation_failure_gives_the_graphs_failure_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    config.reset_settings()
    install_real_agents()
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        if "response_format" in body:
            return httpx.Response(200, json=completion(json.dumps({"tickers": ["SPY"]}), body["model"]))
        return httpx.Response(500, json={"error": {"message": "boom"}})

    respx.post(COMPLETIONS_URL).mock(side_effect=respond)

    reply = await ask("What's the stock price of SPY?", classifier=FakeClassifier(), reviewer=AllowAllReviewer())

    assert reply.route == "market"
    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"
    assert reply.sources == []


@respx.mock
async def test_ticker_extraction_failure_gives_failure_reply_without_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", TEST_API_KEY)
    config.reset_settings()
    install_real_agents()
    av = respx.get(ALPHA_VANTAGE_URL).mock(return_value=httpx.Response(200, json=SPY_GLOBAL_QUOTE))
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))

    reply = await ask("What's the stock price of SPY?", classifier=FakeClassifier(), reviewer=AllowAllReviewer())

    assert reply.route == "market"
    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"
    assert av.call_count == 0


# --- Registration and ask() ----------------------------------------------------------


async def test_install_real_agents_registers_market_and_reset_restores_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    install_real_agents()
    assert isinstance(get_agent("market"), MarketAnalysisAgent)
    reset_agents()
    assert type(get_agent("market")) is StubAgent


class _MarketClassifier:
    def __init__(self, seeks_advice: bool = False) -> None:
        self.seeks_advice = seeks_advice

    async def classify(self, question, history):  # type: ignore[no-untyped-def]
        return Classification(route="market", seeks_advice=self.seeks_advice)


@respx.mock
async def test_ask_routes_to_market_with_review_and_one_disclaimer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    config.reset_settings()
    install_real_agents()
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(["VOO"]))
    reviewed: list[tuple[str, str]] = []

    class RecordingReviewer(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            reviewed.append((question, reply))
            return False

    reply = await ask("What's VOO trading at?", classifier=_MarketClassifier(), reviewer=RecordingReviewer())

    assert reply.route == "market"
    assert reply.text.startswith("**VOO** (demo data, not a real-time price)")
    assert f"- Price: ${mock_quote('VOO').price:,.2f}" in reply.text
    assert reply.text.count(DISCLAIMER) == 1 and reply.text.endswith(DISCLAIMER)
    assert len(reviewed) == 1 and reviewed[0][1].startswith("**VOO**")  # the review was applied


@respx.mock
async def test_ask_reviewer_flagging_advice_replaces_the_market_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    config.reset_settings()
    install_real_agents()
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(["NVDA"], explanation="You should buy NVDA."))

    class FlagAll(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            return True

    reply = await ask("Should I buy NVDA?", classifier=_MarketClassifier(seeks_advice=True), reviewer=FlagAll())

    assert reply.route == "market"
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert "**NVDA**" not in reply.text and "You should buy" not in reply.text
