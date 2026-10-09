"""Live eval of the Market Analysis agent (opt in: ``uv run pytest -m live``, needs OPENAI_API_KEY).

Uses the real OpenAI model for ticker extraction and the explanation, and the
real advice reviewer. Quotes come from a mock-mode market-data client, so the
eval spends no Alpha Vantage quota and the figures are known; one end-to-end
case runs ``ask`` with ``MARKET_DATA_MODE=mock``. Deselected by default;
settings load inside a fixture, never at import.
"""

from __future__ import annotations

import dataclasses

import pytest

from finance_assistant import config
from finance_assistant.market_data import MarketDataClient, reset_client
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    DISCLAIMER,
    AgentRequest,
    MarketAnalysisAgent,
    OpenAIAdviceReviewer,
    ask,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.market_analysis import fallback_explanation
from finance_assistant.tutor.router import review_reply

pytestmark = pytest.mark.live

# (question, seeks_advice, tickers expected in order)
CASES: list[tuple[str, bool, list[str]]] = [
    ("What's the price of AAPL?", False, ["AAPL"]),
    ("What's Apple trading at?", False, ["AAPL"]),  # company name
    ("Compare VOO and QQQ", False, ["VOO", "QQQ"]),  # two tickers
    ("How did NVDA do on its last trading day?", False, ["NVDA"]),
    ("Should I buy NVDA?", True, ["NVDA"]),  # advice-seeking
]


@pytest.fixture(autouse=True)
def _no_network():
    """Override the global guard: this eval must reach the real API."""
    yield


@pytest.fixture(autouse=True)
def _isolated_config():
    """Override the global isolation so the real env/.env (and OPENAI_API_KEY) are used."""
    config.reset_settings()
    reset_client()
    reset_agents()
    yield
    config.reset_settings()
    reset_client()
    reset_agents()


@pytest.fixture
def live_settings() -> config.Settings:
    settings = config.load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    return settings


@pytest.mark.parametrize(("question", "seeks_advice", "tickers"), CASES, ids=[c[0] for c in CASES])
async def test_cap4_quote_and_explanation_pass_review(
    live_settings: config.Settings, question: str, seeks_advice: bool, tickers: list[str]
) -> None:
    client = MarketDataClient(dataclasses.replace(live_settings, market_data_mode="mock"))
    agent = MarketAnalysisAgent(client_provider=lambda: client, settings=live_settings)
    result = await agent.run(AgentRequest(question, seeks_advice=seeks_advice))
    print(f"\n--- {question}\n{result.text}")

    headings = [line.split()[0] for line in result.text.splitlines() if line.startswith("**")]
    assert headings == [f"**{t}**" for t in tickers]
    for ticker in tickers:
        quote = await client.get_quote(ticker)
        assert f"- Price: ${quote.price:,.2f}" in result.text
    assert "demo data, not a real-time price" in result.text
    explanation = result.text.rsplit("\n\n", 1)[1]
    assert not explanation.startswith("- Source:"), "no explanation after the figures"
    assert explanation != fallback_explanation([await client.get_quote(t) for t in tickers]), (
        "the model's explanation was withheld for restating numbers"
    )
    if seeks_advice:
        assert result.text.startswith(ADVICE_REDIRECT)
        assert result.text.count(ADVICE_REDIRECT) == 1
    else:
        assert ADVICE_REDIRECT not in result.text
    verdict = await review_reply(OpenAIAdviceReviewer(live_settings), question, result.text)
    assert verdict == "ok"


async def test_ask_voo_in_mock_mode_end_to_end(live_settings: config.Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", None)  # no .env key either
    monkeypatch.setenv("OPENAI_API_KEY", live_settings.openai_api_key or "")
    monkeypatch.setenv("OPENAI_MODEL", live_settings.openai_model)
    config.reset_settings()
    reset_client()
    install_real_agents()

    reply = await ask("What's VOO trading at?")
    print(f"\n{reply.text}")

    assert reply.route == "market"
    assert reply.text.startswith("**VOO** (demo data, not a real-time price)")
    assert "trading day 2026-09-29" in reply.text
    assert reply.text.count(DISCLAIMER) == 1
