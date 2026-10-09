"""Live eval of the Portfolio Analysis agent (opt in: ``uv run pytest -m live``, needs OPENAI_API_KEY).

Uses the real OpenAI model for holdings extraction and the explanation, and the
real advice reviewer. Quotes come from a mock-mode market-data client, so the
eval spends no Alpha Vantage quota and the figures are known; one end-to-end
case runs ``ask`` with ``MARKET_DATA_MODE=mock``. Deselected by default;
settings load inside a fixture, never at import.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from pathlib import Path

import pytest

from finance_assistant import config
from finance_assistant.market_data import MarketDataClient, reset_client
from finance_assistant.portfolio import Holding
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    DISCLAIMER,
    AgentRequest,
    OpenAIAdviceReviewer,
    Portfolio,
    PortfolioAnalysisAgent,
    ask,
    install_real_agents,
    parse_holdings_csv,
    reset_agents,
)
from finance_assistant.tutor.router import review_reply

pytestmark = pytest.mark.live

SAMPLE = Path(__file__).parent.parent / "data" / "sample_holdings.csv"


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


def sample() -> Portfolio:
    return parse_holdings_csv(SAMPLE.read_bytes())


# (id, question, seeks_advice, uploaded portfolio, tickers expected in the table, level, flagged ticker)
CASES = [
    ("sample-upload", "How diversified is my portfolio?", False, sample(), {"VOO", "AAPL", "BND", "VXUS", "MSFT", "NVDA"}, "low", "AAPL"),
    ("broad-fund-only", "How diversified is my portfolio?", False, Portfolio((Holding("VOO", shares=Decimal(20)),)), {"VOO"}, "high", None),
    ("typed", "I have 10 AAPL, 5 MSFT and $8,000 in VTI. How diversified is that?", False, None, {"AAPL", "MSFT", "VTI"}, "low", "MSFT"),
    ("advice", "Should I sell some AAPL?", True, sample(), {"VOO", "AAPL", "BND", "VXUS", "MSFT", "NVDA"}, "low", "AAPL"),
]


@pytest.mark.parametrize(
    ("question", "seeks_advice", "portfolio", "tickers", "level", "flagged"),
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
async def test_cap3_analysis_and_explanation_pass_review(
    live_settings: config.Settings, question, seeks_advice, portfolio, tickers, level, flagged
) -> None:
    client = MarketDataClient(dataclasses.replace(live_settings, market_data_mode="mock"))
    agent = PortfolioAnalysisAgent(client_provider=lambda: client, settings=live_settings)
    result = await agent.run(AgentRequest(question, seeks_advice=seeks_advice, portfolio=portfolio))
    print(f"\n--- {question}\n{result.text}")

    rows = {line.split("|")[1].strip() for line in result.text.splitlines() if line.startswith("| ") and "Ticker" not in line}
    assert rows == tickers
    assert f"- Diversification level: **{level}**" in result.text
    if flagged:
        assert f"- Concentration risk: {flagged} is a single company" in result.text
    else:
        assert "- Concentration risk: none found." in result.text
    explanation = result.text.rsplit("\n\n", 1)[1]
    assert not explanation.startswith(("Note:", "- ")), "no explanation after the analysis"
    assert not explanation.startswith("Diversification means spreading money"), (
        "the model's explanation was withheld for restating numbers"
    )
    if seeks_advice:
        assert result.text.startswith(ADVICE_REDIRECT)
        assert result.text.count(ADVICE_REDIRECT) == 1
    else:
        assert ADVICE_REDIRECT not in result.text
    verdict = await review_reply(OpenAIAdviceReviewer(live_settings), question, result.text)
    assert verdict == "ok"


async def test_ask_sample_upload_in_mock_mode_end_to_end(
    live_settings: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARKET_DATA_MODE", "mock")
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", None)  # no .env key either
    monkeypatch.setenv("OPENAI_API_KEY", live_settings.openai_api_key or "")
    monkeypatch.setenv("OPENAI_MODEL", live_settings.openai_model)
    config.reset_settings()
    reset_client()
    install_real_agents()

    reply = await ask("How diversified is my portfolio?", portfolio=parse_holdings_csv(SAMPLE.read_bytes()))
    print(f"\n{reply.text}")

    assert reply.route == "portfolio"
    assert "- Concentration risk: AAPL is a single company" in reply.text
    assert "- Diversification level: **low**" in reply.text
    assert reply.text.count(DISCLAIMER) == 1
