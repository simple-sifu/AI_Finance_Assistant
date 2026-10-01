"""The tutor graph end to end with a fake classifier: every I/O-matrix row, offline."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Sequence

import httpx
import pytest
import respx

from finance_assistant.config import ConfigurationError
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    UNAVAILABLE_TEXT,
    DISCLAIMER,
    AgentRequest,
    AgentResult,
    ChatTurn,
    Classification,
    Source,
    TutorReply,
    apply_guardrail,
    ask,
    get_agent,
    register_agent,
)
from finance_assistant.tutor.router import MAX_HISTORY_TURNS

from .test_router import COMPLETIONS_URL
from .test_router import _completion as completion


class FakeClassifier:
    """Routes by keyword (looking at history for follow-ups) and records what it saw."""

    KEYWORDS: tuple[tuple[str, str], ...] = (
        ("roth", "tax_education"),
        ("ira", "tax_education"),
        ("401", "tax_education"),
        ("portfolio", "portfolio"),
        ("holdings", "portfolio"),
        ("stock", "market"),
        ("price", "market"),
        ("save", "goal_planning"),
        ("news", "news"),
        ("compound", "finance_qa"),
        ("etf", "finance_qa"),
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ChatTurn, ...]]] = []

    async def classify(self, question: str, history: Sequence[ChatTurn]) -> Classification:
        self.calls.append((question, tuple(history)))
        q = question.lower()
        seeks_advice = "should i" in q
        for text in [q] + [t.content.lower() for t in reversed(history)]:
            for word, route in self.KEYWORDS:
                if word in text:
                    return Classification(route=route, seeks_advice=seeks_advice)
        return Classification(route="clarify", seeks_advice=seeks_advice)


class FuncClassifier:
    def __init__(self, fn: Callable[[str, Sequence[ChatTurn]], object]) -> None:
        self.fn = fn

    async def classify(self, question, history):  # type: ignore[no-untyped-def]
        return self.fn(question, history)


def assert_disclaimer(reply: TutorReply) -> None:
    assert reply.text.endswith("\n\n" + DISCLAIMER)
    assert reply.text.count(DISCLAIMER) == 1


async def test_clear_topic_routes_to_finance_qa() -> None:
    reply = await ask("What is compound interest?", classifier=FakeClassifier())
    assert reply.route == "finance_qa"
    assert reply.seeks_advice is False
    assert reply.sources == []
    assert "Finance Q&A" in reply.text
    assert_disclaimer(reply)


@pytest.mark.parametrize(
    ("question", "route", "agent_name"),
    [
        ("What is the difference between an ETF and a mutual fund?", "finance_qa", "Finance Q&A"),
        ("How diversified is my portfolio?", "portfolio", "Portfolio Analysis"),
        ("What does the stock price of AAPL mean?", "market", "Market Analysis"),
        ("How much do I need to save each month for $20k in 3 years?", "goal_planning", "Goal Planning"),
        ("Summarize today's market news", "news", "News Synthesizer"),
        ("How is a 401(k) different from an IRA?", "tax_education", "Tax Education"),
    ],
)
async def test_each_agent_route(question: str, route: str, agent_name: str) -> None:
    reply = await ask(question, classifier=FakeClassifier())
    assert reply.route == route
    assert agent_name in reply.text
    assert "placeholder" in reply.text
    assert_disclaimer(reply)


async def test_follow_up_uses_history() -> None:
    classifier = FakeClassifier()
    history = [
        ChatTurn("user", "How does a traditional IRA work?"),
        ChatTurn("assistant", "A traditional IRA lets you contribute pre-tax money..."),
    ]
    reply = await ask("what about Roth?", history, classifier=classifier)
    assert reply.route == "tax_education"
    assert classifier.calls == [("what about Roth?", tuple(history))]


async def test_follow_up_without_keyword_resolved_from_history() -> None:
    history = [ChatTurn("user", "Explain IRAs"), ChatTurn("assistant", "Sure...")]
    reply = await ask("and the other kind?", history, classifier=FakeClassifier())
    assert reply.route == "tax_education"


async def test_router_sees_at_most_six_history_turns() -> None:
    classifier = FakeClassifier()
    history = [ChatTurn("user" if i % 2 == 0 else "assistant", f"turn {i}") for i in range(10)]
    await ask("hi", history, classifier=classifier)
    (_, seen), = classifier.calls
    assert MAX_HISTORY_TURNS == 6
    assert seen == tuple(history[-6:])


async def test_advice_seeking_gets_redirect_not_recommendation() -> None:
    reply = await ask("What stock should I buy?", classifier=FakeClassifier())
    assert reply.seeks_advice is True
    assert reply.route == "market"
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert reply.text == apply_guardrail(ADVICE_REDIRECT)
    assert_disclaimer(reply)


async def test_advice_seeking_on_clarify_also_redirects() -> None:
    reply = await ask("What should I do?", classifier=FakeClassifier())
    assert reply.route == "clarify"
    assert reply.seeks_advice is True
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert_disclaimer(reply)


@pytest.mark.parametrize("question", ["hi", "what's the weather?"])
async def test_ambiguous_or_off_topic_clarifies(question: str) -> None:
    reply = await ask(question, classifier=FakeClassifier())
    assert reply.route == "clarify"
    assert reply.seeks_advice is False
    for topic in ("Investing concepts", "portfolio", "Markets", "Goals", "News", "401(k)"):
        assert topic in reply.text
    assert_disclaimer(reply)


def _raise(_q, _h):  # type: ignore[no-untyped-def]
    raise RuntimeError("LLM exploded")


@pytest.mark.parametrize(
    "fn",
    [
        lambda q, h: Classification(route="weather", seeks_advice=False),  # type: ignore[arg-type]
        lambda q, h: {"route": "market"},
        lambda q, h: None,
        lambda q, h: Classification(route="market", seeks_advice="yes"),  # type: ignore[arg-type]
        lambda q, h: Classification(route="unavailable", seeks_advice=False),
    ],
    ids=["invalid-route", "wrong-type", "none", "bad-flag", "unavailable-not-choosable"],
)
async def test_classifier_bad_output_routes_to_clarify(fn, caplog: pytest.LogCaptureFixture) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level(logging.WARNING, logger="finance_assistant.tutor.router"):
        reply = await ask("What is an ETF?", classifier=FuncClassifier(fn))
    assert reply.route == "clarify"
    assert_disclaimer(reply)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_classifier_exception_routes_to_unavailable(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="finance_assistant.tutor.router"):
        reply = await ask("What should I buy?", classifier=FuncClassifier(_raise))
    assert reply.route == "unavailable"
    assert reply.seeks_advice is False
    assert reply.text == apply_guardrail(UNAVAILABLE_TEXT)
    assert "LLM exploded" not in caplog.text  # only the exception type is logged
    assert "RuntimeError" in caplog.text


async def test_invalid_route_keeps_advice_flag_and_redirects() -> None:
    classifier = FuncClassifier(lambda q, h: Classification(route="weather", seeks_advice=True))  # type: ignore[arg-type]
    reply = await ask("What should I buy?", classifier=classifier)
    assert reply.route == "clarify"
    assert reply.seeks_advice is True
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert_disclaimer(reply)


async def test_history_must_be_chat_turns() -> None:
    classifier = FakeClassifier()
    with pytest.raises(ValueError):
        await ask("q", [{"role": "user", "content": "x"}], classifier=classifier)  # type: ignore[list-item]
    assert classifier.calls == []


def test_chat_turn_rejects_unknown_role() -> None:
    with pytest.raises(ValueError):
        ChatTurn("bot", "x")  # type: ignore[arg-type]


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
async def test_empty_input_raises_before_llm(question: str) -> None:
    classifier = FakeClassifier()
    with pytest.raises(ValueError):
        await ask(question, classifier=classifier)
    assert classifier.calls == []


async def test_empty_input_checked_before_api_key() -> None:
    with pytest.raises(ValueError):
        await ask("  ")  # no key, no classifier: still a ValueError, not a config error


async def test_missing_api_key_is_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        await ask("What is compound interest?")


@respx.mock
def test_key_never_logged_or_in_error(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a key set and OpenAI returning an auth error, the key appears nowhere."""
    secret = "sk-TEST-DO-NOT-LOG-999"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    route = respx.post(COMPLETIONS_URL).mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": f"Incorrect API key provided: {secret}", "type": "invalid_request_error",
                            "code": "invalid_api_key"}},
        )
    )

    with caplog.at_level(logging.DEBUG):
        reply = asyncio.run(ask("What is an ETF?"))
    assert route.called
    assert reply.route == "unavailable"
    assert reply.text == apply_guardrail(UNAVAILABLE_TEXT)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("OPENAI_API_KEY" in r.getMessage() for r in errors)
    assert secret not in caplog.text
    assert secret not in reply.text


@pytest.mark.parametrize(
    "mock",
    [
        {"return_value": httpx.Response(429, json={"error": {"message": "rate limited"}})},
        {"return_value": httpx.Response(500, json={"error": {"message": "boom"}})},
        {"side_effect": httpx.ReadTimeout("timed out")},
    ],
    ids=["429", "500", "timeout"],
)
@respx.mock
def test_provider_errors_route_to_unavailable(mock: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    route = respx.post(COMPLETIONS_URL).mock(**mock)
    reply = asyncio.run(ask("What is an ETF?"))
    assert route.called
    assert reply.route == "unavailable"
    assert reply.seeks_advice is False
    assert reply.text == apply_guardrail(UNAVAILABLE_TEXT)
    assert_disclaimer(reply)


def test_consecutive_asyncio_run_calls() -> None:
    classifier = FakeClassifier()
    first = asyncio.run(ask("What is compound interest?", classifier=classifier))
    second = asyncio.run(ask("What about ETFs?", classifier=classifier))
    assert (first.route, second.route) == ("finance_qa", "finance_qa")


@respx.mock
def test_consecutive_asyncio_run_calls_with_real_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default OpenAI path builds its HTTP client per call, so a second event loop works too."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    decision = json.dumps({"route": "finance_qa", "seeks_advice": False})
    route = respx.post(COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=completion(decision, "gpt-4o-mini"))
    )
    for _ in range(2):
        assert asyncio.run(ask("What is an ETF?")).route == "finance_qa"
    assert route.call_count == 2


async def test_registered_agent_replaces_stub_without_graph_edits() -> None:
    class RealMarketAgent:
        def __init__(self) -> None:
            self.requests: list[AgentRequest] = []

        async def run(self, request: AgentRequest) -> AgentResult:
            self.requests.append(request)
            return AgentResult(
                text="SPY closed at 764.20.",
                sources=[Source("Alpha Vantage", "https://www.alphavantage.co")],
            )

    agent = RealMarketAgent()
    register_agent("market", agent)
    assert get_agent("market") is agent

    history = [ChatTurn("user", "hello")]
    reply = await ask("What is the price of SPY?", history, classifier=FakeClassifier())
    assert reply.route == "market"
    assert reply.text == apply_guardrail("SPY closed at 764.20.")
    assert reply.sources == [Source("Alpha Vantage", "https://www.alphavantage.co")]
    assert agent.requests == [AgentRequest("What is the price of SPY?", tuple(history), False)]


class _RaisingAgent:
    async def run(self, request: AgentRequest) -> AgentResult:
        raise RuntimeError("agent crashed with secret detail")


class _WrongTypeAgent:
    async def run(self, request: AgentRequest) -> AgentResult:
        return "not an AgentResult"  # type: ignore[return-value]


@pytest.mark.parametrize("agent", [_RaisingAgent(), _WrongTypeAgent()], ids=["raises", "wrong-type"])
async def test_agent_failure_gets_failure_reply(agent, caplog: pytest.LogCaptureFixture) -> None:  # type: ignore[no-untyped-def]
    register_agent("market", agent)
    with caplog.at_level(logging.WARNING, logger="finance_assistant.tutor.graph"):
        reply = await ask("What is the price of SPY?", classifier=FakeClassifier())
    assert reply.route == "market"
    assert reply.sources == []
    assert reply.text == apply_guardrail(AGENT_FAILURE_TEXT)
    assert_disclaimer(reply)
    assert "market" in caplog.text
    assert "secret detail" not in caplog.text


def test_stubs_restored_between_tests() -> None:
    assert type(get_agent("market")).__name__ == "StubAgent"


def test_register_agent_rejects_bad_input() -> None:
    class NoRun:
        pass

    with pytest.raises(ValueError):
        register_agent("clarify", get_agent("market"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        register_agent("market", NoRun())  # type: ignore[arg-type]


def test_register_agent_rejects_sync_run() -> None:
    class SyncAgent:
        def run(self, request: AgentRequest) -> AgentResult:
            return AgentResult(text="sync")

    with pytest.raises(TypeError):
        register_agent("market", SyncAgent())  # type: ignore[arg-type]
    assert type(get_agent("market")).__name__ == "StubAgent"
