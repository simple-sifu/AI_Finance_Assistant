"""OpenAIClassifier against a respx-mocked OpenAI API (no real network)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from finance_assistant.config import ConfigurationError, Settings
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    UNAVAILABLE_TEXT,
    ChatTurn,
    Classification,
    OpenAIClassifier,
    ScopedClassifier,
    ask,
)
from finance_assistant.tutor.llm import LLM_TIMEOUT_SECONDS, chat_model

COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


def _completion(content: str, model: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_requires_api_key() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIClassifier(Settings())


@respx.mock
def test_structured_output_parsed_across_event_loops() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        decision = json.dumps({"route": "tax_education", "seeks_advice": True})
        return httpx.Response(200, json=_completion(decision, body["model"]))

    respx.post(COMPLETIONS_URL).mock(side_effect=handler)
    classifier = OpenAIClassifier(Settings(openai_api_key="sk-test", openai_model="gpt-4o-mini"))
    history = [ChatTurn("user", "How do IRAs work?"), ChatTurn("assistant", "An IRA is...")]

    for _ in range(2):  # two event loops, like two Streamlit interactions
        reply = asyncio.run(ask("Should I open a Roth?", history, classifier=classifier))
        assert (reply.route, reply.seeks_advice) == ("tax_education", True)

    assert len(seen) == 2
    assert seen[0]["model"] == "gpt-4o-mini"
    assert seen[0]["response_format"]["type"] == "json_schema"
    user_prompt = seen[0]["messages"][-1]["content"]
    assert "How do IRAs work?" in user_prompt
    assert user_prompt.endswith("Latest question: Should I open a Roth?")


@respx.mock
def test_unparseable_output_routes_to_clarify() -> None:
    respx.post(COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_completion('{"route": "weather"}', "gpt-4o-mini"))
    )
    classifier = OpenAIClassifier(Settings(openai_api_key="sk-test"))
    reply = asyncio.run(ask("What is an ETF?", classifier=classifier))
    assert reply.route == "clarify"


@pytest.mark.parametrize("content", ["not json at all", '{"route": "market"}', '{"route": "unavailable", "seeks_advice": false}'])
@respx.mock
def test_unparseable_structured_output_routes_to_clarify(content: str) -> None:
    respx.post(COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_completion(content, "gpt-4o-mini"))
    )
    classifier = OpenAIClassifier(Settings(openai_api_key="sk-test"))
    reply = asyncio.run(ask("What is an ETF?", classifier=classifier))
    assert reply.route == "clarify"


@respx.mock
def test_long_history_turn_is_clipped_in_prompt() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        decision = json.dumps({"route": "finance_qa", "seeks_advice": False})
        return httpx.Response(200, json=_completion(decision, body["model"]))

    respx.post(COMPLETIONS_URL).mock(side_effect=handler)
    head, tail = "a" * 500, "TAIL-SHOULD-NOT-APPEAR"
    history = [ChatTurn("assistant", head + tail)]
    classifier = OpenAIClassifier(Settings(openai_api_key="sk-test"))
    asyncio.run(ask("What is an ETF?", history, classifier=classifier))

    prompt = seen[0]["messages"][-1]["content"]
    assert f"Assistant: {head}…" in prompt
    assert tail not in prompt


async def test_chat_model_caller_kwargs_override_defaults() -> None:
    async with chat_model(Settings(openai_api_key="sk-test"), timeout=5.0, max_retries=3) as model:
        assert model.request_timeout == 5.0
        assert model.max_retries == 3
    async with chat_model(Settings(openai_api_key="sk-test")) as model:
        assert model.request_timeout == LLM_TIMEOUT_SECONDS
        assert model.max_retries == 1


# --- ScopedClassifier (single-agent UI tabs) ----------------------------------


class _StaticClassifier:
    def __init__(self, result: object = None, exc: Exception | None = None) -> None:
        self.result, self.exc = result, exc
        self.calls: list[tuple[str, tuple[ChatTurn, ...]]] = []

    async def classify(self, question, history):  # type: ignore[no-untyped-def]
        self.calls.append((question, tuple(history)))
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.mark.parametrize("inner_route", ["finance_qa", "market", "clarify"])
@pytest.mark.parametrize("seeks_advice", [False, True])
async def test_scoped_classifier_replaces_route_and_keeps_advice_flag(inner_route: str, seeks_advice: bool) -> None:
    inner = _StaticClassifier(Classification(route=inner_route, seeks_advice=seeks_advice))
    scoped = ScopedClassifier("portfolio", inner)
    history = [ChatTurn("user", "hi")]
    result = await scoped.classify("Is my mix too risky?", history)
    assert result == Classification(route="portfolio", seeks_advice=seeks_advice)
    assert inner.calls == [("Is my mix too risky?", (ChatTurn("user", "hi"),))]


async def test_scoped_classifier_in_ask_reaches_its_agent_with_advice_redirect() -> None:
    scoped = ScopedClassifier("goal_planning", _StaticClassifier(Classification(route="market")))
    reply = await ask("What is AAPL trading at?", classifier=scoped)
    assert reply.route == "goal_planning"
    assert reply.seeks_advice is False
    # The keyword backstop still runs for scoped tabs.
    reply = await ask("Should I buy VOO?", classifier=scoped)
    assert (reply.route, reply.seeks_advice) == ("goal_planning", True)
    assert ADVICE_REDIRECT in reply.text


async def test_scoped_classifier_provider_failure_is_unavailable() -> None:
    scoped = ScopedClassifier("news", _StaticClassifier(exc=RuntimeError("outage")))
    reply = await ask("What happened today?", classifier=scoped)
    assert reply.route == "unavailable"
    assert UNAVAILABLE_TEXT in reply.text


@pytest.mark.parametrize("bad", [ValueError("bad"), object()])
async def test_scoped_classifier_bad_output_keeps_route(bad: object) -> None:
    inner = _StaticClassifier(exc=bad) if isinstance(bad, Exception) else _StaticClassifier(bad)
    reply = await ask("What is AAPL at?", classifier=ScopedClassifier("market", inner))
    assert (reply.route, reply.seeks_advice) == ("market", False)


@pytest.mark.parametrize("route", ["clarify", "unavailable", "weather"])
def test_scoped_classifier_rejects_non_agent_route(route: str) -> None:
    with pytest.raises(ValueError, match="route must be one of"):
        ScopedClassifier(route, _StaticClassifier())
