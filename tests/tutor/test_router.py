"""OpenAIClassifier against a respx-mocked OpenAI API (no real network)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from finance_assistant.config import ConfigurationError, Settings
from finance_assistant.tutor import ChatTurn, OpenAIClassifier, ask
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
