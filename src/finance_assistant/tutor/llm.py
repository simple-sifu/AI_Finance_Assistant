"""Per-call OpenAI chat models that are safe across ``asyncio.run()`` calls.

langchain-openai caches its default ``httpx.AsyncClient`` process-wide, and that
client is bound to the event loop it first ran on. Streamlit calls
``asyncio.run`` per interaction, so we always hand ChatOpenAI a fresh async
client and close it when the call is done.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from langchain_openai import ChatOpenAI

from ..config import Settings

LLM_TIMEOUT_SECONDS = 30.0


@asynccontextmanager
async def chat_model(settings: Settings, **kwargs: object) -> AsyncIterator[ChatOpenAI]:
    """Yield a ChatOpenAI bound to a fresh HTTP client; raise ConfigurationError if no key."""
    api_key = settings.require_openai_api_key()
    options: dict[str, object] = {"timeout": LLM_TIMEOUT_SECONDS, "max_retries": 1, **kwargs}
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as http_client:
        yield ChatOpenAI(
            model=settings.openai_model,
            api_key=api_key,
            http_async_client=http_client,
            **options,
        )
