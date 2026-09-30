"""Shared test fixtures: isolate config from the developer's env/.env and forbid real network."""

from __future__ import annotations

import socket

import pytest

from finance_assistant import config
from finance_assistant.market_data import reset_client

_ENV_VARS = ("ALPHA_VANTAGE_API_KEY", "MARKET_DATA_MODE", "QUOTE_CACHE_TTL_SECONDS")


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch):
    """Ignore any local .env and env vars so tests behave like a fresh clone."""
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", None)
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    config.reset_settings()
    reset_client()
    yield
    config.reset_settings()
    reset_client()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch):
    """Fail loudly if anything tries to open a real network connection."""

    def guard(*_args, **_kwargs):
        raise RuntimeError("tests must not touch the real network")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)
