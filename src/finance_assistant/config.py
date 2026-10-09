"""Application settings, loaded once from the environment and an optional .env file.

This is the single configuration source for the whole app. Real environment
variables win over values in .env, and loading never mutates ``os.environ``.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

MARKET_DATA_MODES = frozenset({"live", "mock"})
DEFAULT_QUOTE_CACHE_TTL_SECONDS = 1800.0
# Cheap model available to the course OpenAI project (gpt-6-luna is not; checked 2026-10-01). Override with OPENAI_MODEL.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Resolved relative to the working directory, like most .env tooling.
DEFAULT_ENV_FILE: Path | None = Path(".env")


class ConfigurationError(RuntimeError):
    """A required setting is missing or unusable. The message names the variable."""


@dataclass(frozen=True)
class Settings:
    """Immutable app configuration."""

    alpha_vantage_api_key: str | None = None
    market_data_mode: str = "live"
    quote_cache_ttl_seconds: float = DEFAULT_QUOTE_CACHE_TTL_SECONDS
    openai_api_key: str | None = None
    openai_model: str = DEFAULT_OPENAI_MODEL
    tavily_api_key: str | None = None
    # Shared password for the deployed app; None means no password screen (local dev).
    app_password: str | None = None

    def __post_init__(self) -> None:
        if self.market_data_mode not in MARKET_DATA_MODES:
            raise ValueError(
                f"MARKET_DATA_MODE must be one of {sorted(MARKET_DATA_MODES)}, "
                f"got {self.market_data_mode!r}"
            )
        if not math.isfinite(self.quote_cache_ttl_seconds) or self.quote_cache_ttl_seconds <= 0:
            raise ValueError("QUOTE_CACHE_TTL_SECONDS must be a positive finite number")
        if not self.openai_model or not self.openai_model.strip():
            raise ValueError("OPENAI_MODEL must not be empty")

    def require_openai_api_key(self) -> str:
        """Return the OpenAI key, or raise ConfigurationError naming OPENAI_API_KEY."""
        if not self.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env "
                "(see .env.example)."
            )
        return self.openai_api_key

    def __repr__(self) -> str:  # never expose a key in logs or tracebacks
        av_key = "<set>" if self.alpha_vantage_api_key else None
        openai_key = "<set>" if self.openai_api_key else None
        tavily_key = "<set>" if self.tavily_api_key else None
        app_password = "<set>" if self.app_password else None
        return (
            f"Settings(alpha_vantage_api_key={av_key!r}, "
            f"market_data_mode={self.market_data_mode!r}, "
            f"quote_cache_ttl_seconds={self.quote_cache_ttl_seconds!r}, "
            f"openai_api_key={openai_key!r}, "
            f"openai_model={self.openai_model!r}, "
            f"tavily_api_key={tavily_key!r}, "
            f"app_password={app_password!r})"
        )


def load_settings(
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> Settings:
    """Build Settings from ``environ`` (default ``os.environ``) layered over ``env_file``.

    ``env_file`` defaults to ``DEFAULT_ENV_FILE``; a missing file is ignored.
    """
    env_path = DEFAULT_ENV_FILE if env_file is None else Path(env_file)
    values: dict[str, str | None] = {}
    if env_path is not None and env_path.is_file():
        values.update(dotenv_values(env_path))
    values.update(os.environ if environ is None else environ)

    key = (values.get("ALPHA_VANTAGE_API_KEY") or "").strip() or None
    mode = (values.get("MARKET_DATA_MODE") or "live").strip().lower() or "live"
    ttl_raw = (values.get("QUOTE_CACHE_TTL_SECONDS") or "").strip()
    try:
        ttl = float(ttl_raw) if ttl_raw else DEFAULT_QUOTE_CACHE_TTL_SECONDS
    except ValueError as exc:
        raise ValueError(f"QUOTE_CACHE_TTL_SECONDS must be a number, got {ttl_raw!r}") from exc
    openai_key = (values.get("OPENAI_API_KEY") or "").strip() or None
    openai_model = (values.get("OPENAI_MODEL") or "").strip() or DEFAULT_OPENAI_MODEL
    tavily_key = (values.get("TAVILY_API_KEY") or "").strip() or None
    app_password = (values.get("APP_PASSWORD") or "").strip() or None
    return Settings(
        alpha_vantage_api_key=key,
        market_data_mode=mode,
        quote_cache_ttl_seconds=ttl,
        openai_api_key=openai_key,
        openai_model=openai_model,
        tavily_api_key=tavily_key,
        app_password=app_password,
    )


_settings: Settings | None = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """Return the process-wide Settings, loading them on first use."""
    global _settings
    with _settings_lock:
        if _settings is None:
            _settings = load_settings()
        return _settings


def reset_settings() -> None:
    """Forget the cached Settings so the next ``get_settings()`` reloads (for tests)."""
    global _settings
    with _settings_lock:
        _settings = None
