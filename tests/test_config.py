"""Settings loading: env over .env, defaults, validation, key never in repr."""

from __future__ import annotations

from pathlib import Path

import pytest

from finance_assistant.config import Settings, load_settings


def test_defaults_without_env_or_file() -> None:
    settings = load_settings(environ={})
    assert settings == Settings(None, "live", 1800.0)


def test_env_file_values_and_env_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPHA_VANTAGE_API_KEY=filekey\nMARKET_DATA_MODE=mock\nQUOTE_CACHE_TTL_SECONDS=60\n"
    )
    from_file = load_settings(environ={}, env_file=env_file)
    assert from_file.alpha_vantage_api_key == "filekey"
    assert from_file.market_data_mode == "mock"
    assert from_file.quote_cache_ttl_seconds == 60

    overridden = load_settings(environ={"MARKET_DATA_MODE": "LIVE"}, env_file=env_file)
    assert overridden.market_data_mode == "live"
    assert overridden.alpha_vantage_api_key == "filekey"


def test_blank_key_means_no_key() -> None:
    assert load_settings(environ={"ALPHA_VANTAGE_API_KEY": "  "}).alpha_vantage_api_key is None


@pytest.mark.parametrize(
    "env", [{"MARKET_DATA_MODE": "turbo"}, {"QUOTE_CACHE_TTL_SECONDS": "abc"},
            {"QUOTE_CACHE_TTL_SECONDS": "0"},
            {"QUOTE_CACHE_TTL_SECONDS": "nan"}, {"QUOTE_CACHE_TTL_SECONDS": "inf"}]
)
def test_invalid_values_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        load_settings(environ=env)


def test_repr_hides_key() -> None:
    settings = Settings(alpha_vantage_api_key="SECRET123")
    assert "SECRET123" not in repr(settings)
    assert "SECRET123" not in str(settings)
