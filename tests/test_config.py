"""Settings loading: env over .env, defaults, validation, key never in repr."""

from __future__ import annotations

from pathlib import Path

import pytest

from finance_assistant.config import (
    DEFAULT_OPENAI_MODEL,
    ConfigurationError,
    Settings,
    load_settings,
)


def test_defaults_without_env_or_file() -> None:
    settings = load_settings(environ={})
    assert settings == Settings(None, "live", 1800.0)
    assert settings.openai_api_key is None
    assert settings.openai_model == DEFAULT_OPENAI_MODEL == "gpt-4o-mini"


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


def test_repr_hides_openai_key() -> None:
    settings = Settings(openai_api_key="sk-SECRET456")
    assert "sk-SECRET456" not in repr(settings)
    assert "<set>" in repr(settings)


def test_openai_settings_from_env_file_and_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-file\nOPENAI_MODEL=file-model\n")
    from_file = load_settings(environ={}, env_file=env_file)
    assert from_file.openai_api_key == "sk-file"
    assert from_file.openai_model == "file-model"

    overridden = load_settings(
        environ={"OPENAI_MODEL": " env-model ", "OPENAI_API_KEY": "  "}, env_file=env_file
    )
    assert overridden.openai_model == "env-model"
    assert overridden.openai_api_key is None  # blank env value means "no key"


def test_blank_openai_model_falls_back_to_default() -> None:
    assert load_settings(environ={"OPENAI_MODEL": "  "}).openai_model == DEFAULT_OPENAI_MODEL


def test_require_openai_api_key() -> None:
    assert Settings(openai_api_key="sk-x").require_openai_api_key() == "sk-x"
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        Settings().require_openai_api_key()


def test_tavily_key_from_env_file_and_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TAVILY_API_KEY=tvly-file\n")
    assert load_settings(environ={}, env_file=env_file).tavily_api_key == "tvly-file"
    assert load_settings(environ={"TAVILY_API_KEY": " tvly-env "}, env_file=env_file).tavily_api_key == "tvly-env"
    assert load_settings(environ={"TAVILY_API_KEY": "  "}).tavily_api_key is None
    assert load_settings(environ={}).tavily_api_key is None


def test_repr_hides_tavily_key() -> None:
    settings = Settings(tavily_api_key="tvly-SECRET789")
    assert "tvly-SECRET789" not in repr(settings)
    assert "tvly-SECRET789" not in str(settings)
    assert "tavily_api_key='<set>'" in repr(settings)
    assert "tavily_api_key=None" in repr(Settings())


def test_app_password_from_env_and_hidden_in_repr() -> None:
    settings = load_settings(environ={"APP_PASSWORD": " hunter2 "})
    assert settings.app_password == "hunter2"
    assert "hunter2" not in repr(settings)
    assert "app_password='<set>'" in repr(settings)
    assert load_settings(environ={"APP_PASSWORD": "  "}).app_password is None
    assert load_settings(environ={}).app_password is None
