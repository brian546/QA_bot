import pytest

from project.backend.app.core.config import Settings
from project.backend.app.core.llm import validate_and_merge_llm_settings


def _settings() -> Settings:
    return Settings(
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_ALLOWED_MODELS="openai/gpt-oss-120b:free,anthropic/claude-3.5-sonnet",
    )


def test_llm_settings_reject_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        validate_and_merge_llm_settings(_settings(), {"temperature": 99})


def test_llm_settings_accept_valid_override() -> None:
    merged = validate_and_merge_llm_settings(
        _settings(),
        {"model": "anthropic/claude-3.5-sonnet", "temperature": 0.1, "top_p": 0.3},
    )
    assert merged["model"] == "anthropic/claude-3.5-sonnet"
    assert merged["temperature"] == 0.1
