import pytest

from project.backend.app.core.config import Settings
from project.backend.app.core.llm import get_chat_model, validate_and_merge_llm_settings


def _settings() -> Settings:
    return Settings(
        LLM_PROVIDER="openrouter",
        EMBEDDING_PROVIDER="openrouter",
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_ALLOWED_MODELS="openai/gpt-oss-120b:free,anthropic/claude-3.5-sonnet",
    )


def _ollama_settings() -> Settings:
    return Settings(
        LLM_PROVIDER="ollama",
        OLLAMA_MODEL="gemma4:26b",
        OLLAMA_ALLOWED_MODELS="gemma4:26b,mxbai-embed-large",
    )


def _split_provider_settings() -> Settings:
    return Settings(
        LLM_PROVIDER="ollama",
        EMBEDDING_PROVIDER="openrouter",
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_EMBEDDING_MODEL="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        OLLAMA_MODEL="gemma4:26b",
        OLLAMA_ALLOWED_MODELS="gemma4:26b,mxbai-embed-large",
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


def test_llm_settings_accept_valid_ollama_override() -> None:
    merged = validate_and_merge_llm_settings(
        _ollama_settings(),
        {"model": "mxbai-embed-large", "temperature": 0.1, "top_p": 0.3},
    )
    assert merged["model"] == "mxbai-embed-large"
    assert merged["top_p"] == 0.3


def test_get_chat_model_returns_ollama_model() -> None:
    model = get_chat_model(_ollama_settings())
    assert model.__class__.__name__ == "ChatOllama"


def test_split_provider_uses_openrouter_embeddings_and_ollama_agent() -> None:
    settings = _split_provider_settings()
    assert settings.llm_provider == "ollama"
    assert settings.embedding_provider == "openrouter"
    assert settings.active_llm_model() == "gemma4:26b"
    assert settings.active_embedding_model() == "nvidia/llama-nemotron-embed-vl-1b-v2:free"
