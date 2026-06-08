from project.backend.app.core.config import get_settings
from fastapi.testclient import TestClient

from project.backend.app.main import create_app


def test_runtime_config_endpoint_hides_secrets() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/config")
    assert response.status_code == 200
    payload = response.json()
    assert "available_models" in payload
    assert "default_llm_settings" in payload
    assert "default_citations_k" in payload
    assert "citations_k" in payload.get("parameter_constraints", {})
    assert "openrouter_api_key" not in payload


def test_runtime_config_endpoint_supports_ollama_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
    monkeypatch.setenv("OLLAMA_ALLOWED_MODELS", "gemma4:26b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    get_settings.cache_clear()

    app = create_app()
    client = TestClient(app)

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_name"] == "Ollama"
    assert payload["default_model"] == "gemma4:26b"
    assert payload["default_llm_settings"]["model"] == "gemma4:26b"
    assert payload["available_models"] == ["gemma4:26b"]


def test_runtime_config_endpoint_supports_split_providers(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
    monkeypatch.setenv("OLLAMA_ALLOWED_MODELS", "gemma4:26b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
    get_settings.cache_clear()

    app = create_app()
    client = TestClient(app)

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent_provider_name"] == "Ollama"
    assert payload["provider_name"] == "Ollama"
    assert payload["embedding_provider_name"] == "OpenRouter"
    assert payload["embedding_provider"] == "openrouter"
    assert payload["default_model"] == "gemma4:26b"
    assert payload["default_embedding_model"] == "nvidia/llama-nemotron-embed-vl-1b-v2:free"
