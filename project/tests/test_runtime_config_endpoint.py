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
