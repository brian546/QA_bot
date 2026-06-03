from fastapi.testclient import TestClient

from project.backend.app.main import create_app


def test_api_smoke_health_and_ask() -> None:
    app = create_app()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    ask = client.post(
        "/ask",
        json={
            "session_id": "s-smoke",
            "question": "What is this about?",
            "chat_history": [],
            "llm_settings": {
                "model": "openai/gpt-oss-120b:free",
                "temperature": 0.0,
                "top_p": 0.2,
                "max_tokens": 500,
            },
        },
    )
    assert ask.status_code == 200
    payload = ask.json()
    assert "answer" in payload
    assert "effective_llm_settings" in payload
