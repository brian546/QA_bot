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


def test_upload_reports_oversized_reason() -> None:
    app = create_app()
    client = TestClient(app)

    oversized = b"0" * (20_000_001)
    response = client.post(
        "/upload",
        data={"session_id": "upload-too-large"},
        files={"files": ("SystemDesignInterview.pdf", oversized, "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted_files"] == []
    assert payload["skipped_files"] == ["SystemDesignInterview.pdf"]
    assert payload["skipped_details"] == [
        {
            "filename": "SystemDesignInterview.pdf",
            "reason": "File exceeds the 20 MB upload limit.",
        }
    ]
