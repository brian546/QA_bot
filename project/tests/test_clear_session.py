from fastapi.testclient import TestClient

from project.backend.app.main import create_app


def test_clear_session_idempotent() -> None:
    app = create_app()
    client = TestClient(app)

    # First call should be safe even if session does not yet exist.
    first = client.post("/clear-session", json={"session_id": "abc"})
    second = client.post("/clear-session", json={"session_id": "abc"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["session_id"] == "abc"
