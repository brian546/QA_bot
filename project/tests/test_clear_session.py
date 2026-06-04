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


def test_list_and_get_sessions() -> None:
    app = create_app()
    store = app.state.session_store

    first = store.get_or_create("session-a")
    first.uploaded_documents.append({"filename": "alpha.pdf"})
    first.chat_history.extend(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    first.processed_files.add("alpha.pdf")
    first.llm_settings = {"model": "openai/gpt-oss-120b:free", "temperature": 0.1}

    second = store.get_or_create("session-b")
    second.uploaded_documents.append({"filename": "beta.pdf"})

    client = TestClient(app)

    listed = client.get("/sessions")
    assert listed.status_code == 200
    assert listed.json() == {
        "sessions": [
            {
                "session_id": "session-b",
                "uploaded_document_count": 1,
                "chat_message_count": 0,
            },
            {
                "session_id": "session-a",
                "uploaded_document_count": 1,
                "chat_message_count": 2,
            },
        ]
    }

    detail = client.get("/sessions/session-a")
    assert detail.status_code == 200
    assert detail.json() == {
        "session_id": "session-a",
        "uploaded_documents": [{"filename": "alpha.pdf"}],
        "chat_history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "processed_files": ["alpha.pdf"],
        "llm_settings": {"model": "openai/gpt-oss-120b:free", "temperature": 0.1},
    }
