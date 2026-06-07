from fastapi.testclient import TestClient

from project.backend.app.main import create_app
from project.backend.app.services.lexical_retrieval import build_bm25_index
from project.backend.app.services.semantic_retrieval import build_faiss_index


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
            },
        },
    )
    assert ask.status_code == 200
    payload = ask.json()
    assert "answer" in payload
    assert "effective_llm_settings" in payload

def test_remove_files_updates_session_state_and_indexes() -> None:
    app = create_app()
    client = TestClient(app)

    session = app.state.session_store.get_or_create("remove-sync")
    session.processed_files = {"alpha.pdf", "beta.pdf"}
    session.uploaded_documents = [
        {"filename": "Alpha.pdf", "normalized_key": "alpha.pdf", "page_count": 1, "chunk_count": 1},
        {"filename": "Beta.pdf", "normalized_key": "beta.pdf", "page_count": 1, "chunk_count": 1},
    ]
    session.chunks = [
        {"chunk_id": "Alpha.pdf:1:0", "filename": "Alpha.pdf", "page": 1, "section": 0, "text": "alpha text"},
        {"chunk_id": "Beta.pdf:1:0", "filename": "Beta.pdf", "page": 1, "section": 0, "text": "beta text"},
    ]
    lexical_index, lexical_tokens = build_bm25_index(session.chunks)
    session.lexical_index = lexical_index
    session.lexical_tokens = lexical_tokens
    session.semantic_index = build_faiss_index(session.chunks, app.state.settings.embedding_dimension)

    response = client.post(
        "/upload/remove",
        json={"session_id": "remove-sync", "file_keys": ["alpha.pdf", "missing.pdf"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["removed_files"] == ["alpha.pdf"]
    assert payload["processed_files"] == ["beta.pdf"]
    assert payload["uploaded_documents"] == [
        {"filename": "Beta.pdf", "normalized_key": "beta.pdf", "page_count": 1, "chunk_count": 1}
    ]

    current = app.state.session_store.get("remove-sync")
    assert current is not None
    assert current.processed_files == {"beta.pdf"}
    assert len(current.chunks) == 1
    assert current.chunks[0]["filename"] == "Beta.pdf"


def test_upload_accepts_multiple_text_document_formats() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/upload",
        data={"session_id": "multi-format-upload"},
        files=[
            ("files", ("notes.txt", b"alpha line\nbeta line", "text/plain")),
            ("files", ("readme.md", b"# Title\nMarkdown body", "text/markdown")),
            ("files", ("table.csv", b"name,value\nfoo,42\nbar,77", "text/csv")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["accepted_files"]) == {"notes.txt", "readme.md", "table.csv"}

    current = app.state.session_store.get("multi-format-upload")
    assert current is not None
    assert current.processed_files == {"notes.txt", "readme.md", "table.csv"}
    assert len(current.uploaded_documents) == 3
    assert len(current.chunks) >= 3
