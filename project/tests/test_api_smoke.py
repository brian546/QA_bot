import io

from fastapi.testclient import TestClient
from openpyxl import Workbook

from project.backend.app.main import create_app
from project.backend.app.services.lexical_retrieval import build_bm25_index
from project.backend.app.services.semantic_retrieval import build_faiss_index


def test_api_smoke_health_and_ask() -> None:
    app = create_app()
    client = TestClient(app)
    default_llm_settings = app.state.settings.default_llm_settings()

    health = client.get("/health")
    assert health.status_code == 200

    ask = client.post(
        "/ask",
        json={
            "session_id": "s-smoke",
            "question": "What is this about?",
            "citations_k": 3,
            "llm_settings": default_llm_settings,
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

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet.append(["name", "value"])
    sheet.append(["foo", 42])
    sheet.append(["bar", 77])
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()
    xlsx_bytes = buffer.getvalue()

    response = client.post(
        "/upload",
        data={"session_id": "multi-format-upload"},
        files=[
            ("files", ("notes.txt", b"alpha line\nbeta line", "text/plain")),
            ("files", ("readme.md", b"# Title\nMarkdown body", "text/markdown")),
            ("files", ("table.csv", b"name,value\nfoo,42\nbar,77", "text/csv")),
            (
                "files",
                (
                    "table.xlsx",
                    xlsx_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            ),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["accepted_files"]) == {"notes.txt", "readme.md", "table.csv", "table.xlsx"}

    current = app.state.session_store.get("multi-format-upload")
    assert current is not None
    assert current.processed_files == {"notes.txt", "readme.md", "table.csv", "table.xlsx"}
    assert len(current.uploaded_documents) == 4
    assert len(current.chunks) >= 4


def test_upload_succeeds_when_semantic_index_build_fails(monkeypatch) -> None:
    app = create_app()
    client = TestClient(app)

    def _raise_index_error(*args, **kwargs):
        raise RuntimeError("embedding backend unavailable")

    monkeypatch.setattr("project.backend.app.services.semantic_retrieval.FAISS.from_documents", _raise_index_error)

    response = client.post(
        "/upload",
        data={"session_id": "semantic-fallback"},
        files=[("files", ("notes.txt", b"alpha line\nbeta line", "text/plain"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted_files"] == ["notes.txt"]

    current = app.state.session_store.get("semantic-fallback")
    assert current is not None
    assert current.lexical_index is not None
    assert current.semantic_index is None
