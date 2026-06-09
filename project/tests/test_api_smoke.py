import io
import base64

from fastapi.testclient import TestClient
from openpyxl import Workbook

from project.backend.app.main import create_app
from project.backend.app.services.qa import answer_with_evidence, is_answer_confident
from project.backend.app.services.lexical_retrieval import build_bm25_index
from project.backend.app.services.semantic_retrieval import build_faiss_index
from project.backend.app.core.config import Settings


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3Z1XUAAAAASUVORK5CYII="
)


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


def test_upload_accepts_standalone_image_files() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/upload",
        data={"session_id": "image-upload"},
        files=[("files", ("diagram.png", PNG_1X1, "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted_files"] == ["diagram.png"]
    assert payload["uploaded_documents"][0]["modality"] == "image"

    session = app.state.session_store.get("image-upload")
    assert session is not None
    assert len(session.image_assets) == 1
    assert session.image_assets[0]["filename"] == "diagram.png"
    assert session.image_index is not None


def test_answer_with_evidence_uses_raw_image_payload(monkeypatch) -> None:
    app = create_app()
    settings = app.state.settings
    captured: dict[str, object] = {}

    class FakeModel:
        def invoke(self, messages):
            captured["messages"] = messages

            class _Response:
                content = "The image appears to show a simple diagram."

            return _Response()

    monkeypatch.setattr("project.backend.app.services.qa.get_chat_model", lambda *args, **kwargs: FakeModel())

    answer, citations = answer_with_evidence(
        settings,
        "What is in the image?",
        "Image asset from diagram.png page 1 (1x1)",
        [
            {
                "chunk_id": "diagram.png:1:0",
                "filename": "diagram.png",
                "page": 1,
                "section": 0,
                "modality": "image",
                "asset_id": "asset-1",
                "image_data_url": f"data:image/png;base64,{base64.b64encode(PNG_1X1).decode()}",
            }
        ],
        settings.default_llm_settings(),
        1,
    )

    assert answer == "The image appears to show a simple diagram."
    assert citations[0]["modality"] == "image"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    human_message = messages[1]
    content = getattr(human_message, "content", None)
    assert isinstance(content, list)
    assert any(isinstance(item, dict) and item.get("type") == "image_url" for item in content)


def test_answer_with_evidence_uses_ollama_image_payload(monkeypatch) -> None:
    settings = Settings(
        LLM_PROVIDER="ollama",
        OLLAMA_MODEL="gemma4:26b",
        OLLAMA_ALLOWED_MODELS="gemma4:26b",
        OPENROUTER_API_KEY="",
    )
    captured: dict[str, object] = {}

    class FakeModel:
        def invoke(self, messages):
            captured["messages"] = messages

            class _Response:
                content = "The image contains a diagram."

            return _Response()

    monkeypatch.setattr("project.backend.app.services.qa.get_chat_model", lambda *args, **kwargs: FakeModel())

    _, citations = answer_with_evidence(
        settings,
        "What is in the image?",
        "Image asset from diagram.png page 1 (1x1)",
        [
            {
                "chunk_id": "diagram.png:1:0",
                "filename": "diagram.png",
                "page": 1,
                "section": 0,
                "modality": "image",
                "asset_id": "asset-1",
                "image_data_url": f"data:image/png;base64,{base64.b64encode(PNG_1X1).decode()}",
            }
        ],
        settings.default_llm_settings(),
        1,
    )

    assert citations[0]["modality"] == "image"
    messages = captured["messages"]
    assert isinstance(messages, list)
    human_message = messages[1]
    content = getattr(human_message, "content", None)
    assert isinstance(content, list)
    image_parts = [item for item in content if isinstance(item, dict) and item.get("type") == "image_url"]
    assert image_parts
    assert image_parts[0]["image_url"].startswith("data:image/png;base64,")


def test_is_answer_confident_uses_raw_image_payload(monkeypatch) -> None:
    app = create_app()
    settings = app.state.settings
    captured: dict[str, object] = {}

    class FakeModel:
        def invoke(self, messages):
            captured["messages"] = messages

            class _Response:
                content = "CONFIDENT"

            return _Response()

    monkeypatch.setattr("project.backend.app.services.qa.get_chat_model", lambda *args, **kwargs: FakeModel())

    confident = is_answer_confident(
        settings=settings,
        question="What is in the image?",
        answer="The image shows a simple icon.",
        compressed_context="[diagram.png:1:0] image evidence",
        citations=[
            {
                "chunk_id": "diagram.png:1:0",
                "filename": "diagram.png",
                "page": 1,
                "modality": "image",
                "asset_id": "asset-1",
                "image_data_url": f"data:image/png;base64,{base64.b64encode(PNG_1X1).decode()}",
            }
        ],
        llm_settings=settings.default_llm_settings(),
    )

    assert confident is True
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    human_message = messages[1]
    content = getattr(human_message, "content", None)
    assert isinstance(content, list)
    assert any(isinstance(item, dict) and item.get("type") == "image_url" for item in content)


def test_ask_falls_back_to_uploaded_image_when_retrieval_is_empty(monkeypatch) -> None:
    app = create_app()
    client = TestClient(app)

    upload = client.post(
        "/upload",
        data={"session_id": "image-fallback"},
        files=[("files", ("diagram.png", PNG_1X1, "image/png"))],
    )
    assert upload.status_code == 200

    monkeypatch.setattr("project.backend.app.services.image_retrieval.retrieve_image_assets", lambda *args, **kwargs: [])

    captured_messages: list[object] = []

    class FakeModel:
        def invoke(self, messages):
            captured_messages.append(messages)

            class _Response:
                content = "The image shows a simple icon."

            return _Response()

    monkeypatch.setattr("project.backend.app.services.qa.get_chat_model", lambda *args, **kwargs: FakeModel())

    response = client.post(
        "/ask",
        json={
            "session_id": "image-fallback",
            "question": "What is in the image?",
            "citations_k": 1,
            "llm_settings": app.state.settings.default_llm_settings(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"][0]["modality"] == "image"

    image_payload_found = False
    for messages in captured_messages:
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        human_message = messages[1]
        content = getattr(human_message, "content", None)
        if isinstance(content, list) and any(isinstance(item, dict) and item.get("type") == "image_url" for item in content):
            image_payload_found = True
            break

    assert image_payload_found
