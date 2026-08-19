from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.services.rag_search import rag_search


def _settings() -> Settings:
    return Settings(
        OPENROUTER_API_KEY="test-key",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_ALLOWED_MODELS="openai/gpt-oss-120b:free",
    )


def test_rag_search_returns_no_documents_for_empty_session() -> None:
    result = rag_search(_settings(), InMemorySessionStore(), "policy", "missing-session")

    assert result["status"] == "no_documents"
    assert result["evidence"] == ""
    assert result["sources"] == []
    assert result["images"] == []


def test_rag_search_accepts_image_only_session(monkeypatch) -> None:
    store = InMemorySessionStore()
    session = store.get_or_create("image-only-session")
    session.image_assets.append(
        {
            "asset_id": "image-1",
            "filename": "screenshot.png",
            "image_data_url": "data:image/png;base64,abc",
        }
    )
    monkeypatch.setattr("project.backend.app.services.rag_search.retrieve_lexical", lambda *args: [])
    monkeypatch.setattr("project.backend.app.services.rag_search.retrieve_semantic", lambda *args: [])
    monkeypatch.setattr(
        "project.backend.app.services.rag_search.retrieve_image_assets",
        lambda *args: [
            {
                "asset_id": "image-1",
                "filename": "screenshot.png",
                "image_data_url": "data:image/png;base64,abc",
                "modality": "image",
            }
        ],
    )
    monkeypatch.setattr(
        "project.backend.app.services.rag_search.compress_evidence",
        lambda *args: "image evidence",
        raising=False,
    )

    result = rag_search(_settings(), store, "What is in the image?", "image-only-session")

    assert result["status"] == "ok"
    assert result["images"]


def test_rag_search_returns_no_results_when_documents_do_not_match(monkeypatch) -> None:
    store = InMemorySessionStore()
    session = store.get_or_create("session-with-docs")
    session.uploaded_documents.append({"filename": "policy.pdf"})
    monkeypatch.setattr("project.backend.app.services.rag_search.retrieve_lexical", lambda *args: [])
    monkeypatch.setattr("project.backend.app.services.rag_search.retrieve_semantic", lambda *args: [])
    monkeypatch.setattr("project.backend.app.services.rag_search.retrieve_image_assets", lambda *args: [])

    result = rag_search(_settings(), store, "unrelated", "session-with-docs")

    assert result["status"] == "no_results"
    assert result["diagnostics"]["status"] == "no_results"