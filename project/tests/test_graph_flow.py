from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.builder import build_graph
from project.backend.app.graph.nodes import GraphNodes


def test_graph_routes_to_direct_answer_without_docs() -> None:
    settings = Settings(
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_ALLOWED_MODELS="openai/gpt-oss-120b:free",
    )
    store = InMemorySessionStore()
    graph = build_graph(settings, store)

    result = graph.invoke(
        {
            "session_id": "s1",
            "current_question": "What is the key policy?",
            "chat_history": [],
            "llm_settings": settings.default_llm_settings(),
        }
    )

    assert result.get("route_decision") == "direct"
    assert result.get("final_answer")


def test_graph_routes_time_sensitive_question_to_web_search(monkeypatch) -> None:
    settings = Settings(
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_ALLOWED_MODELS="openai/gpt-oss-120b:free",
        TAVILY_API_KEY="test-key",
    )
    store = InMemorySessionStore()
    graph = build_graph(settings, store)
    monkeypatch.setattr(
        "project.backend.app.graph.nodes.search_web_with_tavily",
        lambda *_: [
            {
                "title": "Example Headline",
                "url": "https://example.com/news",
                "content": "Snippet",
            }
        ],
    )

    result = graph.invoke(
        {
            "session_id": "s2",
            "current_question": "What are the latest AI news headlines today?",
            "chat_history": [],
            "llm_settings": settings.default_llm_settings(),
        }
    )

    assert result.get("route_decision") == "web_search"
    assert result.get("final_answer")
    assert result.get("citations")
    assert str(result["citations"][0].get("modality", "")) == "web"


def test_evaluate_answer_uses_web_search_when_grounding_is_insufficient_for_time_sensitive_query() -> None:
    settings = Settings(
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_ALLOWED_MODELS="openai/gpt-oss-120b:free",
    )
    store = InMemorySessionStore()
    nodes = GraphNodes(settings, store)

    state = {
        "session_id": "s3",
        "current_question": "What are the latest updates on this topic today?",
        "chat_history": [],
        "route_decision": "search",
        "uploaded_documents": [{"filename": "Doc.pdf"}],
        "fused_results": [],
        "citations": [],
        "llm_settings": settings.default_llm_settings(),
    }
    result = nodes.evaluate_answer(state)

    assert result.get("should_web_search") is True
    assert result.get("should_fallback") is False
