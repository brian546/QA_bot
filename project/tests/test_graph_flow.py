from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.builder import build_graph


def test_graph_routes_to_fallback_without_docs() -> None:
    settings = Settings(OPENROUTER_API_KEY="x", OPENROUTER_MODEL="openai/gpt-4o-mini")
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

    assert "could not find enough evidence" in result["final_answer"].lower()
