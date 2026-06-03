from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.builder import build_graph


def test_graph_routes_to_direct_answer_without_docs() -> None:
    settings = Settings(OPENROUTER_API_KEY="x", OPENROUTER_MODEL="openai/gpt-oss-120b:free")
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
