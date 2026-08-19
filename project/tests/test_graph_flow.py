from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.builder import build_graph
from project.backend.app.graph.nodes import GraphNodes


def test_graph_routes_to_direct_answer_without_docs(monkeypatch) -> None:
    settings = Settings(
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_ALLOWED_MODELS="openai/gpt-oss-120b:free",
    )
    store = InMemorySessionStore()
    graph = build_graph(settings, store)
    class FakeModel:
        def invoke(self, messages):
            class Response:
                content = '{"action": "answer", "query": "", "session_id": "s1", "reason": "Direct answer"}'

            return Response()

    monkeypatch.setattr("project.backend.app.services.qa.get_chat_model", lambda *args, **kwargs: FakeModel())

    result = graph.invoke(
        {
            "session_id": "s1",
            "current_question": "What is the key policy?",
            "chat_history": [],
            "llm_settings": settings.default_llm_settings(),
        }
    )

    assert result.get("route_decision") == "agent"
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
    class FakeModel:
        def __init__(self):
            self.responses = iter(
                [
                    '{"action": "web_search", "query": "latest AI news", "session_id": "s2", "reason": "Current information"}',
                    '{"action": "answer", "query": "", "session_id": "s2", "reason": "Enough web evidence"}',
                    "Latest update [Example Headline](https://example.com/news).",
                ]
            )

        def invoke(self, messages):
            content = next(self.responses)

            class Response:
                pass

            response = Response()
            response.content = content
            return response

    fake_model = FakeModel()
    monkeypatch.setattr("project.backend.app.services.qa.get_chat_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(
        "project.backend.app.services.qa.search_web_with_tavily",
        lambda *_: [{"title": "Example Headline", "url": "https://example.com/news", "content": "Snippet"}],
    )

    result = graph.invoke(
        {
            "session_id": "s2",
            "current_question": "What are the latest AI news headlines today?",
            "chat_history": [],
            "llm_settings": settings.default_llm_settings(),
        }
    )

    assert result.get("route_decision") == "agent"
    assert result.get("final_answer")
    assert result.get("retrieval_diagnostics", {}).get("web_search_queries") == ["latest AI news"]
    assert [item["action"] for item in result["retrieval_diagnostics"]["agent_decisions"]] == [
        "web_search",
        "answer",
    ]
    assert result["citations"][0]["url"] == "https://example.com/news"


def test_evaluate_answer_does_not_start_web_search_after_agent_answer() -> None:
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

    assert result.get("should_fallback") is True


def test_evaluate_answer_routes_nonempty_failed_answer_to_fallback() -> None:
    settings = Settings(
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        OPENROUTER_ALLOWED_MODELS="openai/gpt-oss-120b:free",
    )
    nodes = GraphNodes(settings, InMemorySessionStore())

    result = nodes.evaluate_answer(
        {
            "final_answer": "I could not generate a final answer from the available evidence.",
            "retrieval_diagnostics": {"answer_failed": True},
        }
    )

    assert result["should_fallback"] is True
    assert result["answer_is_confident"] is not True
