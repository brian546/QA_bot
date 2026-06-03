from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.edges import route_after_evaluate, route_after_ingest, route_after_query_router
from project.backend.app.graph.nodes import GraphNodes
from project.backend.app.graph.state import GraphState


def build_graph(settings: Settings, session_store: InMemorySessionStore):
    """Build and compile the LangGraph workflow for retrieval and QA."""
    graph = StateGraph(GraphState)
    nodes = GraphNodes(settings, session_store)

    graph.add_node("ingest_upload", nodes.ingest_upload)
    graph.add_node("query_router", nodes.query_router)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("answer_direct", nodes.answer_direct)
    graph.add_node("lexical_retrieve", nodes.lexical_retrieve)
    graph.add_node("semantic_retrieve", nodes.semantic_retrieve)
    graph.add_node("fuse_results", nodes.fuse_results)
    graph.add_node("compress_context", nodes.compress_context)
    graph.add_node("answer_question", nodes.answer_question)
    graph.add_node("evaluate_answer", nodes.evaluate_answer)
    graph.add_node("fallback", nodes.fallback)

    graph.add_edge(START, "ingest_upload")
    graph.add_conditional_edges(
        "ingest_upload",
        route_after_ingest,
        {
            "query_router": "query_router",
        },
    )

    graph.add_conditional_edges(
        "query_router",
        route_after_query_router,
        {
            "rewrite_query": "rewrite_query",
            "answer_direct": "answer_direct",
        },
    )

    graph.add_edge("answer_direct", END)
    graph.add_edge("rewrite_query", "lexical_retrieve")
    graph.add_edge("lexical_retrieve", "semantic_retrieve")
    graph.add_edge("semantic_retrieve", "fuse_results")
    graph.add_edge("fuse_results", "compress_context")
    graph.add_edge("compress_context", "answer_question")
    graph.add_edge("answer_question", "evaluate_answer")

    graph.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluate,
        {
            "fallback": "fallback",
            "finish": END,
        },
    )

    graph.add_edge("fallback", END)
    return graph.compile()
