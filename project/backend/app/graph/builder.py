from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.edges import route_after_evaluate
from project.backend.app.graph.nodes import GraphNodes
from project.backend.app.graph.state import GraphState


def build_graph(settings: Settings, session_store: InMemorySessionStore):
    """Build and compile the LangGraph workflow for retrieval and QA."""
    graph = StateGraph(GraphState)
    nodes = GraphNodes(settings, session_store)

    graph.add_node("ingest_upload", nodes.ingest_upload)
    graph.add_node("qa_agent", nodes.qa_agent)
    graph.add_node("evaluate_answer", nodes.evaluate_answer)
    graph.add_node("fallback", nodes.fallback)

    graph.add_edge(START, "ingest_upload")
    graph.add_edge("ingest_upload", "qa_agent")
    graph.add_edge("qa_agent", "evaluate_answer")

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
