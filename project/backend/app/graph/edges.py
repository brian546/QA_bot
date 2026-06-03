from __future__ import annotations

from project.backend.app.graph.state import GraphState


def route_after_ingest(state: GraphState) -> str:
    return "query_router"


def route_after_query_router(state: GraphState) -> str:
    if state.get("needs_document_search"):
        return "rewrite_query"
    return "answer_direct"


def route_after_evaluate(state: GraphState) -> str:
    if state.get("should_fallback"):
        return "fallback"
    return "finish"
