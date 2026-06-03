from __future__ import annotations

from project.backend.app.graph.state import GraphState


def route_after_ingest(state: GraphState) -> str:
    if not state.get("uploaded_documents"):
        return "fallback"
    return "rewrite_query"


def route_after_evaluate(state: GraphState) -> str:
    if state.get("should_fallback"):
        return "fallback"
    return "finish"
