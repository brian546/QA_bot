from __future__ import annotations

from project.backend.app.graph.state import GraphState



def route_after_evaluate(state: GraphState) -> str:
    if state.get("should_fallback"):
        return "fallback"
    return "finish"
