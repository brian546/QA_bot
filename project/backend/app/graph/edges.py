from __future__ import annotations

from project.backend.app.graph.state import GraphState



def route_after_query_router(state: GraphState) -> str:
    if state.get("needs_document_search"):
        return "rewrite_query"
    return "decide_web_search"


def route_after_web_decision(state: GraphState) -> str:
    if state.get("needs_web_search"):
        return "web_search"
    return "answer_question"


def route_after_evaluate(state: GraphState) -> str:
    if state.get("should_web_search"):
        return "web_search"
    if state.get("should_fallback"):
        return "fallback"
    return "finish"
