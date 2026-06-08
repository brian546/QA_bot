from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    session_id: str
    uploaded_files: list[str]
    accepted_files: list[str]
    uploaded_documents: list[dict[str, Any]]
    chat_history: list[dict[str, str]]
    current_question: str
    rewritten_query: str
    lexical_results: list[dict[str, Any]]
    semantic_results: list[dict[str, Any]]
    fused_results: list[dict[str, Any]]
    compressed_context: str
    final_answer: str
    citations: list[dict[str, Any]]
    answer_is_confident: bool
    retrieval_diagnostics: dict[str, Any]
    llm_settings: dict[str, Any]
    retrieval_settings: dict[str, Any]
    citations_k: int
    needs_document_search: bool
    route_decision: str
    error: str
    should_fallback: bool
