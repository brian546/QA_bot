from __future__ import annotations

from typing import Any

from project.backend.app.core.config import Settings
from project.backend.app.core.llm import validate_and_merge_llm_settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.state import GraphState
from project.backend.app.services.hybrid_retrieval import build_diagnostics, reciprocal_rank_fusion
from project.backend.app.services.lexical_retrieval import retrieve_lexical
from project.backend.app.services.qa import (
    answer_directly,
    answer_with_evidence,
    compress_evidence,
    rewrite_query_with_history,
    should_search_documents,
)
from project.backend.app.services.semantic_retrieval import retrieve_semantic


class GraphNodes:
    """Collection of LangGraph node handlers."""

    def __init__(self, settings: Settings, session_store: InMemorySessionStore) -> None:
        self.settings = settings
        self.session_store = session_store

    def ingest_upload(self, state: GraphState) -> GraphState:
        session = self.session_store.get_or_create(state["session_id"])
        state["uploaded_documents"] = list(session.uploaded_documents)
        state.setdefault("accepted_files", [])
        state.setdefault("skipped_files", [])
        state.setdefault("uploaded_files", [])
        return state

    def query_router(self, state: GraphState) -> GraphState:
        llm_settings = validate_and_merge_llm_settings(self.settings, state.get("llm_settings"))
        docs_available = bool(state.get("uploaded_documents"))
        needs_search = should_search_documents(
            self.settings,
            state.get("current_question", ""),
            state.get("chat_history", []),
            docs_available,
            llm_settings,
        )

        # No docs means retrieval is impossible, so route direct.
        if not docs_available:
            needs_search = False

        state["effective_llm_settings"] = llm_settings
        state["needs_document_search"] = needs_search
        state["route_decision"] = "search" if needs_search else "direct"
        return state

    def rewrite_query(self, state: GraphState) -> GraphState:
        llm_settings = state.get("effective_llm_settings") or validate_and_merge_llm_settings(
            self.settings, state.get("llm_settings")
        )
        rewritten = rewrite_query_with_history(
            self.settings,
            state.get("current_question", ""),
            state.get("chat_history", []),
            llm_settings,
        )
        state["rewritten_query"] = rewritten
        state["effective_llm_settings"] = llm_settings
        return state

    def lexical_retrieve(self, state: GraphState) -> GraphState:
        session = self.session_store.get_or_create(state["session_id"])
        query = state.get("rewritten_query") or state.get("current_question", "")
        results = retrieve_lexical(query, session.chunks, session.lexical_index, self.settings.retrieval_lexical_k)
        state["lexical_results"] = results
        return state

    def semantic_retrieve(self, state: GraphState) -> GraphState:
        session = self.session_store.get_or_create(state["session_id"])
        query = state.get("rewritten_query") or state.get("current_question", "")
        results = retrieve_semantic(query, session.semantic_index, self.settings.retrieval_semantic_k)
        state["semantic_results"] = results
        return state

    def fuse_results(self, state: GraphState) -> GraphState:
        lexical = state.get("lexical_results", [])
        semantic = state.get("semantic_results", [])
        fused = reciprocal_rank_fusion(
            lexical,
            semantic,
            lexical_weight=self.settings.lexical_weight,
            semantic_weight=self.settings.semantic_weight,
            top_k=max(self.settings.retrieval_lexical_k, self.settings.retrieval_semantic_k),
        )
        state["fused_results"] = fused
        state["retrieval_diagnostics"] = build_diagnostics(lexical, semantic, fused)
        return state

    def compress_context(self, state: GraphState) -> GraphState:
        llm_settings = state.get("effective_llm_settings", {})
        compressed = compress_evidence(
            self.settings,
            state.get("current_question", ""),
            state.get("fused_results", []),
            llm_settings,
        )
        state["compressed_context"] = compressed
        return state

    def answer_question(self, state: GraphState) -> GraphState:
        if state.get("route_decision") == "direct":
            answer, confidence = answer_directly(
                self.settings,
                state.get("current_question", ""),
                state.get("chat_history", []),
                state.get("effective_llm_settings", {}),
            )
            state["final_answer"] = answer
            state["confidence"] = confidence
            state["citations"] = []
            state["retrieval_diagnostics"] = {"lexical_hits": [], "semantic_hits": [], "fused_hits": []}
            return state

        answer, citations, confidence = answer_with_evidence(
            self.settings,
            state.get("current_question", ""),
            state.get("compressed_context", ""),
            state.get("fused_results", []),
            state.get("effective_llm_settings", {}),
        )
        state["final_answer"] = answer
        state["citations"] = citations
        state["confidence"] = confidence
        return state

    def evaluate_answer(self, state: GraphState) -> GraphState:
        if state.get("route_decision") == "direct":
            state["should_fallback"] = False
            return state

        has_docs = bool(state.get("uploaded_documents"))
        has_evidence = bool(state.get("fused_results"))
        has_citations = bool(state.get("citations"))
        confidence = float(state.get("confidence", 0.0))

        if not has_docs:
            state["should_fallback"] = True
            state["error"] = "No uploaded PDFs found for this session."
            return state

        if not has_evidence or not has_citations or confidence < 0.2:
            state["should_fallback"] = True
            state["error"] = "Insufficient evidence for grounded answer."
            return state

        if confidence < 0.45:
            state["final_answer"] = f"{state.get('final_answer', '')}\n\nWarning: low confidence answer."

        state["should_fallback"] = False
        return state

    def fallback(self, state: GraphState) -> GraphState:
        state["final_answer"] = "I could not find enough evidence in the uploaded PDFs."
        state["citations"] = []
        state["confidence"] = 0.1
        state.setdefault("retrieval_diagnostics", {"lexical_hits": [], "semantic_hits": [], "fused_hits": []})
        return state
