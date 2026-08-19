from __future__ import annotations

from typing import Any

from project.backend.app.core.config import Settings
from project.backend.app.core.llm import validate_and_merge_llm_settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.state import GraphState
from project.backend.app.services.hybrid_retrieval import build_diagnostics, reciprocal_rank_fusion
from project.backend.app.services.lexical_retrieval import retrieve_lexical
from project.backend.app.services.image_retrieval import build_image_diagnostics, retrieve_image_assets
from project.backend.app.services.qa import (
    answer_with_agent,
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

    def _resolve_retrieval_settings(self, state: GraphState) -> dict[str, float]:
        """
        Normalizing lexical and semantic weights to sum to 1.0, with defaults and bounds applied. This allows dynamic adjustment of retrieval strategy while ensuring valid weight distribution. If both weights are zero or negative, they will be reset to equal values to avoid division errors and ensure a balanced fusion of results.
        """
        defaults = {
            "lexical_weight": float(self.settings.lexical_weight),
            "semantic_weight": float(self.settings.semantic_weight),
        }
        incoming = state.get("retrieval_settings") or {}

        lexical_weight = float(incoming.get("lexical_weight", defaults["lexical_weight"]))
        semantic_weight = float(incoming.get("semantic_weight", defaults["semantic_weight"]))

        lexical_weight = max(0.0, min(1.0, lexical_weight))
        semantic_weight = max(0.0, min(1.0, semantic_weight))

        total = lexical_weight + semantic_weight
        if total <= 0:
            resolved = {
                "lexical_weight": 0.5,
                "semantic_weight": 0.5,
            }
        else:
            resolved = {
                "lexical_weight": lexical_weight / total,
                "semantic_weight": semantic_weight / total,
            }

        state["retrieval_settings"] = resolved
        return resolved

    def ingest_upload(self, state: GraphState) -> GraphState:
        session = self.session_store.get_or_create(state["session_id"])
        state["uploaded_documents"] = list(session.uploaded_documents)
        state["image_assets"] = list(session.image_assets)
        state.setdefault("accepted_files", [])
        state.setdefault("uploaded_files", [])
        return state

    @staticmethod
    def _looks_like_image_question(question: str) -> bool:
        lowered = question.lower()
        return any(
            token in lowered
            for token in (
                "image",
                "photo",
                "picture",
                "screenshot",
                "figure",
                "diagram",
                "chart",
                "visual",
                "show me",
                "what is in",
                "what do you see",
            )
        )

    def query_router(self, state: GraphState) -> GraphState:
        """
        Decide whether to route directly to answer generation or through the retrieval path based on the presence of uploaded documents and the nature of the question. This allows for efficient handling of questions that may not require retrieval while ensuring that document-based questions are properly processed.
        """
        llm_settings = validate_and_merge_llm_settings(self.settings, state.get("llm_settings"))
        self._resolve_retrieval_settings(state)
        docs_available = bool(state.get("uploaded_documents"))
        needs_search = should_search_documents(
            self.settings,
            state.get("current_question", ""),
            state.get("chat_history", []),
            docs_available,
            llm_settings,
        )

        state["llm_settings"] = llm_settings
        state["needs_document_search"] = needs_search
        state["route_decision"] = "search" if needs_search else "direct"
        return state

    def rewrite_query(self, state: GraphState) -> GraphState:
        """
        Rewrite the current question into a standalone query using the chat history.
        This helps in providing context-aware answers while maintaining clarity and relevance.
        """
        rewritten = rewrite_query_with_history(
            self.settings,
            state.get("current_question", ""),
            state.get("chat_history", []),
            state.get("llm_settings", {}),
        )
        state["rewritten_query"] = rewritten
        return state

    def lexical_retrieve(self, state: GraphState) -> GraphState:
        """
        Perform lexical retrieval based on the rewritten query or the current question.
        This method uses the session's lexical index to find relevant chunks of text.
        """
        session = self.session_store.get_or_create(state["session_id"])
        query = state.get("rewritten_query") or state.get("current_question", "")
        results = retrieve_lexical(query, session.chunks, session.lexical_index, self.settings.retrieval_lexical_k)
        state["lexical_results"] = results
        return state

    def semantic_retrieve(self, state: GraphState) -> GraphState:
        """
        Perform semantic retrieval based on the rewritten query or the current question.
        This method uses the session's semantic index to find relevant chunks of text.
        """
        session = self.session_store.get_or_create(state["session_id"])
        query = state.get("rewritten_query") or state.get("current_question", "")
        results = retrieve_semantic(query, session.semantic_index, self.settings.retrieval_semantic_k)
        state["semantic_results"] = results
        image_results = retrieve_image_assets(query, session.image_index, self.settings, self.settings.retrieval_semantic_k)
        if not image_results and session.image_assets:
            if self._looks_like_image_question(query) or not session.chunks:
                image_results = [
                    dict(asset, score=0.0, source="image", modality="image")
                    for asset in session.image_assets[: self.settings.retrieval_semantic_k]
                ]
        state["image_results"] = image_results
        return state

    def fuse_results(self, state: GraphState) -> GraphState:
        """
        Fuse lexical and semantic retrieval results using reciprocal rank fusion.
        This method combines the strengths of both retrieval strategies to produce a more comprehensive set of results.
        """
        lexical = state.get("lexical_results", [])
        semantic = state.get("semantic_results", [])
        retrieval_settings = state.get("retrieval_settings") or self._resolve_retrieval_settings(state)
        fused = reciprocal_rank_fusion(
            lexical,
            semantic,
            lexical_weight=retrieval_settings["lexical_weight"],
            semantic_weight=retrieval_settings["semantic_weight"],
            top_k=max(self.settings.retrieval_lexical_k, self.settings.retrieval_semantic_k),
            image_results=state.get("image_results", []),
        )
        state["fused_results"] = fused
        state["retrieval_diagnostics"] = build_diagnostics(
            lexical,
            semantic,
            fused,
            top_k=int(self.settings.citations_max_k),
            image_results=state.get("image_results", []),
        )
        return state

    def compress_context(self, state: GraphState) -> GraphState:
        compressed = compress_evidence(
            self.settings,
            state.get("current_question", ""),
            state.get("fused_results", []),
            state.get("llm_settings", {}),
        )
        state["compressed_context"] = compressed
        return state

    def answer_question(self, state: GraphState) -> GraphState:
        local_evidence = state.get("compressed_context", "")
        answer, citations, agent_diagnostics = answer_with_agent(
            self.settings,
            state.get("current_question", ""),
            state.get("chat_history", []),
            local_evidence,
            state.get("fused_results", []),
            state.get("llm_settings", {}),
        )
        state["final_answer"] = answer
        state["citations"] = citations
        diagnostics = state.get("retrieval_diagnostics") or {}
        diagnostics.update(agent_diagnostics)
        diagnostics.setdefault("lexical_hits", [])
        diagnostics.setdefault("semantic_hits", [])
        diagnostics.setdefault("image_hits", [])
        diagnostics.setdefault("fused_hits", [])
        state["retrieval_diagnostics"] = diagnostics
        state["route_decision"] = "web_search" if agent_diagnostics.get("web_hits") else "direct"
        return state

    def evaluate_answer(self, state: GraphState) -> GraphState:
        if state.get("route_decision") in {"direct", "web_search"}:
            state["should_web_search"] = False
            state["should_fallback"] = False
            return state

        has_docs = bool(state.get("uploaded_documents"))
        has_evidence = bool(state.get("fused_results"))
        has_citations = bool(state.get("citations"))

        if not has_docs:
            state["should_web_search"] = False
            state["should_fallback"] = True
            state["error"] = "No uploaded documents found for this session."
            return state

        if not has_evidence or not has_citations:
            state["should_web_search"] = False
            state["should_fallback"] = True
            state["error"] = "Insufficient evidence for grounded answer."
            return state

        state["answer_is_confident"] = True
        state["should_web_search"] = False
        state["should_fallback"] = False
        return state

    def fallback(self, state: GraphState) -> GraphState:
        state["final_answer"] = "I could not find enough evidence in the uploaded documents or images."
        state["citations"] = []
        state["answer_is_confident"] = False
        state.setdefault(
            "retrieval_diagnostics",
            {"lexical_hits": [], "semantic_hits": [], "image_hits": [], "fused_hits": [], "web_hits": []},
        )
        return state
