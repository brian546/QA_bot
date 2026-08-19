from __future__ import annotations

from typing import Any

from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.graph.state import GraphState
from project.backend.app.services.qa import answer_with_agent


class GraphNodes:
    """Collection of LangGraph node handlers."""

    def __init__(self, settings: Settings, session_store: InMemorySessionStore) -> None:
        self.settings = settings
        self.session_store = session_store

    def ingest_upload(self, state: GraphState) -> GraphState:
        session = self.session_store.get_or_create(state["session_id"])
        state["uploaded_documents"] = list(session.uploaded_documents)
        state["image_assets"] = list(session.image_assets)
        state["session_content"] = {
            "document_count": len(session.uploaded_documents),
            "document_names": [str(document.get("filename", "")) for document in session.uploaded_documents],
            "image_count": len(session.image_assets),
        }
        state.setdefault("accepted_files", [])
        state.setdefault("uploaded_files", [])
        return state

    def qa_agent(self, state: GraphState) -> GraphState:
        answer, citations, agent_diagnostics = answer_with_agent(
            self.settings,
            self.session_store,
            state.get("current_question", ""),
            state.get("chat_history", []),
            state.get("session_id", ""),
            state.get("llm_settings", {}),
            state.get("retrieval_settings", {}),
            state.get("citations_k"),
            state.get("session_content", {}),
        )
        state["final_answer"] = answer
        state["citations"] = citations
        diagnostics = state.get("retrieval_diagnostics") or {}
        diagnostics.update(agent_diagnostics)
        diagnostics = {key: value for key, value in diagnostics.items() if value != []}
        state["retrieval_diagnostics"] = diagnostics
        state["route_decision"] = "agent"
        return state

    def evaluate_answer(self, state: GraphState) -> GraphState:
        if not state.get("final_answer", "").strip():
            state["should_web_search"] = False
            state["should_fallback"] = True
            state["error"] = "Agent did not produce an answer."
            return state

        state["answer_is_confident"] = True
        state["should_web_search"] = False
        state["should_fallback"] = False
        return state

    def fallback(self, state: GraphState) -> GraphState:
        state["final_answer"] = "I could not find enough evidence in the uploaded documents or images."
        state["citations"] = []
        state["answer_is_confident"] = False
        state.setdefault("retrieval_diagnostics", {})
        return state
