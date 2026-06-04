from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi


@dataclass
class SessionData:
    """In-memory state for one user session."""

    session_id: str
    processed_files: set[str] = field(default_factory=set)
    uploaded_documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    lexical_tokens: list[list[str]] = field(default_factory=list)
    lexical_index: BM25Okapi | None = None
    semantic_index: FAISS | None = None
    chat_history: list[dict[str, str]] = field(default_factory=list)
    graph_state: dict[str, Any] = field(default_factory=dict)
    llm_settings: dict[str, Any] = field(default_factory=dict)


class InMemorySessionStore:
    """Thread-safe session store for MVP use."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(session_id=session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> SessionData | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[SessionData]:
        with self._lock:
            return list(reversed(list(self._sessions.values())))

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear_all(self) -> None:
        with self._lock:
            self._sessions.clear()


session_store = InMemorySessionStore()
