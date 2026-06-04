from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from project.backend.app.schemas.response import (
    ClearSessionResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionSummary,
)

router = APIRouter(tags=["session"])


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(request: Request) -> SessionListResponse:
    store = request.app.state.session_store
    sessions = [
        SessionSummary(
            session_id=session.session_id,
            uploaded_document_count=len(session.uploaded_documents),
            chat_message_count=len(session.chat_history),
        )
        for session in store.list_sessions()
    ]
    return SessionListResponse(sessions=sessions)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(session_id: str, request: Request) -> SessionDetailResponse:
    store = request.app.state.session_store
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionDetailResponse(
        session_id=session.session_id,
        uploaded_documents=session.uploaded_documents,
        chat_history=session.chat_history,
        processed_files=sorted(session.processed_files),
        llm_settings=session.llm_settings,
    )


@router.post("/clear-session", response_model=ClearSessionResponse)
def clear_session(payload: dict[str, Any], request: Request) -> ClearSessionResponse:
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id is required")

    store = request.app.state.session_store
    cleared = store.clear(session_id)
    # Idempotent endpoint: repeated calls are safe.
    return ClearSessionResponse(session_id=session_id, cleared=cleared)
