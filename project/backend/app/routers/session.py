from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from project.backend.app.schemas.response import ClearSessionResponse

router = APIRouter(tags=["session"])


@router.post("/clear-session", response_model=ClearSessionResponse)
def clear_session(payload: dict[str, Any], request: Request) -> ClearSessionResponse:
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id is required")

    store = request.app.state.session_store
    cleared = store.clear(session_id)
    # Idempotent endpoint: repeated calls are safe.
    return ClearSessionResponse(session_id=session_id, cleared=cleared)
