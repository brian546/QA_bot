from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from project.backend.app.core.llm import validate_and_merge_llm_settings
from project.backend.app.schemas.request import AskRequest
from project.backend.app.schemas.response import AskResponse

router = APIRouter(tags=["chat"])


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    settings = request.app.state.settings
    graph = request.app.state.graph
    store = request.app.state.session_store
    session = store.get_or_create(payload.session_id)

    override_settings = payload.llm_settings.model_dump(exclude_none=True) if payload.llm_settings else None
    try:
        effective_llm_settings = validate_and_merge_llm_settings(settings, override_settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    input_state = {
        "session_id": payload.session_id,
        "current_question": payload.question,
        "chat_history": payload.chat_history or session.chat_history,
        "llm_settings": effective_llm_settings,
        "citations_k": payload.citations_k,
        "retrieval_settings": (
            payload.retrieval_settings.model_dump(exclude_none=True)
            if payload.retrieval_settings
            else session.retrieval_settings
        ),
        "uploaded_files": [doc["filename"] for doc in session.uploaded_documents],
    }

    result = graph.invoke(input_state)
    session.graph_state = result
    session.llm_settings = effective_llm_settings
    session.retrieval_settings = result.get("effective_retrieval_settings", {})

    session.chat_history.append({"role": "user", "content": payload.question})
    session.chat_history.append({"role": "assistant", "content": result.get("final_answer", "")})

    return AskResponse(
        answer=result.get("final_answer", "I could not find enough evidence in the uploaded PDFs."),
        citations=result.get("citations", []),
        confidence=float(result.get("confidence", 0.1)),
        retrieval_diagnostics=result.get(
            "retrieval_diagnostics",
            {"lexical_hits": [], "semantic_hits": [], "fused_hits": []},
        ),
        effective_llm_settings=result.get("effective_llm_settings", effective_llm_settings),
        effective_retrieval_settings=result.get("effective_retrieval_settings", session.retrieval_settings),
    )
