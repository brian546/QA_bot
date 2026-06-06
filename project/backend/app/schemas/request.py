from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMSettingsPayload(BaseModel):
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    chat_history: list[dict[str, str]] | None = None
    llm_settings: LLMSettingsPayload | None = None


class ClearSessionRequest(BaseModel):
    session_id: str = Field(min_length=1)


class UploadRequestMeta(BaseModel):
    session_id: str = Field(min_length=1)


class RemoveFilesRequest(BaseModel):
    session_id: str = Field(min_length=1)
    file_keys: list[str] = Field(min_length=1)


class AskDiagnostics(BaseModel):
    lexical_hits: list[dict[str, Any]]
    semantic_hits: list[dict[str, Any]]
    fused_hits: list[dict[str, Any]]
