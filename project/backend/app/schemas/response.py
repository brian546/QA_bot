from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class UploadResponse(BaseModel):
    session_id: str
    accepted_files: list[str]
    uploaded_documents: list[dict[str, Any]]


class RemoveFilesResponse(BaseModel):
    session_id: str
    removed_files: list[str]
    uploaded_documents: list[dict[str, Any]]
    processed_files: list[str]


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieval_diagnostics: dict[str, Any]
    effective_llm_settings: dict[str, Any]
    effective_retrieval_settings: dict[str, Any]


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool


class SessionSummary(BaseModel):
    session_id: str
    uploaded_document_count: int
    chat_message_count: int


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionDetailResponse(BaseModel):
    session_id: str
    uploaded_documents: list[dict[str, Any]]
    chat_history: list[dict[str, str]]
    processed_files: list[str]
    llm_settings: dict[str, Any]
    retrieval_settings: dict[str, Any]


class RuntimeConfigResponse(BaseModel):
    agent_provider_name: str
    provider_name: str
    embedding_provider_name: str
    embedding_provider: str
    available_models: list[str]
    default_model: str
    default_embedding_model: str
    default_llm_settings: dict[str, Any]
    default_retrieval_settings: dict[str, Any]
    default_citations_k: int
    supported_controls: list[str]
    parameter_constraints: dict[str, Any]
    recommended_defaults: dict[str, Any]
