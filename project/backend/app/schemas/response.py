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
    skipped_files: list[str]
    uploaded_documents: list[dict[str, Any]]


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    retrieval_diagnostics: dict[str, Any]
    effective_llm_settings: dict[str, Any]


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool


class RuntimeConfigResponse(BaseModel):
    provider_name: str
    available_models: list[str]
    default_model: str
    default_llm_settings: dict[str, Any]
    supported_controls: list[str]
    parameter_constraints: dict[str, Any]
    recommended_defaults: dict[str, Any]
