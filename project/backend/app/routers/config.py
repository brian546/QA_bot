from __future__ import annotations

from fastapi import APIRouter, Request

from project.backend.app.core.runtime_config import build_runtime_config
from project.backend.app.schemas.response import RuntimeConfigResponse

router = APIRouter(tags=["config"])


@router.get("/config", response_model=RuntimeConfigResponse)
def get_runtime_config(request: Request) -> RuntimeConfigResponse:
    settings = request.app.state.settings
    payload = build_runtime_config(settings)
    return RuntimeConfigResponse(**payload)
