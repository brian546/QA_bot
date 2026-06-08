from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from project.backend.app.core.config import get_settings, require_llm_provider_configuration
from project.backend.app.core.session_store import session_store
from project.backend.app.graph.builder import build_graph
from project.backend.app.routers import chat, config, session, upload
from project.backend.app.schemas.response import HealthResponse


def create_app() -> FastAPI:
    settings = get_settings()
    require_llm_provider_configuration(settings)

    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.session_store = session_store
    app.state.graph = build_graph(settings, session_store)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", app=settings.app_name, version=settings.app_version)

    app.include_router(config.router)
    app.include_router(upload.router)
    app.include_router(chat.router)
    app.include_router(session.router)
    return app


app = create_app()
