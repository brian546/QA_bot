from __future__ import annotations

from typing import Any

from project.backend.app.core.config import Settings


def build_runtime_config(settings: Settings) -> dict[str, Any]:
    """Build frontend-safe configuration payload.

    This payload intentionally excludes secrets.
    """
    return {
        "agent_provider_name": settings.provider_name(),
        "provider_name": settings.provider_name(),
        "embedding_provider_name": settings.embedding_provider_name(),
        "embedding_provider": settings.embedding_provider,
        "available_models": settings.allowed_models(),
        "default_model": settings.active_llm_model(),
        "default_embedding_model": settings.active_embedding_model(),
        "default_llm_settings": settings.default_llm_settings(),
        "default_retrieval_settings": {
            "lexical_weight": settings.lexical_weight,
            "semantic_weight": settings.semantic_weight,
        },
        "default_citations_k": settings.citations_default_k,
        "supported_controls": [
            "model",
            "temperature",
            "top_p",
            "lexical_weight",
            "semantic_weight",
            "citations_k",
        ],
        "parameter_constraints": {
            "temperature": {
                "min": settings.llm_min_temperature,
                "max": settings.llm_max_temperature,
                "step": 0.05,
            },
            "top_p": {
                "min": settings.llm_min_top_p,
                "max": settings.llm_max_top_p,
                "step": 0.05,
            },
            "lexical_weight": {
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            "semantic_weight": {
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            "citations_k": {
                "min": 1,
                "max": settings.citations_max_k,
                "step": 1,
            },
        },
        "recommended_defaults": settings.default_llm_settings(),
    }
