from __future__ import annotations

from typing import Any

from project.backend.app.core.config import Settings


def build_runtime_config(settings: Settings) -> dict[str, Any]:
    """Build frontend-safe configuration payload.

    This payload intentionally excludes secrets.
    """
    return {
        "provider_name": settings.llm_provider_name,
        "available_models": settings.allowed_models(),
        "default_model": settings.openrouter_model,
        "default_llm_settings": settings.default_llm_settings(),
        "supported_controls": ["model", "temperature", "top_p", "max_tokens"],
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
            "max_tokens": {
                "min": settings.llm_min_max_tokens,
                "max": settings.llm_hard_max_tokens,
                "step": 64,
            },
        },
        "recommended_defaults": settings.default_llm_settings(),
    }
