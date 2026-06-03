from __future__ import annotations

from typing import Any

from langchain_openrouter import ChatOpenRouter

from project.backend.app.core.config import Settings


def validate_and_merge_llm_settings(settings: Settings, overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge defaults with user overrides and enforce constraints."""
    merged = settings.default_llm_settings()
    if overrides:
        for key in ("model", "temperature", "top_p", "max_tokens"):
            if key in overrides and overrides[key] is not None:
                merged[key] = overrides[key]

    allowed_models = settings.allowed_models()
    model = str(merged["model"]).strip()
    if model not in allowed_models:
        raise ValueError(f"Unsupported model '{model}'. Allowed models: {allowed_models}")

    temperature = float(merged["temperature"])
    if not settings.llm_min_temperature <= temperature <= settings.llm_max_temperature:
        raise ValueError(
            f"temperature must be between {settings.llm_min_temperature} and {settings.llm_max_temperature}"
        )

    top_p = float(merged["top_p"])
    if not settings.llm_min_top_p <= top_p <= settings.llm_max_top_p:
        raise ValueError(f"top_p must be between {settings.llm_min_top_p} and {settings.llm_max_top_p}")

    max_tokens = int(merged["max_tokens"])
    if not settings.llm_min_max_tokens <= max_tokens <= settings.llm_hard_max_tokens:
        raise ValueError(
            f"max_tokens must be between {settings.llm_min_max_tokens} and {settings.llm_hard_max_tokens}"
        )

    merged["model"] = model
    merged["temperature"] = temperature
    merged["top_p"] = top_p
    merged["max_tokens"] = max_tokens
    return merged


def get_chat_model(settings: Settings, llm_settings: dict[str, Any] | None = None) -> ChatOpenRouter:
    """Create a shared OpenRouter chat model from validated settings."""
    effective = validate_and_merge_llm_settings(settings, llm_settings)
    return ChatOpenRouter(
        api_key=settings.openrouter_api_key,
        model=effective["model"],
        base_url=settings.openrouter_base_url,
        temperature=effective["temperature"],
        top_p=effective["top_p"],
        max_tokens=effective["max_tokens"],
        timeout=settings.openrouter_timeout,
        max_retries=settings.openrouter_max_retries,
    )
