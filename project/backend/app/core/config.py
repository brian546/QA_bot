from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from environment and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Hybrid Multi-PDF QA"
    app_version: str = "0.1.0"

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_allowed_models_raw: str = Field(default="", alias="OPENROUTER_ALLOWED_MODELS")
    openrouter_timeout: int = Field(default=45, alias="OPENROUTER_TIMEOUT")
    openrouter_max_retries: int = Field(default=1, alias="OPENROUTER_MAX_RETRIES")

    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_default_top_p: float = Field(default=0.2, alias="LLM_DEFAULT_TOP_P")
    llm_default_max_tokens: int = Field(default=1200, alias="LLM_DEFAULT_MAX_TOKENS")

    llm_min_temperature: float = Field(default=0.0, alias="LLM_MIN_TEMPERATURE")
    llm_max_temperature: float = Field(default=1.0, alias="LLM_MAX_TEMPERATURE")
    llm_min_top_p: float = Field(default=0.0, alias="LLM_MIN_TOP_P")
    llm_max_top_p: float = Field(default=1.0, alias="LLM_MAX_TOP_P")
    llm_min_max_tokens: int = Field(default=64, alias="LLM_MIN_MAX_TOKENS")
    llm_hard_max_tokens: int = Field(default=4096, alias="LLM_HARD_MAX_TOKENS")

    retrieval_lexical_k: int = Field(default=6, alias="RETRIEVAL_LEXICAL_K")
    retrieval_semantic_k: int = Field(default=6, alias="RETRIEVAL_SEMANTIC_K")
    lexical_weight: float = Field(default=1.0, alias="LEXICAL_WEIGHT")
    semantic_weight: float = Field(default=1.0, alias="SEMANTIC_WEIGHT")

    max_chunk_chars: int = Field(default=1200, alias="MAX_CHUNK_CHARS")
    chunk_overlap: int = Field(default=180, alias="CHUNK_OVERLAP")
    max_upload_bytes: int = Field(default=20_000_000, alias="MAX_UPLOAD_BYTES")

    llm_provider_name: str = Field(default="OpenRouter", alias="LLM_PROVIDER_NAME")

    embedding_dimension: int = 256

    @field_validator("openrouter_model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("OPENROUTER_MODEL must not be empty")
        return value.strip()

    def allowed_models(self) -> list[str]:
        """Parse OPENROUTER_ALLOWED_MODELS from CSV or JSON list."""
        raw = self.openrouter_allowed_models_raw.strip()
        if not raw:
            return [self.openrouter_model]

        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    values = [str(item).strip() for item in parsed if str(item).strip()]
                    return values or [self.openrouter_model]
            except json.JSONDecodeError:
                pass

        values = [item.strip() for item in raw.split(",") if item.strip()]
        return values or [self.openrouter_model]

    def default_llm_settings(self) -> dict[str, Any]:
        return {
            "model": self.openrouter_model,
            "temperature": self.llm_temperature,
            "top_p": self.llm_default_top_p,
            "max_tokens": self.llm_default_max_tokens,
        }


def require_openrouter_api_key(settings: Settings) -> None:
    """Fail-fast guard for missing OpenRouter API key."""
    if not settings.openrouter_api_key.strip():
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in .env before starting the backend.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings once per process."""
    load_dotenv(override=False)
    return Settings()
