from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from environment and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Hybrid Multi-Document QA"
    app_version: str = "0.1.0"

    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="", alias="EMBEDDING_PROVIDER")

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_embedding_model: str = Field(
        default="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        alias="OPENROUTER_EMBEDDING_MODEL",
    )
    openrouter_allowed_models_raw: str = Field(default="", alias="OPENROUTER_ALLOWED_MODELS")
    openrouter_timeout: int = Field(default=45, alias="OPENROUTER_TIMEOUT")
    openrouter_max_retries: int = Field(default=1, alias="OPENROUTER_MAX_RETRIES")

    ollama_model: str = Field(default="gemma4:26b", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_embedding_model: str = Field(default="", alias="OLLAMA_EMBEDDING_MODEL")
    ollama_allowed_models_raw: str = Field(default="", alias="OLLAMA_ALLOWED_MODELS")
    ollama_timeout: int = Field(default=45, alias="OLLAMA_TIMEOUT")

    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_default_top_p: float = Field(default=0.2, alias="LLM_DEFAULT_TOP_P")

    llm_min_temperature: float = Field(default=0.0, alias="LLM_MIN_TEMPERATURE")
    llm_max_temperature: float = Field(default=1.0, alias="LLM_MAX_TEMPERATURE")
    llm_min_top_p: float = Field(default=0.0, alias="LLM_MIN_TOP_P")
    llm_max_top_p: float = Field(default=1.0, alias="LLM_MAX_TOP_P")

    citations_max_k: int = Field(default=10, alias="CITATIONS_MAX_K")
    lexical_weight: float = Field(default=1.0, alias="LEXICAL_WEIGHT")
    semantic_weight: float = Field(default=1.0, alias="SEMANTIC_WEIGHT")

    max_chunk_chars: int = Field(default=1200, alias="MAX_CHUNK_CHARS")
    chunk_overlap: int = Field(default=180, alias="CHUNK_OVERLAP")
    max_upload_bytes: int = Field(default=20_000_000, alias="MAX_UPLOAD_BYTES")

    embedding_dimension: int = 256

    @field_validator("openrouter_model", "ollama_model")
    @classmethod
    def _strip_model_name(cls, value: str) -> str:
        return value.strip()

    @staticmethod
    def _normalize_provider(value: str, field_name: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"openrouter", "ollama"}:
            raise ValueError(f"{field_name} must be either 'openrouter' or 'ollama'")
        return normalized

    @model_validator(mode="after")
    def _validate_citation_bounds(self) -> "Settings":
        self.llm_provider = self._normalize_provider(self.llm_provider, "LLM_PROVIDER")

        if self.embedding_provider.strip():
            self.embedding_provider = self._normalize_provider(self.embedding_provider, "EMBEDDING_PROVIDER")
        else:
            # Default embedding provider to the agent provider for backward compatibility.
            self.embedding_provider = self.llm_provider

        if self.citations_max_k < 1:
            raise ValueError("CITATIONS_MAX_K must be at least 1")
        if self.llm_provider == "openrouter" and not self.openrouter_model:
            raise ValueError("OPENROUTER_MODEL must not be empty when LLM_PROVIDER=openrouter")
        if self.llm_provider == "ollama" and not self.ollama_model:
            raise ValueError("OLLAMA_MODEL must not be empty when LLM_PROVIDER=ollama")
        return self

    @property
    def retrieval_lexical_k(self) -> int:
        """Use CITATIONS_MAX_K as the unified retrieval depth for lexical retrieval."""
        return int(self.citations_max_k)

    @property
    def retrieval_semantic_k(self) -> int:
        """Use CITATIONS_MAX_K as the unified retrieval depth for semantic retrieval."""
        return int(self.citations_max_k)

    @property
    def citations_default_k(self) -> int:
        """Use CITATIONS_MAX_K as the default citation count."""
        return int(self.citations_max_k)

    def provider_name(self) -> str:
        return "OpenRouter" if self.llm_provider == "openrouter" else "Ollama"

    def embedding_provider_name(self) -> str:
        return "OpenRouter" if self.embedding_provider == "openrouter" else "Ollama"

    def active_llm_model(self) -> str:
        return self.openrouter_model if self.llm_provider == "openrouter" else self.ollama_model

    def active_embedding_model(self) -> str:
        if self.embedding_provider == "openrouter":
            return self.openrouter_embedding_model.strip() or self.openrouter_model
        return self.ollama_embedding_model.strip() or self.ollama_model

    def _parse_allowed_models(self, raw: str, default_model: str) -> list[str]:
        raw = raw.strip()
        if not raw:
            return [default_model]

        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    values = [str(item).strip() for item in parsed if str(item).strip()]
                    return values or [default_model]
            except json.JSONDecodeError:
                pass

        values = [item.strip() for item in raw.split(",") if item.strip()]
        return values or [default_model]

    def allowed_models(self) -> list[str]:
        """Parse provider-specific allowed models from CSV or JSON list."""
        if self.llm_provider == "openrouter":
            return self._parse_allowed_models(self.openrouter_allowed_models_raw, self.openrouter_model)
        return self._parse_allowed_models(self.ollama_allowed_models_raw, self.ollama_model)

    def default_llm_settings(self) -> dict[str, Any]:
        return {
            "model": self.active_llm_model(),
            "temperature": self.llm_temperature,
            "top_p": self.llm_default_top_p,
        }


def require_llm_provider_configuration(settings: Settings) -> None:
    """Fail-fast guard for provider-specific required configuration."""
    uses_openrouter = settings.llm_provider == "openrouter" or settings.embedding_provider == "openrouter"
    if uses_openrouter and not settings.openrouter_api_key.strip():
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in .env before starting the backend.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings once per process."""
    load_dotenv(override=False)
    return Settings()
