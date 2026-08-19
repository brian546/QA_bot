from __future__ import annotations

import math
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

from project.backend.app.core.config import Settings


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 0:
        return vector
    return [value / magnitude for value in vector]


class LocalMultimodalEmbeddings(Embeddings):
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = max(16, int(dimension))

    def _hash_to_vector(self, values: list[bytes]) -> list[float]:
        seed = bytearray()
        for value in values:
            seed.extend(value)
        if not seed:
            seed.extend(b"0")

        vector = [0.0] * self.dimension
        for idx, byte in enumerate(seed):
            slot = idx % self.dimension
            vector[slot] += ((byte % 97) - 48) / 50.0
        return _normalize_vector(vector)

    def embed_text(self, text: str) -> list[float]:
        return self._hash_to_vector([text.encode("utf-8", errors="ignore")])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_image(self, image_data_url: str) -> list[float]:
        return self._hash_to_vector([image_data_url.encode("utf-8", errors="ignore")])


class OpenRouterMultimodalEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str, base_url: str, timeout: float, max_retries: int) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    def _request_embeddings(self, inputs: list[Any]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": inputs}

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    body = response.json()

                data = body.get("data")
                if not isinstance(data, list):
                    raise RuntimeError("OpenRouter embeddings response is missing 'data'")

                ordered = sorted([row for row in data if isinstance(row, dict)], key=lambda row: int(row.get("index", 0)))
                vectors: list[list[float]] = []
                for row in ordered:
                    embedding = row.get("embedding")
                    if not isinstance(embedding, list):
                        raise RuntimeError("OpenRouter embedding vector has invalid format")
                    vectors.append([float(value) for value in embedding])
                if len(vectors) != len(inputs):
                    raise RuntimeError("OpenRouter embeddings count does not match requested inputs")
                return vectors
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Failed to create multimodal embeddings from OpenRouter model '{self.model}'") from last_error

    def embed_text(self, text: str) -> list[float]:
        return self._request_embeddings([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._request_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_image(self, image_data_url: str) -> list[float]:
        payload = {
            "type": "image_url",
            "image_url": {"url": image_data_url},
        }
        return self._request_embeddings([payload])[0]


def get_multimodal_embeddings(settings: Settings) -> LocalMultimodalEmbeddings | OpenRouterMultimodalEmbeddings:
    if settings.embedding_provider == "openrouter" and settings.openrouter_api_key.strip():
        return OpenRouterMultimodalEmbeddings(
            api_key=settings.openrouter_api_key,
            model=settings.active_image_embedding_model(),
            base_url=settings.openrouter_base_url,
            timeout=float(settings.openrouter_timeout),
            max_retries=settings.openrouter_max_retries,
        )
    return LocalMultimodalEmbeddings(dimension=settings.embedding_dimension)



