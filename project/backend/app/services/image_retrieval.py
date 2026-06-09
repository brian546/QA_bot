from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import httpx

from project.backend.app.core.config import Settings


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 0:
        return vector
    return [value / magnitude for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    numerator = sum(left[idx] * right[idx] for idx in range(size))
    left_mag = math.sqrt(sum(value * value for value in left[:size]))
    right_mag = math.sqrt(sum(value * value for value in right[:size]))
    if left_mag <= 0 or right_mag <= 0:
        return 0.0
    return numerator / (left_mag * right_mag)


class LocalMultimodalEmbeddings:
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

    def embed_image(self, image_data_url: str) -> list[float]:
        return self._hash_to_vector([image_data_url.encode("utf-8", errors="ignore")])


class OpenRouterMultimodalEmbeddings:
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


@dataclass
class ImageVectorIndex:
    embeddings: list[list[float]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    embedding_model: str = ""
    backend: str = "memory"

    def is_empty(self) -> bool:
        return not self.embeddings or not self.records


def build_image_index(image_assets: list[dict[str, Any]], settings: Settings) -> ImageVectorIndex | None:
    if not image_assets:
        return None

    embeddings_client = get_multimodal_embeddings(settings)
    vectors: list[list[float]] = []
    records: list[dict[str, Any]] = []

    for asset in image_assets:
        image_data_url = str(asset.get("image_data_url", ""))
        if not image_data_url:
            continue
        try:
            vector = embeddings_client.embed_image(image_data_url)
        except Exception:
            vector = embeddings_client.embed_text(str(asset.get("text", "")))
        vectors.append(vector)
        records.append(dict(asset))

    if not vectors:
        return None
    return ImageVectorIndex(embeddings=vectors, records=records, embedding_model=settings.active_image_embedding_model())


def retrieve_image_assets(query: str, index: ImageVectorIndex | None, settings: Settings, top_k: int) -> list[dict[str, Any]]:
    if index is None or index.is_empty() or not query.strip():
        return []

    embeddings_client = get_multimodal_embeddings(settings)
    try:
        query_vector = embeddings_client.embed_text(query)
    except Exception:
        return []

    scored: list[tuple[int, float]] = []
    for idx, vector in enumerate(index.embeddings):
        scored.append((idx, _cosine_similarity(query_vector, vector)))

    scored.sort(key=lambda item: item[1], reverse=True)

    results: list[dict[str, Any]] = []
    for idx, score in scored[: max(1, int(top_k))]:
        record = dict(index.records[idx])
        record.update({"score": float(score), "source": "image", "modality": "image"})
        results.append(record)
    return results


def build_image_diagnostics(results: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in results[: max(1, int(top_k))]:
        summary.append(
            {
                "chunk_id": row.get("chunk_id"),
                "filename": row.get("filename"),
                "page": row.get("page"),
                "score": row.get("score"),
                "modality": row.get("modality", "image"),
            }
        )
    return summary
