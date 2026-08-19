from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from project.backend.app.core.config import Settings, get_settings
from project.backend.app.services.image_retrieval import get_multimodal_embeddings


logger = logging.getLogger(__name__)


class OpenRouterEmbeddings(Embeddings):
    """OpenRouter embeddings client using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        max_retries: int,
        batch_size: int = 32,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.batch_size = max(1, batch_size)

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenRouterEmbeddings:
        return cls(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_embedding_model,
            base_url=settings.openrouter_base_url,
            timeout=float(settings.openrouter_timeout),
            max_retries=settings.openrouter_max_retries,
        )

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}

        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    body = response.json()

                data = body.get("data")
                if not isinstance(data, list):
                    raise RuntimeError("OpenRouter embeddings response is missing 'data' list")

                indexed = sorted(
                    [row for row in data if isinstance(row, dict)],
                    key=lambda row: int(row.get("index", 0)),
                )
                embeddings = [row.get("embedding") for row in indexed]
                if len(embeddings) != len(texts):
                    raise RuntimeError("OpenRouter embeddings count does not match requested inputs")

                vectors: list[list[float]] = []
                for embedding in embeddings:
                    if not isinstance(embedding, list):
                        raise RuntimeError("OpenRouter embedding vector has invalid format")
                    vectors.append([float(value) for value in embedding])
                return vectors
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Failed to create embeddings from OpenRouter model '{self.model}'") from last_error

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for idx in range(0, len(texts), self.batch_size):
            batch = texts[idx : idx + self.batch_size]
            vectors.extend(self._request_embeddings(batch))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed_many([text])
        return vectors[0] if vectors else []


def build_documents(chunks: list[dict[str, Any]]) -> list[Document]:
    docs: list[Document] = []
    for chunk in chunks:
        docs.append(
            Document(
                page_content=str(chunk["text"]),
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "section": chunk.get("section"),
                },
            )
        )
    return docs


def get_embeddings_client(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.active_embedding_model(),
            base_url=settings.ollama_base_url,
        )
    return OpenRouterEmbeddings.from_settings(settings)


def build_faiss_index(
    chunks: list[dict[str, Any]],
    embedding_dim: int,
    image_assets: list[dict[str, Any]] | None = None,
) -> FAISS | None:
    if not chunks and not image_assets:
        return None
    _ = embedding_dim
    settings = get_settings()
    embeddings = get_multimodal_embeddings(settings)
    try:
        texts = [str(chunk["text"]) for chunk in chunks]
        vectors = embeddings.embed_documents(texts)
        metadatas = [
            {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "page": chunk["page"],
                "section": chunk.get("section"),
                "modality": "text",
            }
            for chunk in chunks
        ]

        if image_assets:
            multimodal_embeddings = get_multimodal_embeddings(settings)
            image_inputs: list[str] = []
            image_metadata: list[dict[str, Any]] = []
            for asset in image_assets:
                image_data_url = str(asset.get("image_data_url", ""))
                if not image_data_url:
                    continue
                image_inputs.append(image_data_url)
                image_metadata.append(
                    {
                        "chunk_id": asset.get("chunk_id") or asset.get("asset_id"),
                        "asset_id": asset.get("asset_id"),
                        "filename": asset.get("filename"),
                        "page": asset.get("page"),
                        "section": asset.get("section"),
                        "modality": "image",
                        "image_data_url": image_data_url,
                        "storage_uri": asset.get("storage_uri"),
                    }
                )
            if image_inputs:
                vectors.extend([multimodal_embeddings.embed_image(value) for value in image_inputs])
                texts.extend([str(metadata.get("filename", "image")) for metadata in image_metadata])
                metadatas.extend(image_metadata)

        return FAISS.from_embeddings(list(zip(texts, vectors)), embeddings, metadatas=metadatas)
    except Exception as exc:
        logger.warning("Semantic index build failed; using lexical retrieval only: %s", exc)
        return None


def retrieve_semantic(query: str, index: FAISS | None, top_k: int) -> list[dict[str, Any]]:
    if index is None:
        return []
    docs_and_scores = index.similarity_search_with_score(query, k=top_k)
    results: list[dict[str, Any]] = []
    for doc, score in docs_and_scores:
        metadata = doc.metadata
        results.append(
            {
                "chunk_id": metadata.get("chunk_id"),
                "filename": metadata.get("filename"),
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "text": doc.page_content,
                "asset_id": metadata.get("asset_id"),
                "modality": metadata.get("modality", "text"),
                "image_data_url": metadata.get("image_data_url"),
                "storage_uri": metadata.get("storage_uri"),
                "score": float(score),
                "source": "semantic",
            }
        )
    return results
