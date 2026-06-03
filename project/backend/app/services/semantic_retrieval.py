from __future__ import annotations

import hashlib
import math
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """Simple deterministic embeddings for MVP and local tests."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % self.dimension
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


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


def build_faiss_index(chunks: list[dict[str, Any]], embedding_dim: int) -> FAISS | None:
    if not chunks:
        return None
    docs = build_documents(chunks)
    embeddings = HashEmbeddings(dimension=embedding_dim)
    return FAISS.from_documents(docs, embeddings)


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
                "score": float(score),
                "source": "semantic",
            }
        )
    return results
