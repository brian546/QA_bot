from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-./]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def build_bm25_index(chunks: list[dict[str, Any]]) -> tuple[BM25Okapi | None, list[list[str]]]:
    tokenized = [tokenize(str(chunk["text"])) for chunk in chunks]
    if not tokenized:
        return None, []
    return BM25Okapi(tokenized), tokenized


def retrieve_lexical(
    query: str,
    chunks: list[dict[str, Any]],
    bm25: BM25Okapi | None,
    top_k: int,
) -> list[dict[str, Any]]:
    if bm25 is None or not chunks:
        return []

    scores = bm25.get_scores(tokenize(query)).tolist()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    results: list[dict[str, Any]] = []
    for idx, score in ranked[:top_k]:
        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "page": chunk["page"],
                "section": chunk.get("section"),
                "text": chunk["text"],
                "score": float(score),
                "source": "lexical",
            }
        )
    return results
