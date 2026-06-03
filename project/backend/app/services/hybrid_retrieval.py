from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    lexical_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    lexical_weight: float,
    semantic_weight: float,
    top_k: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Fuse retrieval results with weighted reciprocal rank fusion."""
    fused_scores: dict[str, float] = defaultdict(float)
    merged_rows: dict[str, dict[str, Any]] = {}

    for rank, row in enumerate(lexical_results, start=1):
        chunk_id = str(row["chunk_id"])
        fused_scores[chunk_id] += lexical_weight / (rank_constant + rank)
        merged_rows.setdefault(chunk_id, row)

    for rank, row in enumerate(semantic_results, start=1):
        chunk_id = str(row["chunk_id"])
        fused_scores[chunk_id] += semantic_weight / (rank_constant + rank)
        if chunk_id not in merged_rows:
            merged_rows[chunk_id] = row

    ranked_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    fused: list[dict[str, Any]] = []
    for chunk_id in ranked_chunk_ids[:top_k]:
        row = dict(merged_rows[chunk_id])
        row["fused_score"] = fused_scores[chunk_id]
        row["source"] = "hybrid"
        fused.append(row)
    return fused


def build_diagnostics(
    lexical_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    fused_results: list[dict[str, Any]],
) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for row in rows[:5]:
            summary.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "filename": row.get("filename"),
                    "page": row.get("page"),
                    "score": row.get("fused_score", row.get("score")),
                }
            )
        return summary

    return {
        "lexical_hits": summarize(lexical_results),
        "semantic_hits": summarize(semantic_results),
        "fused_hits": summarize(fused_results),
    }
