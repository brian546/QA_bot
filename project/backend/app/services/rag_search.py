from __future__ import annotations

from typing import Any

from project.backend.app.core.config import Settings
from project.backend.app.core.session_store import InMemorySessionStore
from project.backend.app.services.hybrid_retrieval import build_diagnostics, reciprocal_rank_fusion
from project.backend.app.services.image_retrieval import retrieve_image_assets
from project.backend.app.services.lexical_retrieval import retrieve_lexical
from project.backend.app.services.semantic_retrieval import retrieve_semantic


def _resolve_weights(settings: Settings, overrides: dict[str, Any] | None) -> tuple[float, float]:
    values = overrides or {}
    lexical = max(0.0, min(1.0, float(values.get("lexical_weight", settings.lexical_weight))))
    semantic = max(0.0, min(1.0, float(values.get("semantic_weight", settings.semantic_weight))))
    total = lexical + semantic
    if total <= 0:
        return 0.5, 0.5
    return lexical / total, semantic / total


def rag_search(
    settings: Settings,
    session_store: InMemorySessionStore,
    query: str,
    session_id: str,
    retrieval_settings: dict[str, Any] | None = None,
    citation_limit: int | None = None,
) -> dict[str, Any]:
    """Search one uploaded-document session and return agent-ready evidence."""
    session = session_store.get(session_id)
    if session is None or (not session.uploaded_documents and not session.image_assets):
        return {
            "query": query,
            "status": "no_documents",
            "evidence": "",
            "sources": [],
            "images": [],
            "diagnostics": {"status": "no_documents"},
        }

    if not query.strip():
        return {
            "query": query,
            "status": "no_results",
            "evidence": "",
            "sources": [],
            "images": [],
            "diagnostics": {"status": "no_results"},
        }

    top_k = max(1, int(citation_limit or settings.citations_max_k))
    lexical_weight, semantic_weight = _resolve_weights(settings, retrieval_settings)
    try:
        lexical_results = retrieve_lexical(query, session.chunks, session.lexical_index, top_k)
        semantic_results = retrieve_semantic(query, session.semantic_index, top_k)
        image_results = retrieve_image_assets(query, session.image_index, settings, top_k)
        if not image_results and session.image_assets and (not session.chunks or any(
            token in query.lower()
            for token in ("image", "photo", "picture", "screenshot", "figure", "diagram", "chart", "visual")
        )):
            image_results = [
                dict(asset, score=0.0, source="image", modality="image")
                for asset in session.image_assets[:top_k]
            ]
        fused_results = reciprocal_rank_fusion(
            lexical_results,
            semantic_results,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            top_k=top_k,
            image_results=image_results,
        )
    except Exception as exc:
        return {
            "query": query,
            "status": "error",
            "evidence": "",
            "sources": [],
            "images": [],
            "diagnostics": {"status": "error", "message": str(exc)},
        }

    if not fused_results:
        return {
            "query": query,
            "status": "no_results",
            "evidence": "",
            "sources": [],
            "images": [],
            "diagnostics": {
                "status": "no_results",
                **build_diagnostics(lexical_results, semantic_results, [], top_k, image_results),
            },
        }

    from project.backend.app.services.qa import compress_evidence

    evidence = compress_evidence(settings, query, fused_results, {})
    sources = [
        {
            "chunk_id": row.get("chunk_id"),
            "filename": row.get("filename"),
            "page": row.get("page"),
            "section": row.get("section"),
            "modality": row.get("modality", "text"),
            "asset_id": row.get("asset_id"),
            "storage_uri": row.get("storage_uri"),
        }
        for row in fused_results[:top_k]
    ]
    images = [
        {
            "asset_id": row.get("asset_id"),
            "filename": row.get("filename"),
            "page": row.get("page"),
            "image_data_url": row.get("image_data_url"),
        }
        for row in fused_results[:top_k]
        if str(row.get("image_data_url", "")).startswith("data:image/")
    ]
    return {
        "query": query,
        "status": "ok",
        "evidence": evidence,
        "sources": sources,
        "images": images,
        "diagnostics": {
            "status": "ok",
            **build_diagnostics(lexical_results, semantic_results, fused_results, top_k, image_results),
        },
        "fused_results": fused_results,
    }