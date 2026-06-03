from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from project.backend.app.core.config import Settings
from project.backend.app.core.llm import get_chat_model


REWRITE_SYSTEM = (
    "Rewrite the user question into a standalone retrieval query. "
    "Preserve names, IDs, product codes, quotes, and dates exactly when present."
)

COMPRESS_SYSTEM = (
    "Select the minimum evidence needed to answer the question. "
    "Return concise bullet points, each prefixed with [chunk_id]."
)

ANSWER_SYSTEM = (
    "You are a grounded QA assistant. Use only the provided evidence. "
    "If evidence is insufficient, say so clearly. Always include citations [filename p.X chunk_id]."
)


def rewrite_query_with_history(
    settings: Settings,
    question: str,
    chat_history: list[dict[str, str]],
    llm_settings: dict[str, Any],
) -> str:
    """Rewrite query into standalone form using OpenRouter when available."""
    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history[-6:])
    prompt = f"Chat history:\n{history_text}\n\nQuestion:\n{question}\n\nStandalone query:"
    try:
        model = get_chat_model(settings, llm_settings)
        response = model.invoke([SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=prompt)])
        rewritten = str(response.content).strip()
        return rewritten or question
    except Exception:
        return question


def compress_evidence(
    settings: Settings,
    question: str,
    fused_rows: list[dict[str, Any]],
    llm_settings: dict[str, Any],
) -> str:
    """Compress fused evidence into a shorter context while preserving attribution."""
    if not fused_rows:
        return ""

    evidence_block = "\n\n".join(
        f"[{row.get('chunk_id')}] {row.get('text', '')}" for row in fused_rows[:8]
    )
    prompt = f"Question:\n{question}\n\nEvidence:\n{evidence_block}\n\nCompressed evidence:"
    try:
        model = get_chat_model(settings, llm_settings)
        response = model.invoke([SystemMessage(content=COMPRESS_SYSTEM), HumanMessage(content=prompt)])
        return str(response.content).strip()
    except Exception:
        # Deterministic fallback keeps system useful offline.
        return "\n".join(f"[{row.get('chunk_id')}] {str(row.get('text', ''))[:240]}" for row in fused_rows[:6])


def answer_with_evidence(
    settings: Settings,
    question: str,
    compressed_context: str,
    fused_rows: list[dict[str, Any]],
    llm_settings: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], float]:
    """Generate answer and citations grounded in retrieved evidence."""
    citations = [
        {
            "chunk_id": row.get("chunk_id"),
            "filename": row.get("filename"),
            "page": row.get("page"),
            "section": row.get("section"),
        }
        for row in fused_rows[:5]
    ]

    if not compressed_context.strip():
        return "I could not find enough evidence in the uploaded PDFs.", [], 0.1

    prompt = (
        f"Question:\n{question}\n\n"
        f"Compressed evidence:\n{compressed_context}\n\n"
        "Answer strictly from evidence. If uncertain, say evidence is insufficient."
    )
    try:
        model = get_chat_model(settings, llm_settings)
        response = model.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=prompt)])
        answer = str(response.content).strip()
    except Exception:
        answer = "Based on the retrieved evidence, here is the most likely answer:\n" + compressed_context[:1200]

    confidence = min(0.95, 0.35 + 0.1 * len(citations)) if citations else 0.1
    return answer, citations, confidence
