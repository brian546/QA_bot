from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from project.backend.app.core.config import Settings
from project.backend.app.core.llm import get_chat_model

import logging

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

IMAGE_KEYWORDS = (
    "image",
    "photo",
    "picture",
    "screenshot",
    "figure",
    "diagram",
    "chart",
    "illustration",
    "visual",
    "look at",
    "show me",
)

ROUTER_SYSTEM = (
    "You are a routing classifier. Decide if a user question requires searching uploaded documents. "
    "Respond with exactly one token: SEARCH or DIRECT. "
    "Choose SEARCH when the answer likely depends on document-specific evidence. "
    "Choose DIRECT for general knowledge or chit-chat not dependent on uploaded files."
)

DIRECT_SYSTEM = (
    "You are a helpful assistant. Answer directly and briefly. "
    "If the user asks about specific uploaded documents and none are available, say they should upload files first."
)

CONFIDENCE_EVAL_SYSTEM = (
    "You are an answer confidence evaluator for grounded QA. "
    "Return exactly one token: CONFIDENT or NOT_CONFIDENT. "
    "Choose CONFIDENT only when the answer is clearly supported by the provided evidence and citations. "
    "Choose NOT_CONFIDENT when evidence is weak, missing, contradictory, or the answer is speculative."
)


def _build_image_human_content(
    prompt: str,
    rows: list[dict[str, Any]],
    effective_limit: int,
    llm_provider: str,
    intro_suffix: str,
) -> list[dict[str, Any]]:
    """Build multimodal human content by attaching raw image payloads with evidence labels."""
    human_content: list[dict[str, Any]] = [{"type": "text", "text": prompt + intro_suffix}]

    for row in rows[:effective_limit]:
        image_data_url = str(row.get("image_data_url", ""))
        if not image_data_url.startswith("data:image/"):
            continue

        human_content.append(
            {
                "type": "text",
                "text": (
                    f"Image evidence: {row.get('filename', 'unknown')} p.{row.get('page', '?')} "
                    f"{row.get('asset_id') or row.get('chunk_id', '')}"
                ),
            }
        )
        if llm_provider == "ollama":
            human_content.append({"type": "image_url", "image_url": image_data_url})
        else:
            human_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    return human_content


def rewrite_query_with_history(
    settings: Settings,
    question: str,
    chat_history: list[dict[str, str]],
    llm_settings: dict[str, Any],
) -> str:
    """Rewrite the user question into a standalone query that can be answered without chat history, if needed for retrieval."""
    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history)
    prompt = f"Chat history:\n{history_text}\n\nQuestion:\n{question}\n\nStandalone query:"
    try:
        model = get_chat_model(settings, llm_settings)
        response = model.invoke([SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=prompt)])
        rewritten = str(response.content).strip()
        return rewritten or question
    except Exception as e:
        logging.error(f"Error in rewrite_query_with_history: {e}")
    return question


def should_search_documents(
    settings: Settings,
    question: str,
    chat_history: list[dict[str, str]],
    docs_available: bool,
    llm_settings: dict[str, Any],
) -> bool:
    """Decide if retrieval over uploaded documents is needed for this query."""
    if not docs_available:
        return False

    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history)
    prompt = (
        f"Documents available: {docs_available}\n"
        f"Recent history:\n{history_text}\n\n"
        f"Question:\n{question}\n\n"
        "Return SEARCH or DIRECT."
    )

    router_settings = dict(llm_settings or {})
    router_settings["temperature"] = 0

    try:
        model = get_chat_model(settings, router_settings)
        response = model.invoke([SystemMessage(content=ROUTER_SYSTEM), HumanMessage(content=prompt)])
        decision = str(response.content).strip().upper()
        if "SEARCH" in decision:
            return True
        if "DIRECT" in decision:
            return False
    except Exception as e:
        logging.error(f"Error in should_search_documents: {e}")

    # Conservative heuristic fallback if router model is unavailable.
    lowered = question.lower()
    keywords = ("pdf", "document", "uploaded", "file", "page", "policy", "contract", "report", *IMAGE_KEYWORDS)
    return any(token in lowered for token in keywords)


def answer_directly(
    settings: Settings,
    question: str,
    chat_history: list[dict[str, str]],
    llm_settings: dict[str, Any],
) -> str:
    """Answer without retrieval when query does not require document search."""
    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history)
    prompt = (
        f"Chat history:\n{history_text}\n\n"
        f"Question:\n{question}\n\n"
        "Provide a concise direct answer."
    )

    try:
        model = get_chat_model(settings, llm_settings)
        response = model.invoke([SystemMessage(content=DIRECT_SYSTEM), HumanMessage(content=prompt)])
        answer = str(response.content).strip()
        if answer:
            return answer
    except Exception as e:
        logging.error(f"Error in answer_directly: {e}")

    return "Chat model is unavailable. Try again later."


def compress_evidence(
    settings: Settings,
    question: str,
    fused_rows: list[dict[str, Any]],
    llm_settings: dict[str, Any],
) -> str:
    """Compress fused evidence into a shorter context while preserving attribution."""
    if not fused_rows:
        return ""

    evidence_block = "\n\n".join(f"[{row.get('chunk_id')}] {row.get('text', '')}" for row in fused_rows)
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
    citation_limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate answer and citations grounded in retrieved evidence."""
    effective_limit = max(1, int(citation_limit))
    citations = [
        {
            "chunk_id": row.get("chunk_id"),
            "filename": row.get("filename"),
            "page": row.get("page"),
            "section": row.get("section"),
            "asset_id": row.get("asset_id"),
            "modality": row.get("modality", "text"),
            "image_data_url": row.get("image_data_url"),
            "storage_uri": row.get("storage_uri"),
        }
        for row in fused_rows[:effective_limit]
    ]

    has_image_evidence = any(str(row.get("image_data_url", "")).startswith("data:image/") for row in fused_rows)
    if not compressed_context.strip() and not has_image_evidence:
        return "I could not find enough evidence in the uploaded documents.", []

    prompt = (
        f"Question:\n{question}\n\n"
        f"Compressed evidence:\n{compressed_context}\n\n"
        "Answer strictly from evidence. If uncertain, say evidence is insufficient."
    )
    try:
        model = get_chat_model(settings, llm_settings)

        if has_image_evidence:
            human_content = _build_image_human_content(
                prompt,
                fused_rows,
                effective_limit,
                settings.llm_provider,
                "\n\nInspect the attached images directly before answering.",
            )
            response = model.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=human_content)])
        else:
            response = model.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=prompt)])
        answer = str(response.content).strip()
    except Exception as e:
        logging.error(f"Error in answer_with_evidence: {e}")
        answer = "Based on the retrieved evidence, here is the most likely answer:\n" + compressed_context[:1200]

    return answer, citations


def is_answer_confident(
    settings: Settings,
    question: str,
    answer: str,
    compressed_context: str,
    citations: list[dict[str, Any]],
    llm_settings: dict[str, Any] | None,
) -> bool:
    """Use the model to classify whether the grounded answer is confident."""
    has_image_citation = any(str(c.get("modality", "")).lower() == "image" for c in citations)
    if not answer.strip() or not citations or (not compressed_context.strip() and not has_image_citation):
        return False

    effective_limit = max(1, min(8, len(citations)))
    prompt = (
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Compressed evidence:\n{compressed_context}\n\n"
        "Return CONFIDENT or NOT_CONFIDENT."
    )

    confidence_settings = dict(llm_settings or {})
    confidence_settings["temperature"] = 0

    try:
        model = get_chat_model(settings, confidence_settings)
        if has_image_citation:
            human_content = _build_image_human_content(
                prompt,
                citations,
                effective_limit,
                settings.llm_provider,
                "\n\nUse the compressed text evidence and inspect the attached images before deciding.",
            )
            response = model.invoke([SystemMessage(content=CONFIDENCE_EVAL_SYSTEM), HumanMessage(content=human_content)])
        else:
            response = model.invoke([SystemMessage(content=CONFIDENCE_EVAL_SYSTEM), HumanMessage(content=prompt)])
        verdict = str(response.content).strip().upper()
        if "NOT_CONFIDENT" in verdict:
            return False
        if "CONFIDENT" in verdict:
            return True
    except Exception as e:
        logging.error(f"Error in is_answer_confident: {e}")

    lowered = answer.lower()
    refusal_markers = (
        "insufficient",
        "not enough evidence",
        "could not find enough evidence",
        "uncertain",
    )
    return not any(marker in lowered for marker in refusal_markers)
