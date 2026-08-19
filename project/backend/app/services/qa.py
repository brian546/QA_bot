from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage

from project.backend.app.core.config import Settings
from project.backend.app.core.llm import get_chat_model
from project.backend.app.services.web_search import search_web_with_tavily

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

WEB_ROUTER_SYSTEM = (
    "You are a web-search routing classifier. Decide if a question needs current web information. "
    "Return exactly one token: WEB_SEARCH or NO_WEB_SEARCH. "
    "Choose WEB_SEARCH for time-sensitive, current-events, up-to-date facts, recent prices, "
    "live status, or when specific evidence is likely not present in local context. "
    "Choose NO_WEB_SEARCH for timeless facts, opinions, or questions answerable without current web data."
)

WEB_ANSWER_SYSTEM = (
    "You are a helpful assistant using web snippets as evidence. "
    "Answer with concise factual statements and cite web sources inline as Markdown links using the exact source title and URL. "
    "If sources are weak or missing, say uncertainty clearly."
)

AGENT_DECISION_SYSTEM = (
    "You are qa_agent, an answer planning agent for grounded QA. "
    "Choose exactly one action: rag_search, web_search, or answer. "
    "Use rag_search for uploaded-document evidence, web_search for external/current evidence, "
    "and answer when the available evidence is sufficient. "
    "When uploaded session content exists but available evidence is empty, inspect it with rag_search before answering. "
    "Return JSON only with exactly these fields: action, query, session_id, reason."
)

AGENT_DECISION_RETRY_SYSTEM = (
    "Return only valid JSON, with no markdown or explanation, using exactly these fields: "
    "action, query, session_id, reason. action must be rag_search, web_search, or answer."
)

TIME_SENSITIVE_KEYWORDS = (
    "today",
    "now",
    "current",
    "latest",
    "recent",
    "this week",
    "this month",
    "this year",
    "as of",
    "news",
    "price",
    "weather",
    "stock",
    "exchange rate",
    "breaking",
)

logger = logging.getLogger(__name__)


def _direct_answer_unavailable_message(settings: Settings, error: Exception) -> str:
    """Build a user-facing fallback message for direct-answer failures."""
    generic = "Chat model is unavailable. Try again later."
    if settings.llm_provider != "openrouter":
        return generic

    error_text = str(error).strip().lower()
    error_type = type(error).__name__.lower()

    if "toomanyrequests" in error_type or "rate limit" in error_text or "429" in error_text:
        return (
            "OpenRouter is rate-limited right now. "
            "Try again in a minute, switch to another OpenRouter model, or use Ollama."
        )

    if "provider returned error" in error_text or "no endpoints found" in error_text:
        return (
            "OpenRouter is temporarily unavailable for the selected model. "
            "Try again later or choose a different OpenRouter model."
        )

    return generic

def _build_image_human_content(
    prompt: str,
    rows: list[dict[str, Any]],
    effective_limit: int,
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
        logger.error(f"Error in rewrite_query_with_history: {e}")
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
        logger.error(f"Error in should_search_documents: {e}")

    # Conservative heuristic fallback if router model is unavailable.
    lowered = question.lower()
    keywords = ("pdf", "document", "uploaded", "file", "page", "policy", "contract", "report", *IMAGE_KEYWORDS)
    return any(token in lowered for token in keywords)


def should_search_web(
    settings: Settings,
    question: str,
    chat_history: list[dict[str, str]],
    llm_settings: dict[str, Any],
    allow_model_inference: bool = False,
) -> bool:
    """Decide whether the question should trigger a live web search."""
    if not settings.web_search_enabled:
        return False

    lowered = question.lower()
    if any(token in lowered for token in TIME_SENSITIVE_KEYWORDS):
        return True

    if not allow_model_inference:
        return False

    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history)
    today = datetime.now(UTC).date().isoformat()
    prompt = (
        f"Today's date: {today}\n"
        f"Recent history:\n{history_text}\n\n"
        f"Question:\n{question}\n\n"
        "Return WEB_SEARCH or NO_WEB_SEARCH."
    )

    router_settings = dict(llm_settings or {})
    router_settings["temperature"] = 0

    try:
        model = get_chat_model(settings, router_settings)
        response = model.invoke([SystemMessage(content=WEB_ROUTER_SYSTEM), HumanMessage(content=prompt)])
        decision = str(response.content).strip().upper()
        if "WEB_SEARCH" in decision and "NO_WEB_SEARCH" not in decision:
            return True
        if "NO_WEB_SEARCH" in decision:
            return False
    except Exception as e:
        logger.error(f"Error in should_search_web: {e}")

    return False


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
        logger.error("Error in answer_directly: %s", e, exc_info=True)
        return _direct_answer_unavailable_message(settings, e)

    return "Chat model is unavailable. Try again later."


def _parse_agent_action(content: str) -> dict[str, Any] | None:
    """Parse a structured qa_agent action, including JSON wrapped in markdown fences."""
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("action") not in {"rag_search", "web_search", "answer"}:
        return None
    return {
        "action": parsed["action"],
        "query": str(parsed.get("query", "")).strip(),
        "session_id": str(parsed.get("session_id", "")).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def _agent_action(
    settings: Settings,
    question: str,
    chat_history: list[dict[str, str]],
    session_id: str,
    evidence: str,
    session_content: dict[str, Any],
    llm_settings: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Ask qa_agent for its next action, retrying malformed JSON once."""
    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history[-6:])
    prompt = (
        f"Question:\n{question}\n\n"
        f"Recent history:\n{history_text}\n\n"
        f"Available evidence:\n{evidence or '(none)'}\n\n"
        f"Current session ID: {session_id}\n"
        f"Uploaded session content: {json.dumps(session_content, ensure_ascii=True)}\n"
        "Choose the next action."
    )
    try:
        model = get_chat_model(settings, llm_settings)
        response = model.invoke([SystemMessage(content=AGENT_DECISION_SYSTEM), HumanMessage(content=prompt)])
        action = _parse_agent_action(str(response.content))
        if action is not None:
            return action, False

        retry = model.invoke(
            [
                SystemMessage(content=AGENT_DECISION_RETRY_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        action = _parse_agent_action(str(retry.content))
        if action is not None:
            return action, True
    except Exception as exc:
        logger.error("Error in agent web-search decision: %s", exc)
    return {"action": "answer", "query": "", "session_id": session_id, "reason": "Agent decision was unavailable."}, True


def _merge_web_results(results: list[dict[str, Any]], new_results: list[dict[str, Any]]) -> None:
    """Append new web results while deduplicating by URL."""
    known_urls = {str(row.get("url", "")).strip() for row in results if str(row.get("url", "")).strip()}
    for row in new_results:
        url = str(row.get("url", "")).strip()
        if url and url in known_urls:
            continue
        results.append(row)
        if url:
            known_urls.add(url)


def _markdown_link_citations(answer: str) -> list[dict[str, Any]]:
    """Extract direct Markdown links from an answer for frontend citation display."""
    citations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", answer):
        if url in seen_urls:
            continue
        seen_urls.add(url)
        citations.append(
            {
                "modality": "web",
                "source": urlparse(url).netloc or "web",
                "title": title.strip() or url,
                "url": url,
            }
        )
    return citations


def answer_with_agent(
    settings: Settings,
    session_store: Any,
    question: str,
    chat_history: list[dict[str, str]],
    session_id: str,
    llm_settings: dict[str, Any],
    retrieval_settings: dict[str, Any] | None = None,
    citation_limit: int | None = None,
    session_content: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Plan, use RAG/web tools, and answer a question through one agent controller."""
    from project.backend.app.services.rag_search import rag_search

    evidence_parts: list[str] = []
    local_rows: list[dict[str, Any]] = []
    web_results: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []
    seen_actions: set[tuple[str, str]] = set()
    action, decision_retry = _agent_action(
        settings,
        question,
        chat_history,
        session_id,
        "",
        session_content or {},
        llm_settings,
    )
    agent_actions: list[dict[str, Any]] = [dict(action)]

    for _ in range(6):
        action_name = str(action.get("action", "answer"))
        if action_name == "answer":
            break

        query = str(action.get("query", "")).strip() or question
        action_key = (action_name, query.casefold())
        if action_key in seen_actions:
            break
        seen_actions.add(action_key)

        if action_name == "rag_search":
            result = rag_search(
                settings,
                session_store,
                query,
                str(action.get("session_id", "")).strip() or session_id,
                retrieval_settings,
                citation_limit,
            )
            tool_trace.append(
                {
                    "action": "rag_search",
                    "query": query,
                    "status": result.get("status"),
                    "source_count": len(result.get("sources", [])),
                }
            )
            if result.get("evidence"):
                evidence_parts.append(str(result["evidence"]))
            local_rows.extend(result.get("fused_results", []))
        elif action_name == "web_search":
            if settings.web_search_enabled and settings.tavily_api_key.strip():
                results = search_web_with_tavily(settings, query)
                _merge_web_results(web_results, results)
                status = "ok" if results else "no_results"
            else:
                results = []
                status = "unavailable"
            tool_trace.append(
                {
                    "action": "web_search",
                    "query": query,
                    "status": status,
                    "result_count": len(results),
                }
            )
            if results:
                evidence_parts.append(
                    "\n\n".join(
                        f"Title: {row.get('title', '')}\nURL: {row.get('url', '')}\nSnippet: {row.get('content', '')}"
                        for row in results
                    )
                )
        else:
            break

        action, retry_used = _agent_action(
            settings,
            question,
            chat_history,
            session_id,
            "\n\n".join(evidence_parts),
            session_content or {},
            llm_settings,
        )
        agent_actions.append(dict(action))
        decision_retry = decision_retry or retry_used

    evidence = "\n\n".join(evidence_parts)
    prompt = (
        f"Question:\n{question}\n\nEvidence:\n{evidence or '(none)'}\n\n"
        "Answer using the evidence. Cite web evidence inline as direct Markdown links [title](url). "
        "Do not use numbered source citations. If evidence is insufficient, say so clearly."
    )
    try:
        model = get_chat_model(settings, llm_settings)
        has_image_evidence = any(
            str(row.get("image_data_url", "")).startswith("data:image/") for row in local_rows
        )
        if has_image_evidence:
            human_content = _build_image_human_content(
                prompt,
                local_rows,
                len(local_rows),
                "\n\nInspect the attached images directly before answering.",
            )
            response = model.invoke([SystemMessage(content=WEB_ANSWER_SYSTEM), HumanMessage(content=human_content)])
        else:
            response = model.invoke([SystemMessage(content=WEB_ANSWER_SYSTEM), HumanMessage(content=prompt)])
        answer = str(response.content).strip()
    except Exception as exc:
        logger.error("Error in agent answer generation: %s", exc)
        answer = "I could not generate a final answer from the available evidence."

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
        for row in local_rows
    ]
    citations.extend(_markdown_link_citations(answer))
    diagnostics = {
        "agent_decisions": agent_actions,
        "agent_decision_retry": decision_retry,
        "tool_trace": tool_trace,
        "rag_searches": [item for item in tool_trace if item["action"] == "rag_search"],
        "web_search_queries": [item["query"] for item in tool_trace if item["action"] == "web_search"],
        "web_hits": web_results,
    }
    return answer, citations, diagnostics


def answer_with_web_results(
    settings: Settings,
    question: str,
    chat_history: list[dict[str, str]],
    web_results: list[dict[str, Any]],
    llm_settings: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Generate an answer grounded in Tavily web results."""
    if not settings.tavily_api_key.strip() or not settings.web_search_enabled:
        return (
            "I do not have web search enabled right now. Please configure Tavily to answer this with live web data.",
            [],
        )

    if not web_results:
        return ("I could not find relevant web results for this question.", [])

    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history[-6:])
    sources_block = "\n\n".join(
        (
            f"[source {idx}] title: {row.get('title', '')}\n"
            f"url: {row.get('url', '')}\n"
            f"snippet: {row.get('content', '')}"
        )
        for idx, row in enumerate(web_results, start=1)
    )
    prompt = (
        f"Recent history:\n{history_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Web sources:\n{sources_block}\n\n"
        "Answer using only these sources. Cite claims directly with Markdown links in the format "
        "[source title](source URL), using the exact title and URL provided above. Do not invent sources."
    )

    try:
        model = get_chat_model(settings, llm_settings)
        response = model.invoke([SystemMessage(content=WEB_ANSWER_SYSTEM), HumanMessage(content=prompt)])
        answer = str(response.content).strip()
    except Exception as e:
        logger.error(f"Error in answer_with_web_results: {e}")
        answer = "I found web results but could not generate a final response from them."

    cited_source_numbers = {
        int(match)
        for match in re.findall(r"\[source\s+(\d+)\]", answer, flags=re.IGNORECASE)
        if int(match) <= len(web_results)
    }
    cited_source_urls = {
        str(row.get("url", "")).strip()
        for row in web_results
        if str(row.get("url", "")).strip() and str(row.get("url", "")).strip() in answer
    }

    def replace_source_marker(match: re.Match[str]) -> str:
        source_number = int(match.group(1))
        if source_number < 1 or source_number > len(web_results):
            return ""
        row = web_results[source_number - 1]
        title = str(row.get("title", "")).strip() or str(row.get("url", "")).strip() or "Web source"
        url = str(row.get("url", "")).strip()
        return f"[{title}]({url})" if url else title

    answer = re.sub(
        r"\[source\s+(\d+)\]",
        replace_source_marker,
        answer,
        flags=re.IGNORECASE,
    )
    citations: list[dict[str, Any]] = []
    for index, row in enumerate(web_results, start=1):
        url = str(row.get("url", "")).strip()
        if index not in cited_source_numbers and url not in cited_source_urls:
            continue
        host = urlparse(url).netloc or "web"
        citations.append(
            {
                "modality": "web",
                "source": host,
                "title": row.get("title", ""),
                "url": url,
                "published_date": row.get("published_date"),
            }
        )

    return answer, citations


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
                "\n\nInspect the attached images directly before answering.",
            )
            response = model.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=human_content)])
        else:
            response = model.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=prompt)])
        answer = str(response.content).strip()
    except Exception as e:
        logger.error(f"Error in answer_with_evidence: {e}")
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
        logger.error(f"Error in is_answer_confident: {e}")

    lowered = answer.lower()
    refusal_markers = (
        "insufficient",
        "not enough evidence",
        "could not find enough evidence",
        "uncertain",
    )
    return not any(marker in lowered for marker in refusal_markers)
