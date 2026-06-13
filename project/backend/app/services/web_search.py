from __future__ import annotations

import logging
from typing import Any

from project.backend.app.core.config import Settings

logger = logging.getLogger(__name__)


def search_web_with_tavily(settings: Settings, query: str) -> list[dict[str, Any]]:
    """Execute a web search via Tavily and normalize result fields."""
    if not settings.web_search_enabled:
        return []
    if not settings.tavily_api_key.strip():
        return []
    if not query.strip():
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        payload = client.search(
            query=query,
            max_results=int(settings.web_search_max_results),
            search_depth="basic",
            include_raw_content=False,
            include_answer=False,
        )
    except Exception as exc:
        logger.error("Tavily web search failed: %s", exc)
        return []

    results = payload.get("results", []) if isinstance(payload, dict) else []
    normalized: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url", "")).strip()
        title = str(row.get("title", "")).strip()
        content = str(row.get("content", "")).strip()
        if not (url or title or content):
            continue
        normalized.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "score": row.get("score"),
                "published_date": row.get("published_date"),
            }
        )
    return normalized
