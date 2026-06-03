from __future__ import annotations

import io
from collections import Counter

from pypdf import PdfReader


def _strip_repeated_page_edges(page_texts: list[str]) -> list[str]:
    """Remove repeated first and last lines that likely represent headers/footers."""
    first_lines: list[str] = []
    last_lines: list[str] = []

    split_pages: list[list[str]] = []
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        split_pages.append(lines)
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])

    first_common = {line for line, count in Counter(first_lines).items() if count > 1}
    last_common = {line for line, count in Counter(last_lines).items() if count > 1}

    cleaned: list[str] = []
    for lines in split_pages:
        if lines and lines[0] in first_common:
            lines = lines[1:]
        if lines and lines[-1] in last_common:
            lines = lines[:-1]
        cleaned.append("\n".join(lines).strip())
    return cleaned


def parse_pdf_pages(pdf_bytes: bytes, filename: str) -> list[dict[str, object]]:
    """Parse PDF by page and keep source metadata."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw_pages: list[str] = []
    for page in reader.pages:
        raw_pages.append((page.extract_text() or "").strip())

    cleaned_pages = _strip_repeated_page_edges(raw_pages)

    parsed: list[dict[str, object]] = []
    for idx, text in enumerate(cleaned_pages, start=1):
        if not text:
            continue
        parsed.append({"filename": filename, "page": idx, "text": text})
    return parsed
