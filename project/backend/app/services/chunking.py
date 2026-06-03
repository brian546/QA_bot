from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict[str, object]],
    max_chunk_chars: int,
    chunk_overlap: int,
) -> list[dict[str, object]]:
    """Split parsed pages into chunks while preserving metadata linkage."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, object]] = []
    for page in pages:
        filename = str(page["filename"])
        page_no = int(page["page"])
        page_text = str(page["text"])
        parts = splitter.split_text(page_text)
        for chunk_idx, chunk_text in enumerate(parts):
            chunk_id = f"{filename}:{page_no}:{chunk_idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "filename": filename,
                    "page": page_no,
                    "section": chunk_idx,
                    "text": chunk_text,
                }
            )
    return chunks
