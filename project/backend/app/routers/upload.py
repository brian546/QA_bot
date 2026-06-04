from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from project.backend.app.schemas.response import UploadResponse
from project.backend.app.services.chunking import chunk_pages
from project.backend.app.services.dedupe import normalize_file_key
from project.backend.app.services.lexical_retrieval import build_bm25_index
from project.backend.app.services.parser import parse_pdf_pages
from project.backend.app.services.semantic_retrieval import build_faiss_index

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    request: Request,
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    settings = request.app.state.settings
    store = request.app.state.session_store
    session = store.get_or_create(session_id)

    accepted_files: list[str] = []
    skipped_files: list[str] = []
    skipped_details: list[dict[str, str]] = []
    batch_seen: set[str] = set()
    new_chunks: list[dict[str, object]] = []

    def mark_skipped(file_name: str, reason: str) -> None:
        skipped_files.append(file_name)
        skipped_details.append({"filename": file_name, "reason": reason})

    for file in files:
        file_name = file.filename or ""
        key = normalize_file_key(file_name)
        if not key.endswith(".pdf"):
            mark_skipped(file_name, "Only PDF files are supported.")
            continue

        if key in session.processed_files or key in batch_seen:
            mark_skipped(file_name, "File already exists in this session.")
            continue

        raw = await file.read()
        if len(raw) > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes / 1_000_000
            mark_skipped(file_name, f"File exceeds the {limit_mb:.0f} MB upload limit.")
            continue

        try:
            pages = parse_pdf_pages(raw, file_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse {file_name}: {exc}") from exc

        page_count = len(pages)
        chunks = chunk_pages(pages, settings.max_chunk_chars, settings.chunk_overlap)
        new_chunks.extend(chunks)

        session.processed_files.add(key)
        batch_seen.add(key)
        accepted_files.append(file_name)
        session.uploaded_documents.append(
            {
                "filename": file_name,
                "normalized_key": key,
                "page_count": page_count,
                "chunk_count": len(chunks),
            }
        )

    if new_chunks:
        session.chunks.extend(new_chunks)
        lexical_index, lexical_tokens = build_bm25_index(session.chunks)
        session.lexical_index = lexical_index
        session.lexical_tokens = lexical_tokens
        session.semantic_index = build_faiss_index(session.chunks, settings.embedding_dimension)

    return UploadResponse(
        session_id=session_id,
        accepted_files=accepted_files,
        skipped_files=skipped_files,
        skipped_details=skipped_details,
        uploaded_documents=session.uploaded_documents,
    )
