from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from project.backend.app.schemas.request import RemoveFilesRequest
from project.backend.app.schemas.response import RemoveFilesResponse
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
    new_chunks: list[dict[str, object]] = []

    for file in files:
        file_name = file.filename or ""
        key = normalize_file_key(file_name)
        if not key.endswith(".pdf"):
            continue

        if key in session.processed_files:
            continue

        raw = await file.read()
        if len(raw) > settings.max_upload_bytes:
            continue

        try:
            pages = parse_pdf_pages(raw, file_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse {file_name}: {exc}") from exc

        page_count = len(pages)
        chunks = chunk_pages(pages, settings.max_chunk_chars, settings.chunk_overlap)
        new_chunks.extend(chunks)

        session.processed_files.add(key)
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
        uploaded_documents=session.uploaded_documents,
    )


@router.post("/upload/remove", response_model=RemoveFilesResponse)
def remove_files(payload: RemoveFilesRequest, request: Request) -> RemoveFilesResponse:
    settings = request.app.state.settings
    store = request.app.state.session_store
    session = store.get_or_create(payload.session_id)

    removed_files: list[str] = []
    seen: set[str] = set()

    for raw_key in payload.file_keys:
        key = normalize_file_key(raw_key)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)

        if key not in session.processed_files:
            continue

        session.processed_files.discard(key)
        session.uploaded_documents = [
            doc
            for doc in session.uploaded_documents
            if normalize_file_key(str(doc.get("normalized_key") or doc.get("filename", ""))) != key
        ]
        session.chunks = [
            chunk
            for chunk in session.chunks
            if normalize_file_key(str(chunk.get("filename", ""))) != key
        ]
        removed_files.append(key)

    lexical_index, lexical_tokens = build_bm25_index(session.chunks)
    session.lexical_index = lexical_index
    session.lexical_tokens = lexical_tokens
    session.semantic_index = build_faiss_index(session.chunks, settings.embedding_dimension)

    return RemoveFilesResponse(
        session_id=payload.session_id,
        removed_files=removed_files,
        uploaded_documents=session.uploaded_documents,
        processed_files=sorted(session.processed_files),
    )
