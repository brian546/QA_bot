from __future__ import annotations

import os

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from project.backend.app.schemas.request import RemoveFilesRequest
from project.backend.app.schemas.response import RemoveFilesResponse
from project.backend.app.schemas.response import UploadResponse
from project.backend.app.services.chunking import chunk_pages
from project.backend.app.services.dedupe import normalize_file_key
from project.backend.app.services.image_assets import build_image_asset_record, extract_pdf_page_images, is_image_filename
from project.backend.app.services.image_retrieval import build_image_index
from project.backend.app.services.lexical_retrieval import build_bm25_index
from project.backend.app.services.parser import SUPPORTED_UPLOAD_EXTENSIONS, parse_document_pages
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
    media_store = request.app.state.media_store
    session = store.get_or_create(session_id)

    accepted_files: list[str] = []
    new_chunks: list[dict[str, object]] = []
    new_image_chunks: list[dict[str, object]] = []

    for file in files:
        file_name = file.filename or ""
        key = normalize_file_key(file_name)
        ext = os.path.splitext(key)[1]
        if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
            continue

        if key in session.processed_files:
            continue

        raw = await file.read()
        if len(raw) > settings.max_upload_bytes:
            continue

        session.processed_files.add(key)
        accepted_files.append(file_name)

        if is_image_filename(file_name):
            image_asset = build_image_asset_record(
                filename=file_name,
                raw_bytes=raw,
            )
            storage_record = media_store.save(
                session_id,
                str(image_asset["asset_id"]),
                file_name,
                raw,
                {"kind": "image"},
            )
            image_asset.update(
                {
                    "storage_uri": storage_record.storage_uri,
                    "storage_backend": storage_record.storage_backend,
                }
            )
            session.image_assets.append(image_asset)
            session.uploaded_documents.append(
                {
                    "filename": file_name,
                    "normalized_key": key,
                    "page_count": 1,
                    "chunk_count": 1,
                    "modality": "image",
                }
            )
            new_image_chunks.append(image_asset)
            continue

        try:
            pages = parse_document_pages(raw, file_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse {file_name}: {exc}") from exc

        page_count = len(pages)
        chunks = chunk_pages(pages, settings.max_chunk_chars, settings.chunk_overlap)
        new_chunks.extend(chunks)

        session.uploaded_documents.append(
            {
                "filename": file_name,
                "normalized_key": key,
                "page_count": page_count,
                "chunk_count": len(chunks),
                "modality": "text",
            }
        )

        if ext == ".pdf" and settings.enable_pdf_page_image_extraction:
            for page_image in extract_pdf_page_images(raw, file_name):
                raw_bytes = page_image["raw_bytes"]
                asset = page_image["asset"]
                storage_record = media_store.save(
                    session_id,
                    str(asset["asset_id"]),
                    file_name,
                    raw_bytes,
                    {"kind": "pdf_page_image"},
                )
                asset.update(
                    {
                        "storage_uri": storage_record.storage_uri,
                        "storage_backend": storage_record.storage_backend,
                    }
                )
                session.image_assets.append(asset)
                new_image_chunks.append(asset)

    if new_chunks:
        session.chunks.extend(new_chunks)
        lexical_index, lexical_tokens = build_bm25_index(session.chunks)
        session.lexical_index = lexical_index
        session.lexical_tokens = lexical_tokens
        session.semantic_index = build_faiss_index(session.chunks, settings.embedding_dimension)

    if new_image_chunks:
        session.image_chunks.extend(new_image_chunks)
        session.image_index = build_image_index(session.image_chunks, settings)

    return UploadResponse(
        session_id=session_id,
        accepted_files=accepted_files,
        uploaded_documents=session.uploaded_documents,
    )


@router.post("/upload/remove", response_model=RemoveFilesResponse)
def remove_files(payload: RemoveFilesRequest, request: Request) -> RemoveFilesResponse:
    settings = request.app.state.settings
    store = request.app.state.session_store
    media_store = request.app.state.media_store
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
        session.image_assets = [
            asset
            for asset in session.image_assets
            if normalize_file_key(str(asset.get("normalized_key") or asset.get("filename", ""))) != key
        ]
        session.image_chunks = [
            asset
            for asset in session.image_chunks
            if normalize_file_key(str(asset.get("normalized_key") or asset.get("filename", ""))) != key
        ]
        removed_files.append(key)

    lexical_index, lexical_tokens = build_bm25_index(session.chunks)
    session.lexical_index = lexical_index
    session.lexical_tokens = lexical_tokens
    session.semantic_index = build_faiss_index(session.chunks, settings.embedding_dimension)
    session.image_index = build_image_index(session.image_chunks, settings)

    return RemoveFilesResponse(
        session_id=payload.session_id,
        removed_files=removed_files,
        uploaded_documents=session.uploaded_documents,
        processed_files=sorted(session.processed_files),
    )
