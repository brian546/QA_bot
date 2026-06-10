from __future__ import annotations

import base64
import hashlib
import os
from typing import Any


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

import logging

logger = logging.getLogger(__name__)

def is_image_filename(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_IMAGE_EXTENSIONS


def detect_image_mime_type(filename: str, raw_bytes: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".gif":
        return "image/gif"
    if ext == ".webp":
        return "image/webp"
    if ext in {".tif", ".tiff"}:
        return "image/tiff"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _png_dimensions(raw_bytes: bytes) -> tuple[int, int] | None:
    if not raw_bytes.startswith(b"\x89PNG\r\n\x1a\n") or len(raw_bytes) < 24:
        return None
    width = int.from_bytes(raw_bytes[16:20], "big")
    height = int.from_bytes(raw_bytes[20:24], "big")
    return width, height


def _gif_dimensions(raw_bytes: bytes) -> tuple[int, int] | None:
    if not (raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a")) or len(raw_bytes) < 10:
        return None
    width = int.from_bytes(raw_bytes[6:8], "little")
    height = int.from_bytes(raw_bytes[8:10], "little")
    return width, height


def _jpeg_dimensions(raw_bytes: bytes) -> tuple[int, int] | None:
    if not raw_bytes.startswith(b"\xff\xd8"):
        return None
    idx = 2
    data_len = len(raw_bytes)
    while idx + 1 < data_len:
        if raw_bytes[idx] != 0xFF:
            idx += 1
            continue
        marker = raw_bytes[idx + 1]
        idx += 2
        if marker in {0xD8, 0xD9}:
            continue
        if idx + 2 > data_len:
            break
        segment_length = int.from_bytes(raw_bytes[idx : idx + 2], "big")
        if segment_length < 2:
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if idx + 7 <= data_len:
                height = int.from_bytes(raw_bytes[idx + 3 : idx + 5], "big")
                width = int.from_bytes(raw_bytes[idx + 5 : idx + 7], "big")
                return width, height
            return None
        idx += segment_length
    return None


def extract_image_dimensions(raw_bytes: bytes, mime_type: str) -> tuple[int, int] | None:
    if mime_type == "image/png":
        return _png_dimensions(raw_bytes)
    if mime_type == "image/gif":
        return _gif_dimensions(raw_bytes)
    if mime_type == "image/jpeg":
        return _jpeg_dimensions(raw_bytes)
    return None


def sha256_hex(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def to_data_url(raw_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_image_caption(filename: str, width: int | None, height: int | None, page: int | None = None) -> str:
    size = f"{width}x{height}" if width and height else "unknown size"
    page_part = f" page {page}" if page is not None else ""
    return f"Image asset from {filename}{page_part} ({size})"


def build_image_asset_record(
    *,
    filename: str,
    raw_bytes: bytes,
    page: int | None = None,
    asset_kind: str = "image",
    storage_uri: str | None = None,
    storage_backend: str = "memory",
) -> dict[str, Any]:
    mime_type = detect_image_mime_type(filename, raw_bytes)
    dims = extract_image_dimensions(raw_bytes, mime_type)
    width, height = dims if dims is not None else (None, None)
    asset_id = sha256_hex(raw_bytes)
    return {
        "asset_id": asset_id,
        "chunk_id": f"{filename}:{page or 1}:0",
        "filename": filename,
        "normalized_key": filename.lower(),
        "page": page or 1,
        "section": 0,
        "text": build_image_caption(filename, width, height, page),
        "modality": "image",
        "asset_kind": asset_kind,
        "mime_type": mime_type,
        "image_width": width,
        "image_height": height,
        "image_sha256": asset_id,
        "image_data_url": to_data_url(raw_bytes, mime_type),
        "byte_size": len(raw_bytes),
        "storage_backend": storage_backend,
        "storage_uri": storage_uri,
    }


def extract_pdf_page_images(pdf_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore
    except Exception as e:
        logger.error(f"Error importing fitz: {e}")
        return []

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    assets: list[dict[str, Any]] = []
    try:
        for page_idx, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image_bytes = pixmap.tobytes("png")
            assets.append(
                {
                    "raw_bytes": image_bytes,
                    "asset": build_image_asset_record(
                        filename=filename,
                        raw_bytes=image_bytes,
                        page=page_idx,
                        asset_kind="pdf_page_image",
                    ),
                }
            )
    finally:
        document.close()
    return assets
