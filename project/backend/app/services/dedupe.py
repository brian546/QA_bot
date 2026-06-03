from __future__ import annotations

import os


def normalize_file_key(filename: str) -> str:
    """Normalize file key using lowercase name + extension."""
    base = os.path.basename(filename).strip().lower()
    if not base:
        return ""
    stem, ext = os.path.splitext(base)
    stem = " ".join(stem.split())
    return f"{stem}{ext}"
