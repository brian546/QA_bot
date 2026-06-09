from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class StoredMedia:
    session_id: str
    asset_id: str
    filename: str
    storage_uri: str
    storage_backend: str


class InMemoryMediaStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._data: dict[str, dict[str, bytes]] = {}

    def save(self, session_id: str, asset_id: str, filename: str, raw_bytes: bytes, metadata: dict[str, Any]) -> StoredMedia:
        with self._lock:
            self._data.setdefault(session_id, {})[asset_id] = raw_bytes
        return StoredMedia(
            session_id=session_id,
            asset_id=asset_id,
            filename=filename,
            storage_uri=f"memory://{session_id}/{asset_id}",
            storage_backend="memory",
        )

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


class FilesystemMediaStore:
    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, asset_id: str, filename: str, raw_bytes: bytes, metadata: dict[str, Any]) -> StoredMedia:
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        file_path = session_dir / f"{asset_id}.bin"
        meta_path = session_dir / f"{asset_id}.json"
        file_path.write_bytes(raw_bytes)
        meta_path.write_text(json.dumps({"filename": filename, **metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
        return StoredMedia(
            session_id=session_id,
            asset_id=asset_id,
            filename=filename,
            storage_uri=str(file_path),
            storage_backend="filesystem",
        )

    def delete_session(self, session_id: str) -> None:
        session_dir = self.root / session_id
        if not session_dir.exists():
            return
        for child in session_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
        session_dir.rmdir()


def build_media_store(settings: Any) -> InMemoryMediaStore | FilesystemMediaStore:
    backend = str(getattr(settings, "asset_storage_backend", "memory")).strip().lower()
    if backend == "filesystem":
        return FilesystemMediaStore(getattr(settings, "asset_storage_path", ".pdfrag-assets"))
    return InMemoryMediaStore()
