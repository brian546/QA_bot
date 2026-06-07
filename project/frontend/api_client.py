from __future__ import annotations

import mimetypes
from typing import Any

import requests


class APIClient:
    """Thin HTTP client for Streamlit frontend to call FastAPI backend."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_config(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/config", timeout=20)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def upload(self, session_id: str, files: list[Any]) -> dict[str, Any]:
        payload = []
        for file_obj in files:
            mime_type = mimetypes.guess_type(file_obj.name)[0] or "application/octet-stream"
            payload.append(("files", (file_obj.name, file_obj.getvalue(), mime_type)))
        response = requests.post(
            f"{self.base_url}/upload",
            data={"session_id": session_id},
            files=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def remove_files(self, session_id: str, file_keys: list[str]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/upload/remove",
            json={"session_id": session_id, "file_keys": file_keys},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def ask(
        self,
        session_id: str,
        question: str,
        chat_history: list[dict[str, str]],
        llm_settings: dict[str, Any],
        retrieval_settings: dict[str, Any],
        citations_k: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "question": question,
            "chat_history": chat_history,
            "llm_settings": llm_settings,
            "retrieval_settings": retrieval_settings,
        }
        if citations_k is not None:
            payload["citations_k"] = int(citations_k)

        response = requests.post(
            f"{self.base_url}/ask",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def clear_session(self, session_id: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/clear-session",
            json={"session_id": session_id},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def list_sessions(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/sessions", timeout=20)
        response.raise_for_status()
        return response.json()

    def get_session(self, session_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/sessions/{session_id}", timeout=20)
        response.raise_for_status()
        return response.json()
