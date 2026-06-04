from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from project.frontend.api_client import APIClient
from project.frontend.components.llm_controls import initialize_llm_settings_from_runtime, normalize_llm_settings


DEFAULT_BACKEND_URL = "http://localhost:8000"


def _new_session_id() -> str:
    return str(uuid.uuid4())


def ensure_state(client: APIClient) -> None:
    """Initialize Streamlit session state and load runtime config once."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = _new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
    if "citations" not in st.session_state:
        st.session_state.citations = []
    if "retrieval_diagnostics" not in st.session_state:
        st.session_state.retrieval_diagnostics = {}
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "runtime_config" not in st.session_state:
        st.session_state.runtime_config = client.get_config()
    if "llm_settings" not in st.session_state:
        st.session_state.llm_settings = normalize_llm_settings(
            st.session_state.runtime_config,
            initialize_llm_settings_from_runtime(st.session_state.runtime_config),
        )
    else:
        st.session_state.llm_settings = normalize_llm_settings(
            st.session_state.runtime_config,
            st.session_state.llm_settings,
        )


def normalize_file_key(filename: str) -> str:
    name = filename.strip().lower()
    if "." not in name:
        return name
    stem, ext = name.rsplit(".", 1)
    stem = " ".join(stem.split())
    return f"{stem}.{ext}"


def process_new_uploads(client: APIClient, uploaded_files: list[Any]) -> tuple[list[str], list[str]]:
    """Process only newly uploaded files and skip duplicates in-session."""
    if not uploaded_files:
        return [], []

    new_files = []
    locally_skipped: list[str] = []
    for file in uploaded_files:
        key = normalize_file_key(file.name)
        if key in st.session_state.processed_files:
            locally_skipped.append(f"{file.name}: File already exists in this session.")
            continue
        new_files.append(file)

    if not new_files:
        return [], locally_skipped

    response = client.upload(st.session_state.session_id, new_files)
    for accepted_name in response.get("accepted_files", []):
        st.session_state.processed_files.add(normalize_file_key(accepted_name))

    st.session_state.uploaded_docs = response.get("uploaded_documents", [])
    remote_skipped = [
        f"{item.get('filename', 'Unknown file')}: {item.get('reason', 'Upload skipped.')}"
        for item in response.get("skipped_details", [])
    ]
    return response.get("accepted_files", []), locally_skipped + remote_skipped


def reset_llm_settings_to_defaults() -> None:
    st.session_state.llm_settings = normalize_llm_settings(
        st.session_state.runtime_config,
        initialize_llm_settings_from_runtime(st.session_state.runtime_config),
    )


def clear_session_state(client: APIClient) -> None:
    """Clear frontend state and trigger backend session cleanup."""
    old_session_id = st.session_state.session_id
    client.clear_session(old_session_id)

    st.session_state.messages = []
    st.session_state.processed_files = set()
    st.session_state.uploaded_docs = []
    st.session_state.citations = []
    st.session_state.retrieval_diagnostics = {}
    st.session_state.session_id = _new_session_id()
    st.session_state.uploader_key += 1
    st.session_state.runtime_config = client.get_config()
    reset_llm_settings_to_defaults()


def switch_session_state(client: APIClient, session_id: str) -> None:
    """Load one backend session into the current Streamlit state."""
    session = client.get_session(session_id)

    st.session_state.session_id = session["session_id"]
    st.session_state.messages = session.get("chat_history", [])
    st.session_state.processed_files = set(session.get("processed_files", []))
    st.session_state.uploaded_docs = session.get("uploaded_documents", [])
    st.session_state.citations = []
    st.session_state.retrieval_diagnostics = {}
    st.session_state.uploader_key += 1

    runtime_config = st.session_state.runtime_config
    st.session_state.llm_settings = normalize_llm_settings(
        runtime_config,
        session.get("llm_settings") or initialize_llm_settings_from_runtime(runtime_config),
    )
