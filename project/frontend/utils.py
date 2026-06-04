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
    if "selected_file_keys" not in st.session_state:
        st.session_state.selected_file_keys = set()
    if "upload_feedback" not in st.session_state:
        st.session_state.upload_feedback = {
            "accepted": [],
            "skipped": [],
            "removed": [],
            "error": "",
        }
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


def handle_uploader_change(client: APIClient, uploader_state_key: str) -> None:
    """Sync uploader selection changes to backend: add new files and remove deselected files."""
    uploaded_files = st.session_state.get(uploader_state_key) or []
    current_files_by_key: dict[str, Any] = {}
    for file in uploaded_files:
        key = normalize_file_key(file.name)
        if key and key not in current_files_by_key:
            current_files_by_key[key] = file

    previous_keys = set(st.session_state.get("selected_file_keys", set()))
    current_keys = set(current_files_by_key.keys())

    added_keys = sorted(current_keys - previous_keys)
    removed_keys = sorted(previous_keys - current_keys)

    accepted: list[str] = []
    skipped: list[str] = []
    removed: list[str] = []

    try:
        if removed_keys:
            remove_response = client.remove_files(st.session_state.session_id, removed_keys)
            st.session_state.uploaded_docs = remove_response.get("uploaded_documents", [])
            st.session_state.processed_files = set(remove_response.get("processed_files", []))
            removed.extend(remove_response.get("removed_files", []))
            skipped.extend(
                [
                    f"{item.get('filename', 'Unknown file')}: {item.get('reason', 'Remove skipped.')}"
                    for item in remove_response.get("skipped_details", [])
                ]
            )

        if added_keys:
            added_files = [current_files_by_key[key] for key in added_keys]
            accepted_names, skipped_items = process_new_uploads(client, added_files)
            accepted.extend(accepted_names)
            skipped.extend(skipped_items)

        st.session_state.selected_file_keys = current_keys
        st.session_state.upload_feedback = {
            "accepted": accepted,
            "skipped": skipped,
            "removed": removed,
            "error": "",
        }
    except Exception as exc:
        st.session_state.upload_feedback = {
            "accepted": accepted,
            "skipped": skipped,
            "removed": removed,
            "error": f"Upload sync failed: {exc}",
        }


def reset_llm_settings_to_defaults() -> None:
    defaults = normalize_llm_settings(
        st.session_state.runtime_config,
        initialize_llm_settings_from_runtime(st.session_state.runtime_config),
    )
    st.session_state.llm_settings = defaults
    # Defer widget key updates until the next rerun before controls are instantiated.
    st.session_state["llm_reset_requested"] = True


def start_new_session_state(client: APIClient) -> None:
    """Start a fresh frontend session without deleting backend session data."""
    st.session_state.messages = []
    st.session_state.processed_files = set()
    st.session_state.uploaded_docs = []
    st.session_state.citations = []
    st.session_state.retrieval_diagnostics = {}
    st.session_state.session_id = _new_session_id()
    st.session_state.uploader_key += 1
    st.session_state.selected_file_keys = set()
    st.session_state.upload_feedback = {
        "accepted": [],
        "skipped": [],
        "removed": [],
        "error": "",
    }
    st.session_state.runtime_config = client.get_config()
    reset_llm_settings_to_defaults()


def clear_session_state(client: APIClient) -> None:
    """Clear frontend state and trigger backend session cleanup."""
    old_session_id = st.session_state.session_id
    client.clear_session(old_session_id)
    start_new_session_state(client)


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
    st.session_state.selected_file_keys = set()
    st.session_state.upload_feedback = {
        "accepted": [],
        "skipped": [],
        "removed": [],
        "error": "",
    }

    runtime_config = st.session_state.runtime_config
    st.session_state.llm_settings = normalize_llm_settings(
        runtime_config,
        session.get("llm_settings") or initialize_llm_settings_from_runtime(runtime_config),
    )
