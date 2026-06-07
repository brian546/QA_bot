from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
for candidate in (str(REPO_ROOT), str(CURRENT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from project.frontend.api_client import APIClient
from project.frontend.components.llm_controls import render_llm_controls
from project.frontend.components.retrieval_controls import (
    initialize_retrieval_settings_from_runtime,
    normalize_retrieval_settings,
    render_retrieval_controls,
)
from project.frontend.utils import (
    DEFAULT_BACKEND_URL,
    clear_session_state,
    ensure_state,
    handle_uploader_change,
    reset_llm_settings_to_defaults,
    start_new_session_state,
    switch_session_state,
)


def build_session_label(session: dict[str, object]) -> str:
    session_id = str(session.get("session_id", ""))
    short_id = session_id if len(session_id) <= 12 else f"{session_id[:8]}...{session_id[-4:]}"
    doc_count = int(session.get("uploaded_document_count", 0))
    chat_count = int(session.get("chat_message_count", 0))
    return f"{short_id} ({doc_count} docs, {chat_count} msgs)"


def reset_retrieval_settings_locally() -> None:
    st.session_state.retrieval_settings = normalize_retrieval_settings(
        st.session_state.runtime_config,
        {
            "lexical_weight": 0.5,
            "semantic_weight": 0.5,
        },
    )
    st.session_state["retrieval_reset_requested"] = True


def main() -> None:
    st.set_page_config(page_title="Hybrid Document QA", layout="wide")
    st.title("Hybrid Multi-Document QA")

    # backend_url = st.sidebar.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    client = APIClient(DEFAULT_BACKEND_URL)

    try:
        ensure_state(client)
    except requests.RequestException as exc:
        st.error(f"Failed to load runtime config from backend: {exc}")
        st.stop()

    if "retrieval_settings" not in st.session_state:
        st.session_state.retrieval_settings = normalize_retrieval_settings(
            st.session_state.runtime_config,
            initialize_retrieval_settings_from_runtime(st.session_state.runtime_config),
        )

    try:
        backend_sessions = client.list_sessions().get("sessions", [])
    except requests.RequestException as exc:
        backend_sessions = []
        st.sidebar.error(f"Failed to load backend sessions: {exc}")

    st.sidebar.markdown(f"Session: `{st.session_state.session_id}`")
    st.sidebar.subheader("Stored sessions")
    if backend_sessions:
        for session in backend_sessions:
            session_id = str(session.get("session_id", ""))
            is_active = session_id == st.session_state.session_id
            col_switch, col_del = st.sidebar.columns([5, 1])
            with col_switch:
                if st.button(
                    build_session_label(session),
                    key=f"session-switch-{session_id}",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    try:
                        switch_session_state(client, session_id)
                        st.rerun()
                    except requests.RequestException as exc:
                        st.sidebar.error(f"Failed to switch session: {exc}")
            with col_del:
                if st.button(
                    "🗑",
                    key=f"session-delete-{session_id}",
                    help="Delete this session",
                ):
                    try:
                        client.clear_session(session_id)
                        if is_active:
                            clear_session_state(client)
                        st.rerun()
                    except requests.RequestException as exc:
                        st.sidebar.error(f"Failed to delete session: {exc}")
    else:
        st.sidebar.caption("No sessions stored in backend yet.")

    if st.sidebar.button("New session", use_container_width=True):
        try:
            start_new_session_state(client)
            st.sidebar.success("Started a new session.")
            st.rerun()
        except requests.RequestException as exc:
            st.sidebar.error(f"New-session failed: {exc}")

    llm_settings, reset_clicked = render_llm_controls(
        st.session_state.runtime_config,
        st.session_state.llm_settings,
    )
    st.session_state.llm_settings = llm_settings
    if reset_clicked:
        reset_llm_settings_to_defaults()
        st.rerun()

    retrieval_settings, retrieval_reset_clicked = render_retrieval_controls(
        st.session_state.runtime_config,
        st.session_state.retrieval_settings,
    )
    st.session_state.retrieval_settings = retrieval_settings
    if retrieval_reset_clicked:
        reset_retrieval_settings_locally()
        st.rerun()

    uploader_state_key = f"uploader_files_{st.session_state.uploader_key}"
    uploader_label = "Upload documents (PDF, TXT, MD, CSV, DOCX, PPTX, XLSX)"
    if st.session_state.uploaded_docs:
        loaded_names = ", ".join(str(doc.get("filename", "unknown")) for doc in st.session_state.uploaded_docs)
        uploader_label = f"{uploader_label}\n\nLoaded in this session: {loaded_names}"

    st.file_uploader(
        uploader_label,
        type=["pdf", "txt", "md", "markdown", "csv", "docx", "pptx", "xlsx"],
        accept_multiple_files=True,
        key=uploader_state_key,
        on_change=handle_uploader_change,
        args=(client, uploader_state_key),
    )

    feedback = st.session_state.upload_feedback
    if feedback.get("accepted"):
        st.success(f"Processed: {', '.join(feedback['accepted'])}")
    if feedback.get("removed"):
        st.info(f"Removed: {', '.join(feedback['removed'])}")
    if feedback.get("error"):
        st.error(str(feedback["error"]))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your uploaded documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating grounded answer..."):
                try:
                    response = client.ask(
                        session_id=st.session_state.session_id,
                        question=question,
                        llm_settings=st.session_state.llm_settings,
                        retrieval_settings=st.session_state.retrieval_settings,
                    )
                except requests.RequestException as exc:
                    st.error(f"Ask failed: {exc}")
                    return

            answer = response.get("answer", "I could not find enough evidence in the uploaded documents.")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.citations = response.get("citations", [])
            st.session_state.retrieval_diagnostics = response.get("retrieval_diagnostics", {})
            citations = response.get("citations", [])

            if citations:
                with st.expander("Retrieval diagnostics", expanded=False):
                    st.json(response.get("retrieval_diagnostics", {}))

            with st.expander("Effective LLM settings", expanded=False):
                st.json(response.get("effective_llm_settings", {}))

            with st.expander("Effective retrieval settings", expanded=False):
                st.json(response.get("effective_retrieval_settings", {}))


if __name__ == "__main__":
    main()
