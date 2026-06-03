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
from project.frontend.utils import (
    DEFAULT_BACKEND_URL,
    clear_session_state,
    ensure_state,
    process_new_uploads,
    reset_llm_settings_to_defaults,
)


def build_chat_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def main() -> None:
    st.set_page_config(page_title="Hybrid PDF QA", layout="wide")
    st.title("Hybrid Multi-PDF QA")

    # backend_url = st.sidebar.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    client = APIClient(DEFAULT_BACKEND_URL)

    try:
        ensure_state(client)
    except requests.RequestException as exc:
        st.error(f"Failed to load runtime config from backend: {exc}")
        st.stop()

    st.sidebar.markdown(f"Session: `{st.session_state.session_id}`")
    if st.sidebar.button("Clear session", use_container_width=True):
        try:
            clear_session_state(client)
            st.sidebar.success("Session cleared.")
            st.rerun()
        except requests.RequestException as exc:
            st.sidebar.error(f"Clear-session failed: {exc}")

    llm_settings, reset_clicked = render_llm_controls(
        st.session_state.runtime_config,
        st.session_state.llm_settings,
    )
    st.session_state.llm_settings = llm_settings
    if reset_clicked:
        reset_llm_settings_to_defaults()
        st.rerun()

    uploads = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    accepted, skipped = process_new_uploads(client, uploads or [])
    if accepted:
        st.success(f"Processed: {', '.join(accepted)}")
    if skipped:
        st.info(f"Skipped duplicates or invalid files: {', '.join(skipped)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your uploaded PDFs")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating grounded answer..."):
                try:
                    response = requests.post(
                                f"http://localhost:8000/ask",
                                json={
                                    "session_id": st.session_state.session_id,
                                    "question": question,
                                    "chat_history": build_chat_history(st.session_state.messages[:-1]),
                                    "llm_settings": st.session_state.llm_settings,
                                },
                                timeout=120,
                            )
                    response.raise_for_status()
                    response = response.json()
                    # response = client.ask(
                    #     session_id=st.session_state.session_id,
                    #     question=question,
                    #     chat_history=build_chat_history(st.session_state.messages[:-1]),
                    #     llm_settings=st.session_state.llm_settings,
                    # )
                except requests.RequestException as exc:
                    st.error(f"Ask failed: {exc}")
                    return

            answer = response.get("answer", "I could not find enough evidence in the uploaded PDFs.")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.citations = response.get("citations", [])
            st.session_state.retrieval_diagnostics = response.get("retrieval_diagnostics", {})

            with st.expander("Citations", expanded=True):
                citations = response.get("citations", [])
                if citations:
                    for cite in citations:
                        st.write(
                            f"- {cite.get('filename')} page {cite.get('page')} chunk {cite.get('chunk_id')}"
                        )
                else:
                    st.caption("No citations returned.")

            with st.expander("Retrieval diagnostics", expanded=False):
                st.json(response.get("retrieval_diagnostics", {}))

            with st.expander("Effective LLM settings", expanded=False):
                st.json(response.get("effective_llm_settings", {}))


if __name__ == "__main__":
    main()
