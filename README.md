# Hybrid Multi-PDF QA MVP

Production-friendly MVP for grounded multi-document question answering using FastAPI, Streamlit, LangGraph, LangChain, and OpenRouter.

## Product Overview

- Users upload multiple PDF files in Streamlit.
- New files are processed immediately as soon as they appear in the uploader.
- Duplicate uploads are skipped per session using normalized filename + extension.
- Retrieval is hybrid: lexical BM25 + semantic vector search.
- Answers are evidence-grounded and returned with citations.
- Chat supports follow-up questions via chat history.
- Clear session resets both frontend and backend session state.
- Browser-close cleanup is best-effort using beacon events.

## Project Structure

- `project/backend/app/main.py`
- `project/backend/app/core/config.py`
- `project/backend/app/core/llm.py`
- `project/backend/app/core/runtime_config.py`
- `project/backend/app/core/session_store.py`
- `project/backend/app/graph/*`
- `project/backend/app/routers/*`
- `project/backend/app/services/*`
- `project/frontend/app.py`
- `project/frontend/api_client.py`
- `project/frontend/utils.py`
- `project/frontend/components/*`
- `project/tests/*`

## LangGraph Architecture

Graph state tracks:

- session_id
- uploaded_files, accepted_files, skipped_files
- uploaded_documents
- chat_history
- current_question, rewritten_query
- lexical_results, semantic_results, fused_results
- compressed_context
- final_answer, citations, confidence
- retrieval_diagnostics
- llm_settings, effective_llm_settings
- error

Nodes:

1. ingest_upload
2. rewrite_query
3. lexical_retrieve
4. semantic_retrieve
5. fuse_results
6. compress_context
7. answer_question
8. evaluate_answer
9. fallback

Routing:

- If no uploaded docs exist for the session, route to fallback.
- Otherwise run hybrid retrieval and grounded answering.
- If evidence or citations are insufficient, route to fallback.

## Immediate Upload and Duplicate Skipping

- Frontend uses `st.file_uploader(..., accept_multiple_files=True)`.
- There is no manual process button.
- On each rerun, only newly uploaded files are sent to `POST /upload`.
- Duplicate detection uses normalized filename + extension and skips repeats.
- Skipped duplicates are shown in the UI.

## Hybrid Retrieval

- Lexical retrieval: BM25 over chunk text for sparse exact-match strength.
- Semantic retrieval: LangChain embeddings with FAISS vector store.
- Fusion: weighted reciprocal rank fusion.
- Diagnostics include top lexical, semantic, and fused hit summaries.

## Runtime Config and OpenRouter

- Backend is the source of truth for model options and defaults.
- Configuration is centralized in `backend/app/core/config.py` with Pydantic Settings.
- Settings are cached (`lru_cache`) to avoid reparsing `.env` on each request.
- LLM creation is centralized in `backend/app/core/llm.py` via `get_chat_model()`.
- Frontend fetches runtime-safe configuration from `GET /config` and never reads `.env`.
- `GET /config` excludes secrets and returns models, defaults, supported controls, and constraints.

## LLM Behavior Controls

Session-scoped controls (rendered dynamically from backend config):

- model
- temperature
- top_p
- max_tokens

Defaults are backend-provided and optimized for grounded QA (low randomness).

## Clear Session and Browser-Close Cleanup

Clear session button:

1. Calls `POST /clear-session` for current session_id.
2. Clears Streamlit chat/doc/retrieval/LLM state.
3. Increments uploader key to visually reset file uploader.
4. Creates a fresh session_id.
5. Restores backend defaults for LLM controls.

Browser-exit cleanup:

- Frontend mounts JS listeners for `visibilitychange` and `pagehide`.
- Uses `navigator.sendBeacon()` to post to `POST /clear-session`.
- Cleanup is best-effort only; explicit Clear session is the reliable reset path.

## API Endpoints

- `GET /health`
- `GET /config`
- `POST /upload`
- `POST /ask`
- `POST /clear-session`

## Local Setup and Run

1. Install dependencies.

```bash
uv sync
```

2. Create environment file and set API key.

```bash
cp .env.example .env
```

3. Run FastAPI backend.

```bash
uvicorn project.backend.app.main:app --reload --port 8000
```

4. Run Streamlit frontend.

```bash
streamlit run project/frontend/app.py --server.port 8511
```

5. Run tests.

```bash
pytest project/tests -q
```

## Why Hybrid Retrieval?

Hybrid retrieval combines lexical precision and semantic recall.
It handles exact entities (IDs, names, policy language) while still catching semantically related follow-up questions.

## Manual Testing Checklist

- Upload two PDFs and confirm immediate processing.
- Re-upload a normalized duplicate and confirm it is skipped.
- Ask a question and verify answer, citations, and diagnostics.
- Ask a follow-up and verify chat history is used.
- Change LLM controls and confirm effective settings in the response.
- Click Clear session and verify uploader, chat, docs, and diagnostics reset.
- Close or hide the tab and reopen to confirm best-effort backend cleanup behavior.
