# Hybrid Multi-Document QA MVP

Grounded multi-document question answering app built with FastAPI, Streamlit, LangGraph, LangChain, and OpenRouter.

## What It Does

- Upload multiple documents (PDF, TXT, Markdown, CSV, DOCX, PPTX, XLSX) and process them immediately.
- Keep uploads session-scoped, with duplicate skipping by normalized file key.
- Sync uploader deselection to backend removal (`/upload/remove`).
- Answer questions with hybrid retrieval (BM25 + semantic vectors) and citations.
- Support follow-up questions using session chat history.
- Expose session-scoped LLM and retrieval controls from backend runtime config.
- Manage stored sessions from the UI: list, switch, delete, and start new session.

## Architecture At A Glance

- Backend: FastAPI app with routers for config, upload, chat, and session management.
- Frontend: Streamlit app with runtime-driven controls and uploader sync callbacks.
- Orchestration: LangGraph state machine for direct answer vs retrieval workflow.
- Retrieval: BM25 lexical index + FAISS semantic index with weighted reciprocal rank fusion.

## Local Setup

1. Install dependencies.

```bash
uv sync
```
On macOS, need to install `faiss-cpu` separately due to `uv` constraints:

```bash
uv pip install faiss-cpu
```

2. Create and edit environment file.

```bash
cp .env.example .env
```

Required values in `.env`:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`

Optional runtime tuning values are documented in `.env.example`.

3. Start backend.

```bash
uv run uvicorn project.backend.app.main:app --reload --port 8000
```

4. Start frontend.

```bash
uv run streamlit run project/frontend/app.py --server.port 8511
```

5. Run tests.

```bash
uv run pytest project/tests -q
```

## LangGraph Flow

Core nodes:

1. `ingest_upload`
2. `query_router`
3. `rewrite_query` (search path)
4. `lexical_retrieve`
5. `semantic_retrieve`
6. `fuse_results`
7. `compress_context`
8. `answer_question`
9. `evaluate_answer`
10. `fallback`

Routing behavior:

- If no documents are available in session, route to `direct` answer path.
- If document search is needed, run full retrieval pipeline.
- If grounded answer quality is insufficient (missing evidence/citations or too low confidence), route to `fallback`.

## Runtime Config And Controls

- Frontend fetches safe runtime config from `GET /config`.
- Backend is source of truth for models, defaults, and parameter constraints.
- Config payload excludes secrets.
- Supported controls currently include `model`, `temperature`, `top_p`, `lexical_weight`, and `semantic_weight`.
- Retrieval weights are normalized server-side before fusion.

## Session And Upload Behavior

- `st.file_uploader(..., accept_multiple_files=True)` is used with on-change sync.
- Newly selected files are uploaded immediately.
- Deselected files are removed from backend indexes/doc state.
- Duplicate uploads in a session are skipped via normalized key.
- Frontend supports starting a new session, listing stored sessions, switching to a prior session, and deleting a specific session.

## API Endpoints

- `GET /health`
- `GET /config`
- `POST /upload` (multipart form: `session_id`, `files`)
- `POST /upload/remove` (json: `session_id`, `file_keys`)
- `POST /ask` (json: `session_id`, `question`, optional `chat_history`, `llm_settings`, `retrieval_settings`)
- `POST /clear-session` (json: `session_id`)
- `GET /sessions`
- `GET /sessions/{session_id}`

## Key Paths

- `project/backend/app/main.py`
- `project/backend/app/core/config.py`
- `project/backend/app/core/llm.py`
- `project/backend/app/core/runtime_config.py`
- `project/backend/app/core/session_store.py`
- `project/backend/app/graph/`
- `project/backend/app/routers/`
- `project/backend/app/services/`
- `project/frontend/app.py`
- `project/frontend/api_client.py`
- `project/frontend/utils.py`
- `project/frontend/components/`
- `project/tests/`

## Manual Verification Checklist

- Upload 2 supported documents and confirm immediate processing.
- Remove one file in uploader and confirm backend document/index removal effects.
- Re-add a previously removed file and confirm it is processed again.
- Re-upload a duplicate normalized key and confirm it is skipped.
- Ask a question with docs and verify citations + retrieval diagnostics.
- Ask a follow-up and verify context continuity via chat history.
- Change LLM/retrieval controls and verify effective settings in `/ask` response.
- Create a new session, switch between sessions, and delete a session from sidebar.
- Clear current session and confirm frontend/backend reset for that session.

## Privacy Notice

Warning: Do not upload documents containing personal, sensitive, or confidential information. This app calls external LLM/embedding services and stores session data in memory for app functionality.
