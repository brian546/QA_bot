from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.backend.app.core.config import get_settings
from project.backend.app.core.session_store import session_store


@pytest.fixture(autouse=True)
def _test_env() -> None:
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
    os.environ.setdefault("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    os.environ.setdefault("OPENROUTER_ALLOWED_MODELS", "openai/gpt-4o-mini,anthropic/claude-3.5-sonnet")
    get_settings.cache_clear()
    session_store.clear_all()
    yield
    session_store.clear_all()
