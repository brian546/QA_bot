from project.frontend.components.browser_cleanup import build_cleanup_payload, build_cleanup_url


def test_browser_cleanup_helper_payload() -> None:
    payload = build_cleanup_payload("session-123")
    assert payload == {"session_id": "session-123"}


def test_browser_cleanup_helper_url() -> None:
    assert build_cleanup_url("http://localhost:8000/") == "http://localhost:8000/clear-session"
