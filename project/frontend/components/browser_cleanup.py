from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components


def build_cleanup_payload(session_id: str) -> dict[str, str]:
  """Create cleanup request payload for browser-exit events."""
  return {"session_id": session_id}


def build_cleanup_url(api_base_url: str) -> str:
  """Create cleanup endpoint URL used by beacon requests."""
  return f"{api_base_url.rstrip('/')}/clear-session"


def mount_browser_cleanup_listener(api_base_url: str, session_id: str) -> None:
    """Attach best-effort browser lifecycle cleanup beacon for this session."""
    script = f"""
    <script>
      (function() {{
        if (!window.__pdfragCleanup) {{
          window.__pdfragCleanup = {{ attached: false, lastSession: null, sent: {{}} }};
        }}

        window.__pdfragCleanup.lastSession = {json.dumps(session_id)};
        if (window.__pdfragCleanup.attached) return;
        window.__pdfragCleanup.attached = true;

        function sendCleanup(reason) {{
          var sid = window.__pdfragCleanup.lastSession;
          if (!sid) return;
          if (window.__pdfragCleanup.sent[sid]) return;
          window.__pdfragCleanup.sent[sid] = true;
          var url = {json.dumps(build_cleanup_url(api_base_url))};
          var body = JSON.stringify({{ session_id: sid, reason: reason }});
          var blob = new Blob([body], {{ type: 'application/json' }});
          navigator.sendBeacon(url, blob);
        }}

        document.addEventListener('visibilitychange', function() {{
          if (document.visibilityState === 'hidden') {{
            sendCleanup('visibility_hidden');
          }}
        }});

        window.addEventListener('pagehide', function() {{
          sendCleanup('pagehide');
        }});
      }})();
    </script>
    """
    if hasattr(st, "html"):
      st.html(script)
    else:
      components.html(script, height=0, width=0)
