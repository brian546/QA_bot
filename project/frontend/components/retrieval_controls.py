from __future__ import annotations

from typing import Any

import streamlit as st


def _align_slider_value(value: float, min_value: float, max_value: float, step: float) -> float:
    if step <= 0:
        return min(max(value, min_value), max_value)

    bounded = min(max(value, min_value), max_value)
    steps = round((bounded - min_value) / step)
    aligned = min_value + (steps * step)
    return min(max(aligned, min_value), max_value)


def _sync_widget_value(key: str, value: Any, *, force: bool = False) -> None:
    if force or key not in st.session_state:
        st.session_state[key] = value


def _to_blend_ratio(settings: dict[str, Any]) -> float:
    lexical = max(0.0, float(settings.get("lexical_weight", 0.0)))
    semantic = max(0.0, float(settings.get("semantic_weight", 0.0)))
    total = lexical + semantic
    if total <= 0:
        return 0.5
    return semantic / total


def _from_blend_ratio(blend_ratio: float) -> dict[str, float]:
    semantic_weight = max(0.0, min(1.0, float(blend_ratio)))
    lexical_weight = 1.0 - semantic_weight
    return {
        "lexical_weight": lexical_weight,
        "semantic_weight": semantic_weight,
    }


def initialize_retrieval_settings_from_runtime(runtime_config: dict[str, Any]) -> dict[str, Any]:
    defaults = runtime_config.get("default_retrieval_settings", {})
    return {
        "lexical_weight": float(defaults.get("lexical_weight", 1.0)),
        "semantic_weight": float(defaults.get("semantic_weight", 1.0)),
    }


def normalize_retrieval_settings(
    runtime_config: dict[str, Any],
    current: dict[str, Any] | None,
) -> dict[str, Any]:
    current = current or {}
    constraints = runtime_config.get("parameter_constraints", {})
    defaults = initialize_retrieval_settings_from_runtime(runtime_config)

    l_c = constraints.get("lexical_weight", {"min": 0.0, "max": 2.0, "step": 0.05})
    s_c = constraints.get("semantic_weight", {"min": 0.0, "max": 2.0, "step": 0.05})

    return {
        "lexical_weight": float(
            _align_slider_value(
                float(current.get("lexical_weight", defaults.get("lexical_weight", 1.0))),
                float(l_c.get("min", 0.0)),
                float(l_c.get("max", 2.0)),
                float(l_c.get("step", 0.05)),
            )
        ),
        "semantic_weight": float(
            _align_slider_value(
                float(current.get("semantic_weight", defaults.get("semantic_weight", 1.0))),
                float(s_c.get("min", 0.0)),
                float(s_c.get("max", 2.0)),
                float(s_c.get("step", 0.05)),
            )
        ),
    }


def render_retrieval_controls(
    runtime_config: dict[str, Any],
    current: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    normalized = normalize_retrieval_settings(runtime_config, current)
    blend_key = "retrieval_blend_control"

    reset_requested = bool(st.session_state.get("retrieval_reset_requested", False))
    if reset_requested:
        _sync_widget_value(blend_key, 0.5, force=True)
        st.session_state["retrieval_reset_requested"] = False
    else:
        current_blend = st.session_state.get(blend_key)
        if current_blend is None:
            _sync_widget_value(blend_key, _to_blend_ratio(normalized))

    effective_blend = float(st.session_state.get(blend_key, 0.5))
    effective_blend = _align_slider_value(effective_blend, 0.0, 1.0, 0.01)
    _sync_widget_value(blend_key, effective_blend, force=True)

    with st.sidebar.expander("Retrieval blend controls", expanded=False):
        st.caption("0.0 means 100% lexical. 1.0 means 100% semantic.")

        blend_ratio = st.slider(
            "Lexical ↔ Semantic blend",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key=blend_key,
            help="Blend rank fusion between keyword overlap and embedding similarity.",
        )

        st.caption(
            f"Current blend: lexical {(1.0 - blend_ratio) * 100:.0f}% | semantic {blend_ratio * 100:.0f}%"
        )

        reset_clicked = st.button("Reset retrieval blend", use_container_width=True)

    new_settings = _from_blend_ratio(float(blend_ratio))
    return normalize_retrieval_settings(runtime_config, new_settings), reset_clicked