from __future__ import annotations

from typing import Any

import streamlit as st


def _align_slider_value(value: float, min_value: float, max_value: float, step: float) -> float:
    """Snap a numeric slider value to the nearest valid step within bounds."""
    if step <= 0:
        return min(max(value, min_value), max_value)

    bounded = min(max(value, min_value), max_value)
    steps = round((bounded - min_value) / step)
    aligned = min_value + (steps * step)
    return min(max(aligned, min_value), max_value)


def _sync_widget_value(key: str, value: Any, *, force: bool = False) -> None:
    if force or key not in st.session_state:
        st.session_state[key] = value


def initialize_llm_settings_from_runtime(runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Initialize session LLM settings from backend defaults."""
    defaults = runtime_config.get("default_llm_settings", {})
    return {
        "model": defaults.get("model"),
        "temperature": float(defaults.get("temperature", 0.0)),
        "top_p": float(defaults.get("top_p", 0.2)),
        "max_tokens": int(defaults.get("max_tokens", 1200)),
    }


def normalize_llm_settings(runtime_config: dict[str, Any], current: dict[str, Any] | None) -> dict[str, Any]:
    """Clamp and snap LLM settings to values accepted by the frontend controls."""
    current = current or {}
    constraints = runtime_config.get("parameter_constraints", {})
    available_models = runtime_config.get("available_models", [])
    defaults = initialize_llm_settings_from_runtime(runtime_config)

    temperature_constraints = constraints.get("temperature", {"min": 0.0, "max": 1.0, "step": 0.05})
    top_p_constraints = constraints.get("top_p", {"min": 0.0, "max": 1.0, "step": 0.05})
    max_tokens_constraints = constraints.get("max_tokens", {"min": 64, "max": 4096, "step": 1})

    model = current.get("model", defaults.get("model"))
    if available_models and model not in available_models:
        model = available_models[0]

    return {
        "model": model,
        "temperature": float(
            _align_slider_value(
                float(current.get("temperature", defaults.get("temperature", 0.0))),
                float(temperature_constraints.get("min", 0.0)),
                float(temperature_constraints.get("max", 1.0)),
                float(temperature_constraints.get("step", 0.05)),
            )
        ),
        "top_p": float(
            _align_slider_value(
                float(current.get("top_p", defaults.get("top_p", 0.2))),
                float(top_p_constraints.get("min", 0.0)),
                float(top_p_constraints.get("max", 1.0)),
                float(top_p_constraints.get("step", 0.05)),
            )
        ),
        "max_tokens": int(
            _align_slider_value(
                float(current.get("max_tokens", defaults.get("max_tokens", 1200))),
                float(max_tokens_constraints.get("min", 64)),
                float(max_tokens_constraints.get("max", 4096)),
                float(max_tokens_constraints.get("step", 64)),
            )
        ),
    }


def render_llm_controls(runtime_config: dict[str, Any], current: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Render dynamic model controls from backend-provided runtime config."""
    constraints = runtime_config.get("parameter_constraints", {})
    available_models = runtime_config.get("available_models", [])

    model_key = "llm_model_control"
    temperature_key = "llm_temperature_control"
    top_p_key = "llm_top_p_control"
    max_tokens_key = "llm_max_tokens_control"

    reset_requested = bool(st.session_state.get("llm_reset_requested", False))
    if reset_requested:
        defaults = normalize_llm_settings(
            runtime_config,
            initialize_llm_settings_from_runtime(runtime_config),
        )
        _sync_widget_value(model_key, defaults["model"], force=True)
        _sync_widget_value(temperature_key, defaults["temperature"], force=True)
        _sync_widget_value(top_p_key, defaults["top_p"], force=True)
        _sync_widget_value(max_tokens_key, defaults["max_tokens"], force=True)
        st.session_state["llm_reset_requested"] = False
        effective_current = defaults
    else:
        # Respect any existing widget values from prior reruns so user interactions persist.
        effective_current = dict(current or {})
        if model_key in st.session_state:
            effective_current["model"] = st.session_state[model_key]
        if temperature_key in st.session_state:
            effective_current["temperature"] = st.session_state[temperature_key]
        if top_p_key in st.session_state:
            effective_current["top_p"] = st.session_state[top_p_key]
        if max_tokens_key in st.session_state:
            effective_current["max_tokens"] = st.session_state[max_tokens_key]

    normalized = normalize_llm_settings(runtime_config, effective_current)

    current_model = st.session_state.get(model_key)
    current_temperature = st.session_state.get(temperature_key)
    current_top_p = st.session_state.get(top_p_key)
    current_max_tokens = st.session_state.get(max_tokens_key)

    _sync_widget_value(
        model_key,
        normalized["model"],
        force=bool(available_models) and current_model not in available_models,
    )
    _sync_widget_value(
        temperature_key,
        normalized["temperature"],
        force=current_temperature is not None and float(current_temperature) != float(normalized["temperature"]),
    )
    _sync_widget_value(
        top_p_key,
        normalized["top_p"],
        force=current_top_p is not None and float(current_top_p) != float(normalized["top_p"]),
    )
    _sync_widget_value(
        max_tokens_key,
        normalized["max_tokens"],
        force=current_max_tokens is not None and int(current_max_tokens) != int(normalized["max_tokens"]),
    )

    with st.sidebar.expander("LLM behavior controls", expanded=False):
        st.caption("Grounded QA works best with low temperature and conservative sampling.")

        model = st.selectbox(
            "Model",
            options=available_models,
            key=model_key,
            help="Choose from backend-approved OpenRouter models for this session.",
        )

        t_c = constraints.get("temperature", {"min": 0.0, "max": 1.0, "step": 0.05})
        temperature_step = float(t_c.get("step", 0.05))
        temperature = st.slider(
            "Temperature",
            min_value=float(t_c.get("min", 0.0)),
            max_value=float(t_c.get("max", 1.0)),
            step=temperature_step,
            key=temperature_key,
            help="Lower values reduce randomness and improve grounding.",
        )

        p_c = constraints.get("top_p", {"min": 0.0, "max": 1.0, "step": 0.05})
        top_p_step = float(p_c.get("step", 0.05))
        top_p = st.slider(
            "Top-p",
            min_value=float(p_c.get("min", 0.0)),
            max_value=float(p_c.get("max", 1.0)),
            step=top_p_step,
            key=top_p_key,
            help="Nucleus sampling cap. Usually tune carefully with temperature.",
        )

        m_c = constraints.get("max_tokens", {"min": 64, "max": 4096, "step": 1})
        max_tokens_step = int(m_c.get("step", 1))
        max_tokens = st.slider(
            "Max tokens",
            min_value=int(m_c.get("min", 64)),
            max_value=int(m_c.get("max", 4096)),
            step=max_tokens_step,
            key=max_tokens_key,
            help="Upper bound for response length.",
        )

        reset_clicked = st.button("Reset model settings", use_container_width=True)

    new_settings = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    return normalize_llm_settings(runtime_config, new_settings), reset_clicked
