from __future__ import annotations

from typing import Any

import streamlit as st


def initialize_llm_settings_from_runtime(runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Initialize session LLM settings from backend defaults."""
    defaults = runtime_config.get("default_llm_settings", {})
    return {
        "model": defaults.get("model"),
        "temperature": float(defaults.get("temperature", 0.0)),
        "top_p": float(defaults.get("top_p", 0.2)),
        "max_tokens": int(defaults.get("max_tokens", 1200)),
    }


def render_llm_controls(runtime_config: dict[str, Any], current: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Render dynamic model controls from backend-provided runtime config."""
    constraints = runtime_config.get("parameter_constraints", {})
    available_models = runtime_config.get("available_models", [])
    defaults = runtime_config.get("default_llm_settings", {})

    with st.sidebar.expander("LLM behavior controls", expanded=False):
        st.caption("Grounded QA works best with low temperature and conservative sampling.")

        model = st.selectbox(
            "Model",
            options=available_models,
            index=max(0, available_models.index(current.get("model"))) if current.get("model") in available_models else 0,
            help="Choose from backend-approved OpenRouter models for this session.",
        )

        t_c = constraints.get("temperature", {"min": 0.0, "max": 1.0, "step": 0.05})
        temperature = st.slider(
            "Temperature",
            min_value=float(t_c.get("min", 0.0)),
            max_value=float(t_c.get("max", 1.0)),
            value=float(current.get("temperature", defaults.get("temperature", 0.0))),
            step=float(t_c.get("step", 0.05)),
            help="Lower values reduce randomness and improve grounding.",
        )

        p_c = constraints.get("top_p", {"min": 0.0, "max": 1.0, "step": 0.05})
        top_p = st.slider(
            "Top-p",
            min_value=float(p_c.get("min", 0.0)),
            max_value=float(p_c.get("max", 1.0)),
            value=float(current.get("top_p", defaults.get("top_p", 0.2))),
            step=float(p_c.get("step", 0.05)),
            help="Nucleus sampling cap. Usually tune carefully with temperature.",
        )

        m_c = constraints.get("max_tokens", {"min": 64, "max": 4096, "step": 64})
        max_tokens = st.slider(
            "Max tokens",
            min_value=int(m_c.get("min", 64)),
            max_value=int(m_c.get("max", 4096)),
            value=int(current.get("max_tokens", defaults.get("max_tokens", 1200))),
            step=int(m_c.get("step", 64)),
            help="Upper bound for response length.",
        )

        reset_clicked = st.button("Reset model settings", use_container_width=True)

    new_settings = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    return new_settings, reset_clicked
