"""Streamlit Community Cloud entry point.

Sets up the Python path, bridges st.secrets into os.environ (so optional API keys
like ANTHROPIC_API_KEY upgrade the copilot from deterministic fallback to LLM-powered
dispositions), then calls the dashboard render function.
"""
import os
import sys
from pathlib import Path

# Add src/ to Python path so the sentinel package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Bridge Streamlit secrets into environment variables (if configured in Cloud)
try:
    import streamlit as st

    for key in ("ANTHROPIC_API_KEY", "AWS_DEFAULT_REGION", "SENTINEL_RETRIEVER"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:  # noqa: BLE001
    pass  # secrets not configured — deterministic fallback is fine

from sentinel.serving.dashboard import render  # noqa: E402

render()
