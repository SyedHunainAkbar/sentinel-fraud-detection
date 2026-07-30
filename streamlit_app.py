"""Streamlit Community Cloud entry point.

This file must be at the repo root for Streamlit Cloud to find it.
It sets up the path and imports the dashboard module.
"""
import sys
from pathlib import Path

# Add src/ to Python path so sentinel package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the dashboard (executes on import due to Streamlit's execution model)
from sentinel.serving import dashboard  # noqa: F401, E402
