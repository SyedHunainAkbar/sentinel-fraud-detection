"""Live scoring logic for the interactive Streamlit dashboard.

Pure Python — no Streamlit imports. Provides model loading, scoring, validation,
and example transactions for the interactive tab.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Path to committed lightweight model assets
APP_ASSETS = Path(__file__).resolve().parents[3] / "app_assets"

# Required columns for the FeatureBuilder (matches config.REQUIRED_COLUMNS minus is_fraud)
REQUIRED_INPUT_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "city", "state", "zip", "lat", "long", "city_pop", "job", "dob",
    "trans_num", "unix_time", "merch_lat", "merch_long",
]

# Three editable example transactions covering low / elevated / high risk
EXAMPLES: list[dict[str, Any]] = [
    {
        "name": "Everyday grocery — local, midday (low risk)",
        "txn": {
            "trans_date_trans_time": "2020-06-15 13:20:00",
            "cc_num": 4012000033330026,
            "merchant": "fraud_Kutch-Heller",
            "category": "grocery_pos",
            "amt": 42.10,
            "gender": "F",
            "city": "New York",
            "state": "NY",
            "zip": 10001,
            "lat": 40.71,
            "long": -74.00,
            "city_pop": 800000,
            "job": "Accountant",
            "dob": "1985-03-02",
            "trans_num": "demo_low_001",
            "unix_time": 1592227200,
            "merch_lat": 40.73,
            "merch_long": -74.02,
        },
    },
    {
        "name": "Large purchase — but local & daytime (model stays calm, ~19%)",
        "txn": {
            "trans_date_trans_time": "2020-06-15 14:10:00",
            "cc_num": 4012000033330026,
            "merchant": "fraud_Stiedemann Ltd",
            "category": "shopping_net",
            "amt": 900.00,
            "gender": "F",
            "city": "New York",
            "state": "NY",
            "zip": 10001,
            "lat": 40.71,
            "long": -74.00,
            "city_pop": 800000,
            "job": "Accountant",
            "dob": "1985-03-02",
            "trans_num": "demo_med_002",
            "unix_time": 1592231400,
            "merch_lat": 40.73,
            "merch_long": -74.02,
        },
    },
    {
        "name": "High-value online — far from home, 3am (high risk)",
        "txn": {
            "trans_date_trans_time": "2020-06-16 03:05:00",
            "cc_num": 4012000033330026,
            "merchant": "fraud_Bergnaum Inc",
            "category": "misc_net",
            "amt": 800.00,
            "gender": "F",
            "city": "New York",
            "state": "NY",
            "zip": 10001,
            "lat": 40.71,
            "long": -74.00,
            "city_pop": 800000,
            "job": "Accountant",
            "dob": "1985-03-02",
            "trans_num": "demo_high_003",
            "unix_time": 1592276700,
            "merch_lat": 29.76,
            "merch_long": -95.36,
        },
    },
]


def load_scorer() -> tuple[Any, Any]:
    """Load the committed XGBoost model and FeatureBuilder from app_assets/.

    Returns
    -------
    tuple[model, feature_builder]
        (None, None) if loading fails — app should fall back to heuristic.
    """
    try:
        import joblib

        model_path = APP_ASSETS / "xgboost.joblib"
        fb_path = APP_ASSETS / "feature_builder.joblib"
        if not model_path.exists() or not fb_path.exists():
            logger.warning("Model assets not found in %s", APP_ASSETS)
            return None, None
        model = joblib.load(model_path)
        fb = joblib.load(fb_path)
        return model, fb
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load model assets")
        return None, None


def score_frame(
    df: pd.DataFrame,
    model: Any,
    fb: Any,
) -> np.ndarray:
    """Score a DataFrame of transactions using the trained model.

    Parameters
    ----------
    df : DataFrame
        Raw transactions with required schema columns.
    model : sklearn Pipeline
        Trained XGBoost pipeline.
    fb : FeatureBuilder
        Fitted feature builder.

    Returns
    -------
    ndarray
        Fraud probabilities for each row.
    """
    X, _, _ = fb.transform(df)
    return model.predict_proba(X)[:, 1]


def heuristic_score(txn: dict) -> float:
    """Transparent fallback scorer when model is unavailable.

    Uses amount and night-time as simple risk signals.
    """
    amt = float(txn.get("amt", 0))
    # Parse hour if possible
    try:
        hour = int(str(txn.get("trans_date_trans_time", ""))[-8:-6])
    except (ValueError, IndexError):
        hour = 12
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    # Simple logistic-like combination
    score = min(0.99, (amt / 2000.0) + (0.15 * is_night))
    return max(0.01, score)


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return list of required columns missing from the uploaded DataFrame."""
    return [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
