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
        "name": "Low risk — small grocery purchase near home",
        "txn": {
            "trans_date_trans_time": "2020-03-15 14:22:00",
            "cc_num": 4532015112830366,
            "merchant": "fraud_Kutch-Heller",
            "category": "grocery_pos",
            "amt": 23.50,
            "gender": "F",
            "city": "Phoenix",
            "state": "AZ",
            "zip": 85001,
            "lat": 33.4484,
            "long": -112.0740,
            "city_pop": 1680992,
            "job": "Accountant",
            "dob": "1985-06-15",
            "trans_num": "demo_low_001",
            "unix_time": 1584280920,
            "merch_lat": 33.4500,
            "merch_long": -112.0800,
        },
    },
    {
        "name": "Elevated risk — online purchase, moderate amount",
        "txn": {
            "trans_date_trans_time": "2020-03-15 23:45:00",
            "cc_num": 4532015112830366,
            "merchant": "fraud_Stiedemann Ltd",
            "category": "shopping_net",
            "amt": 312.00,
            "gender": "F",
            "city": "Phoenix",
            "state": "AZ",
            "zip": 85001,
            "lat": 33.4484,
            "long": -112.0740,
            "city_pop": 1680992,
            "job": "Accountant",
            "dob": "1985-06-15",
            "trans_num": "demo_med_002",
            "unix_time": 1584314700,
            "merch_lat": 40.7128,
            "merch_long": -74.0060,
        },
    },
    {
        "name": "High risk — large night purchase, merchant far from home",
        "txn": {
            "trans_date_trans_time": "2020-03-16 02:10:00",
            "cc_num": 4532015112830366,
            "merchant": "fraud_Bergnaum Inc",
            "category": "shopping_net",
            "amt": 947.80,
            "gender": "F",
            "city": "Phoenix",
            "state": "AZ",
            "zip": 85001,
            "lat": 33.4484,
            "long": -112.0740,
            "city_pop": 1680992,
            "job": "Accountant",
            "dob": "1985-06-15",
            "trans_num": "demo_high_003",
            "unix_time": 1584323400,
            "merch_lat": 51.5074,
            "merch_long": -0.1278,
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
