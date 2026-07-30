"""SHAP-based explainability helpers for the Sentinel fraud model.

Provides global importance loading and local (single-transaction) explanations.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import config


def load_shap_summary() -> dict[str, Any]:
    """Load the pre-computed SHAP summary from reports/.

    Returns
    -------
    dict
        Contains 'global_importances' and 'local_explanations' keys.

    Raises
    ------
    FileNotFoundError
        If the SHAP summary has not been generated yet.
    """
    path = config.REPORTS_DIR / "shap_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP summary not found at {path}. Run the shap-explainability skill first."
        )
    with open(path) as f:
        return json.load(f)


def explain_transaction(
    model,
    feature_builder,
    txn_df: pd.DataFrame,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Generate a local SHAP explanation for a single transaction.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained pipeline with 'pre' (ColumnTransformer) and 'clf' (tree model) steps.
    feature_builder : FeatureBuilder
        Fitted feature builder instance.
    txn_df : pd.DataFrame
        Single-row DataFrame with raw transaction fields.
    top_k : int
        Number of top contributing features to return.

    Returns
    -------
    dict
        Keys: 'probability', 'base_value', 'top_contributions' (list of
        {'feature', 'shap_value', 'direction'} dicts).
    """
    import shap  # noqa: PLC0415 — lazy import to keep startup fast

    X_eng, _, _ = feature_builder.transform(txn_df)

    # Transform through the pipeline preprocessor
    preprocessor = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    X_transformed = preprocessor.transform(X_eng)

    # Get feature names from the preprocessor
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # fraud class

    sv = shap_values[0]  # single row
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    # Top-k by absolute SHAP value
    top_idx = np.argsort(np.abs(sv))[::-1][:top_k]
    contributions = []
    for i in top_idx:
        fname = feature_names[i] if i < len(feature_names) else f"f{i}"
        contributions.append({
            "feature": str(fname),
            "shap_value": round(float(sv[i]), 4),
            "direction": "fraud" if sv[i] > 0 else "legit",
        })

    prob = float(clf.predict_proba(X_transformed)[:, 1][0])
    return {
        "probability": round(prob, 4),
        "base_value": round(float(base_value), 4),
        "top_contributions": contributions,
    }
