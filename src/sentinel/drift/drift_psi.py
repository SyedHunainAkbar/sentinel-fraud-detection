"""Per-feature PSI computation and severity classification."""
from __future__ import annotations

import numpy as np

from .. import config
from ..risk_quant.stability import psi


def classify_severity(psi_value: float) -> str:
    """Classify a PSI value into a severity bucket.

    Parameters
    ----------
    psi_value : float
        Population Stability Index value.

    Returns
    -------
    str
        One of 'stable', 'warning', or 'critical'.
    """
    if psi_value >= config.PSI_CRITICAL_THRESHOLD:
        return "critical"
    if psi_value >= config.PSI_WARNING_THRESHOLD:
        return "warning"
    return "stable"


def compute_feature_psi(
    ref_features: np.ndarray,
    live_features: np.ndarray,
    feature_names: list[str],
    bins: int = 10,
) -> list[dict]:
    """Compute PSI for each numeric feature column.

    Parameters
    ----------
    ref_features : np.ndarray
        Reference (training) feature matrix, shape (n_ref, n_features).
    live_features : np.ndarray
        Live (current window) feature matrix, same column order.
    feature_names : list[str]
        Column names corresponding to feature indices.
    bins : int
        Number of bins for PSI histogram.

    Returns
    -------
    list[dict]
        Per-feature results: {'feature', 'psi', 'severity'}.
    """
    ref_features = np.asarray(ref_features, dtype=float)
    live_features = np.asarray(live_features, dtype=float)
    results: list[dict] = []
    for i, name in enumerate(feature_names):
        psi_val = psi(ref_features[:, i], live_features[:, i], bins=bins)
        results.append({
            "feature": name,
            "psi": round(float(psi_val), 6),
            "severity": classify_severity(psi_val),
        })
    return results


def compute_score_psi(
    ref_scores: np.ndarray,
    live_scores: np.ndarray,
    bins: int = 10,
) -> dict:
    """Compute PSI on model predicted-probability distributions.

    Parameters
    ----------
    ref_scores : np.ndarray
        Predicted probabilities on the reference (train) set.
    live_scores : np.ndarray
        Predicted probabilities on the live (test/production) set.
    bins : int
        Number of bins for PSI histogram.

    Returns
    -------
    dict
        {'psi': float, 'severity': str}.
    """
    psi_val = psi(np.asarray(ref_scores, dtype=float),
                  np.asarray(live_scores, dtype=float), bins=bins)
    return {
        "psi": round(float(psi_val), 6),
        "severity": classify_severity(psi_val),
    }
