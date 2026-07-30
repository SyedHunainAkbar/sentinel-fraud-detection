"""Reliability-curve calibration and Brier score decomposition.

Treats the fraud score as a PD (probability of default/fraud) estimate and evaluates
how well-calibrated it is — a credit-risk staple.
"""
from __future__ import annotations

import numpy as np


def reliability_curve(y_true, y_prob, n_bins: int = 10) -> dict:
    """Compute a reliability (calibration) curve.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_prob : array-like
        Predicted probabilities in [0, 1].
    n_bins : int
        Number of equal-width bins across [0, 1].

    Returns
    -------
    dict
        Keys: bin_edges, mean_predicted, fraction_positive, bin_counts.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted: list[float] = []
    fraction_positive: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            mean_predicted.append(float((lo + hi) / 2))
            fraction_positive.append(0.0)
        else:
            mean_predicted.append(float(y_prob[mask].mean()))
            fraction_positive.append(float(y_true[mask].mean()))

    return {
        "bin_edges": [float(e) for e in bin_edges],
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "bin_counts": bin_counts,
    }


def brier_decomposition(y_true, y_prob, n_bins: int = 10) -> dict:
    """Brier score decomposition into reliability, resolution, and uncertainty.

    Uses the Murphy (1973) decomposition:
        Brier = Reliability - Resolution + Uncertainty

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_prob : array-like
        Predicted probabilities in [0, 1].
    n_bins : int
        Number of equal-width bins for grouping predictions.

    Returns
    -------
    dict
        brier_score, reliability, resolution, uncertainty, and per-bin details.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    # Overall Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Base rate (climatological probability)
    base_rate = float(y_true.mean())
    uncertainty = base_rate * (1.0 - base_rate)

    # Bin predictions
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        mean_pred_k = float(y_prob[mask].mean())
        obs_freq_k = float(y_true[mask].mean())

        reliability += n_k * (mean_pred_k - obs_freq_k) ** 2
        resolution += n_k * (obs_freq_k - base_rate) ** 2

    reliability /= n
    resolution /= n

    return {
        "brier_score": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "base_rate": base_rate,
        "n_bins": n_bins,
    }
