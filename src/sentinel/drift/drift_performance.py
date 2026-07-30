"""Rolling-window model performance tracking and degradation detection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from .. import config
from ..evaluation import expected_dollar_loss, precision_recall_at_budget


def rolling_window_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    unix_time: np.ndarray,
    threshold: float,
    n_windows: int | None = None,
) -> list[dict]:
    """Compute PR-AUC, expected loss, and recall@budget per time window.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels.
    y_prob : np.ndarray
        Predicted probabilities.
    amount : np.ndarray
        Transaction amounts.
    unix_time : np.ndarray
        Unix timestamps for temporal ordering.
    threshold : float
        Cost-optimal decision threshold.
    n_windows : int, optional
        Number of sequential windows. Defaults to ``config.DRIFT_N_WINDOWS``.

    Returns
    -------
    list[dict]
        Per-window metrics: window index, n_transactions, pr_auc, expected_loss,
        recall_at_budget.
    """
    if n_windows is None:
        n_windows = config.DRIFT_N_WINDOWS

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    amount = np.asarray(amount, dtype=float)
    unix_time = np.asarray(unix_time)

    order = np.argsort(unix_time)
    y_true = y_true[order]
    y_prob = y_prob[order]
    amount = amount[order]

    bounds = np.linspace(0, len(y_true), n_windows + 1, dtype=int)
    results: list[dict] = []

    for w in range(n_windows):
        lo, hi = int(bounds[w]), int(bounds[w + 1])
        if hi - lo < 2:
            continue
        y_w = y_true[lo:hi]
        p_w = y_prob[lo:hi]
        a_w = amount[lo:hi]

        # PR-AUC requires at least one positive
        if y_w.sum() > 0 and y_w.sum() < len(y_w):
            pr_auc = float(average_precision_score(y_w, p_w))
        else:
            pr_auc = None

        loss = expected_dollar_loss(y_w, p_w, a_w, threshold)
        _, recall = precision_recall_at_budget(y_w, p_w)

        results.append({
            "window": w + 1,
            "n_transactions": int(hi - lo),
            "pr_auc": pr_auc,
            "expected_loss": round(loss, 2),
            "recall_at_budget": round(recall, 4),
        })

    return results


def detect_degradation(
    window_metrics: list[dict],
    baseline_pr_auc: float,
    degradation_ratio: float | None = None,
) -> list[dict]:
    """Identify windows where PR-AUC drops below a fraction of the baseline.

    Parameters
    ----------
    window_metrics : list[dict]
        Output from :func:`rolling_window_metrics`.
    baseline_pr_auc : float
        Full-test-set PR-AUC as the reference level.
    degradation_ratio : float, optional
        Fraction of baseline below which we alert. Defaults to
        ``config.DEGRADATION_RATIO``.

    Returns
    -------
    list[dict]
        Alerts for degraded windows.
    """
    if degradation_ratio is None:
        degradation_ratio = config.DEGRADATION_RATIO

    threshold_auc = degradation_ratio * baseline_pr_auc
    alerts: list[dict] = []
    for wm in window_metrics:
        if wm["pr_auc"] is not None and wm["pr_auc"] < threshold_auc:
            alerts.append({
                "type": "performance_degradation",
                "window": wm["window"],
                "pr_auc": wm["pr_auc"],
                "message": (
                    f"PR-AUC dropped to {wm['pr_auc']:.3f} "
                    f"(< {degradation_ratio:.0%} of baseline {baseline_pr_auc:.3f})"
                ),
            })
    return alerts
