"""Population Stability Index (PSI) — a credit-risk drift staple."""
from __future__ import annotations

import numpy as np


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between a reference and a live sample.

    PSI < 0.10 : no significant shift
    0.10-0.25  : moderate shift, monitor
    > 0.25     : material shift, investigate/revalidate
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, bins=quantiles)[0] / max(len(expected), 1)
    a_perc = np.histogram(actual, bins=quantiles)[0] / max(len(actual), 1)
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def feature_psi(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    bins: int = 10,
) -> dict[str, float]:
    """Compute per-feature PSI between train and test feature matrices.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, n_features)
        Training feature matrix (reference distribution).
    X_test : ndarray of shape (n_test, n_features)
        Test feature matrix (live/actual distribution).
    feature_names : list[str]
        Column names corresponding to each feature column.
    bins : int
        Number of bins for the PSI histogram.

    Returns
    -------
    dict[str, float]
        Mapping of feature name to its PSI value.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    result = {}
    for i, name in enumerate(feature_names):
        result[name] = psi(X_train[:, i], X_test[:, i], bins=bins)
    return result
