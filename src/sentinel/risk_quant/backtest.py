"""Out-of-time rolling backtest and bootstrap confidence intervals.

Validates the deployed policy like a strategy: fit the cost-optimal threshold on earlier
windows, measure realized dollars-saved P&L on strictly later windows, and report the
stability of that P&L across time.
"""
from __future__ import annotations

import numpy as np

from ..evaluation import expected_dollar_loss, optimize_threshold


def rolling_backtest(y_true, y_prob, amount, unix_time, n_windows=5, review_cost=None):
    """Walk-forward backtest.

    For each of ``n_windows`` sequential test windows, fit the cost-optimal threshold on
    all earlier data and evaluate dollars-saved (vs a naive always-legit rule) on the
    window. Returns per-window P&L plus stability statistics.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    amount = np.asarray(amount, dtype=float)
    order = np.argsort(np.asarray(unix_time))
    y_true, y_prob, amount = y_true[order], y_prob[order], amount[order]

    bounds = np.linspace(0, len(y_true), n_windows + 1, dtype=int)
    pnl = []
    for w in range(1, n_windows + 1):
        lo, hi = bounds[w - 1], bounds[w]
        if hi - lo < 2 or lo == 0:
            continue
        t_star, _ = optimize_threshold(y_true[:lo], y_prob[:lo], amount[:lo], review_cost)
        deployed = expected_dollar_loss(y_true[lo:hi], y_prob[lo:hi], amount[lo:hi],
                                        t_star, review_cost)
        naive = expected_dollar_loss(y_true[lo:hi], y_prob[lo:hi], amount[lo:hi],
                                     threshold=1.1, review_cost=review_cost)
        pnl.append(naive - deployed)
    pnl = np.asarray(pnl, dtype=float)
    return {
        "per_window_pnl": [float(x) for x in pnl],
        "mean_pnl": float(pnl.mean()) if pnl.size else 0.0,
        "pnl_volatility": float(pnl.std(ddof=0)) if pnl.size else 0.0,
        "worst_window": float(pnl.min()) if pnl.size else 0.0,
        "consistency": float((pnl > 0).mean()) if pnl.size else 0.0,
    }


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for the mean of ``values``."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, alpha / 2)),
        "upper": float(np.quantile(means, 1 - alpha / 2)),
    }
