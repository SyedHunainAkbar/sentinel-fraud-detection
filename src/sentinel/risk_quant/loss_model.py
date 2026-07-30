"""Loss-distribution risk analytics for undetected fraud.

Frames residual (undetected) fraud loss as a random variable and quantifies its tail with
Value-at-Risk and Expected Shortfall, estimated by block bootstrap and cross-checked with
a Monte Carlo Bernoulli-detection simulation.
"""
from __future__ import annotations

import numpy as np


def value_at_risk(losses, alpha: float = 0.95) -> float:
    """Alpha-quantile of a loss array (e.g. 95% VaR)."""
    return float(np.quantile(np.asarray(losses, dtype=float), alpha))


def expected_shortfall(losses, alpha: float = 0.95) -> float:
    """Mean loss in the worst (1-alpha) tail — coherent risk measure."""
    losses = np.asarray(losses, dtype=float)
    var = value_at_risk(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else var


def bootstrap_loss_distribution(y_true, y_prob, amount, threshold,
                                period_size=1000, n_boot=2000, seed=42):
    """Empirical residual-loss distribution via block bootstrap over transactions.

    Each bootstrap draw samples ``period_size`` transactions with replacement and sums the
    amounts of frauds that fall below the alert threshold (i.e. undetected).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    amount = np.asarray(amount, dtype=float)
    n = len(y_true)
    undetected = (y_true == 1) & (y_prob < threshold)
    per_txn_loss = np.where(undetected, amount, 0.0)
    idx = rng.integers(0, n, size=(n_boot, min(period_size, n)))
    return per_txn_loss[idx].sum(axis=1)


def monte_carlo_loss(y_prob, amount, threshold, n_sims=2000, seed=42):
    """Monte Carlo residual loss treating fraud occurrence as Bernoulli(p=prob).

    A cross-check on the bootstrap: simulate whether each transaction is fraud using the
    model's own probability, then sum amounts for simulated frauds below threshold.

    Only transactions below the threshold can contribute undetected loss, so we filter
    first to keep memory usage manageable on large datasets.
    """
    rng = np.random.default_rng(seed)
    y_prob = np.asarray(y_prob)
    amount = np.asarray(amount, dtype=float)

    # Only sub-threshold transactions can be undetected
    below_mask = y_prob < threshold
    prob_below = y_prob[below_mask]
    amt_below = amount[below_mask]

    if len(prob_below) == 0:
        return np.zeros(n_sims)

    # Process in chunks to limit memory (max ~200M elements per chunk)
    chunk_size = max(1, 200_000_000 // len(prob_below))
    results = []
    remaining = n_sims
    while remaining > 0:
        batch = min(chunk_size, remaining)
        draws = rng.random((batch, len(prob_below))) < prob_below
        results.append(draws.astype(np.float32) @ amt_below)
        remaining -= batch

    return np.concatenate(results)


def loss_risk_summary(y_true, y_prob, amount, threshold, alpha=0.95, seed=42) -> dict:
    """VaR/ES summary from both the bootstrap and Monte Carlo estimators."""
    boot = bootstrap_loss_distribution(y_true, y_prob, amount, threshold, seed=seed)
    mc = monte_carlo_loss(y_prob, amount, threshold, seed=seed)
    return {
        "alpha": alpha,
        "threshold": float(threshold),
        "bootstrap": {
            "mean": float(boot.mean()),
            "var": value_at_risk(boot, alpha),
            "expected_shortfall": expected_shortfall(boot, alpha),
        },
        "monte_carlo": {
            "mean": float(mc.mean()),
            "var": value_at_risk(mc, alpha),
            "expected_shortfall": expected_shortfall(mc, alpha),
        },
    }
