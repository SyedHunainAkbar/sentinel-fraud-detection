"""Cost-sensitive evaluation: dollar-loss threshold optimization and metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config


def expected_dollar_loss(y_true, y_prob, amount, threshold, review_cost=None):
    """Total dollar cost at ``threshold`` per the risk.md cost matrix.

    cost = review_cost * (#alerts) + sum(amount of missed frauds)
    """
    review_cost = config.REVIEW_COST if review_cost is None else review_cost
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype=float)
    alert = np.asarray(y_prob) >= threshold
    review = review_cost * alert.sum()
    missed = amount[(~alert) & (y_true == 1)].sum()
    return float(review + missed)


def optimize_threshold(y_true, y_prob, amount, review_cost=None):
    """Return (best_threshold, curve) minimizing expected dollar loss."""
    y_prob = np.asarray(y_prob)
    grid = np.unique(np.concatenate([y_prob, np.linspace(0, 1, 101)]))
    costs = [expected_dollar_loss(y_true, y_prob, amount, t, review_cost) for t in grid]
    best_i = int(np.argmin(costs))
    curve = [{"threshold": float(t), "cost": float(c)} for t, c in zip(grid, costs, strict=False)]
    return float(grid[best_i]), curve


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov separation between positive and negative score CDFs."""
    y_true = np.asarray(y_true)
    pos = np.sort(np.asarray(y_prob)[y_true == 1])
    neg = np.sort(np.asarray(y_prob)[y_true == 0])
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([pos, neg])))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def precision_recall_at_budget(y_true, y_prob, budget_frac=None):
    """Precision@k and recall when alerting the top ``budget_frac`` by score."""
    budget_frac = config.ALERT_BUDGET_FRAC if budget_frac is None else budget_frac
    y_true = np.asarray(y_true)
    k = max(int(len(y_true) * budget_frac), 1)
    top = np.argsort(y_prob)[::-1][:k]
    tp = y_true[top].sum()
    precision = tp / k
    recall = tp / max(y_true.sum(), 1)
    return float(precision), float(recall)


def evaluate(y_true, y_prob, amount, review_cost=None) -> dict:
    """Full metric bundle for one model."""
    y_true = np.asarray(y_true)
    t_star, curve = optimize_threshold(y_true, y_prob, amount, review_cost)
    prec_k, rec_b = precision_recall_at_budget(y_true, y_prob)
    naive = expected_dollar_loss(y_true, y_prob, amount, threshold=1.1, review_cost=review_cost)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)) if y_true.sum() else None,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if 0 < y_true.sum() < len(y_true) else None,
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)) if y_true.sum() else None,
        "precision_at_k": prec_k,
        "recall_at_budget": rec_b,
        "optimal_threshold": t_star,
        "expected_loss": expected_dollar_loss(y_true, y_prob, amount, t_star, review_cost),
        "naive_loss": naive,
        "cost_curve": curve,
    }
