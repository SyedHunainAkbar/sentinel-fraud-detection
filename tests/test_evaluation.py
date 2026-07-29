"""Tests for cost-sensitive evaluation."""
import numpy as np

from sentinel.evaluation import (
    expected_dollar_loss,
    ks_statistic,
    optimize_threshold,
)


def test_optimizer_beats_default_half():
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.1).astype(int)
    # scores correlated with y so a good threshold exists
    prob = np.clip(0.2 * rng.random(n) + 0.6 * y, 0, 1)
    amount = rng.gamma(2, 50, n)
    t_star, curve = optimize_threshold(y, prob, amount)
    loss_star = expected_dollar_loss(y, prob, amount, t_star)
    loss_half = expected_dollar_loss(y, prob, amount, 0.5)
    assert loss_star <= loss_half
    assert len(curve) > 10


def test_ks_within_unit_interval():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    ks = ks_statistic(y, prob)
    assert 0.0 <= ks <= 1.0
    assert ks > 0.5  # well separated
