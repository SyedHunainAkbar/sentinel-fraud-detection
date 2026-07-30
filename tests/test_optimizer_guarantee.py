"""Task 4.3: Prove the cost optimizer NEVER selects a worse threshold than 0.5.

Tests the guarantee across multiple data regimes:
- Perfectly separable scores
- Random (uninformative) scores
- Adversarial (inverted) scores
- Extreme imbalance (0.1% fraud)
- High-amount outlier fraud
- Uniform scores (no discrimination)

The invariant: optimize_threshold always finds cost <= cost(0.5).
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel.evaluation import expected_dollar_loss, optimize_threshold


def _assert_optimizer_beats_half(y, prob, amount, label=""):
    """Core assertion: optimized threshold never worse than naive 0.5."""
    t_star, curve = optimize_threshold(y, prob, amount)
    loss_star = expected_dollar_loss(y, prob, amount, t_star)
    loss_half = expected_dollar_loss(y, prob, amount, 0.5)
    assert loss_star <= loss_half, (
        f"[{label}] Optimizer loss ${loss_star:.2f} > naive-0.5 loss ${loss_half:.2f} "
        f"(threshold={t_star:.4f})"
    )
    return t_star, loss_star, loss_half


class TestOptimizerGuarantee:
    """The cost optimizer must NEVER produce a worse outcome than t=0.5."""

    def test_perfectly_separable_scores(self):
        """When fraud and legit are perfectly separated, optimizer finds optimal cost."""
        y = np.array([0]*100 + [1]*10)
        prob = np.array([0.01]*100 + [0.99]*10)
        amount = np.full(110, 100.0)
        t, loss_opt, loss_half = _assert_optimizer_beats_half(y, prob, amount, "separable")
        # Optimizer should be at least as good as 0.5
        assert loss_opt <= loss_half

    def test_random_uninformative_scores(self):
        """Even with random scores, optimizer is at least as good as 0.5."""
        rng = np.random.default_rng(42)
        y = (rng.random(1000) < 0.05).astype(int)
        prob = rng.random(1000)  # uninformative
        amount = rng.exponential(200, 1000)
        _assert_optimizer_beats_half(y, prob, amount, "random")

    def test_adversarial_inverted_scores(self):
        """When scores are anti-correlated (worst model), optimizer still >= 0.5."""
        rng = np.random.default_rng(7)
        y = (rng.random(500) < 0.1).astype(int)
        # Inverted: fraud gets low scores, legit gets high
        prob = np.where(y == 1, rng.uniform(0, 0.3, 500), rng.uniform(0.7, 1.0, 500))
        amount = rng.gamma(3, 100, 500)
        _assert_optimizer_beats_half(y, prob, amount, "adversarial")

    def test_extreme_imbalance(self):
        """0.1% fraud rate — optimizer still finds cost <= 0.5 default."""
        rng = np.random.default_rng(99)
        n = 5000
        y = np.zeros(n, dtype=int)
        y[rng.choice(n, size=5, replace=False)] = 1
        prob = np.clip(0.3 * rng.random(n) + 0.5 * y, 0, 1)
        amount = rng.lognormal(4, 1, n)
        _assert_optimizer_beats_half(y, prob, amount, "extreme_imbalance")

    def test_high_amount_outlier_fraud(self):
        """Single massive fraud transaction — optimizer accounts for it."""
        rng = np.random.default_rng(13)
        y = np.array([0]*200 + [1]*5)
        prob = np.array([rng.uniform(0, 0.4) for _ in range(200)] +
                        [rng.uniform(0.6, 0.95) for _ in range(5)])
        amount = np.array([50.0]*200 + [10_000.0]*5)  # 5 huge frauds
        t, loss_opt, loss_half = _assert_optimizer_beats_half(y, prob, amount, "outlier")
        # With huge fraud amounts, optimizer should eagerly alert
        assert t < 0.5

    def test_uniform_scores_no_discrimination(self):
        """All predictions identical — optimizer can't do worse than 0.5."""
        y = np.array([0]*90 + [1]*10)
        prob = np.full(100, 0.5)  # constant predictions
        amount = np.full(100, 100.0)
        _assert_optimizer_beats_half(y, prob, amount, "uniform")

    def test_all_fraud(self):
        """Edge case: every transaction is fraud."""
        y = np.ones(50, dtype=int)
        prob = np.linspace(0.1, 0.9, 50)
        amount = np.full(50, 200.0)
        _assert_optimizer_beats_half(y, prob, amount, "all_fraud")

    def test_no_fraud(self):
        """Edge case: no fraud at all — optimal is to alert nothing."""
        y = np.zeros(100, dtype=int)
        prob = np.linspace(0, 1, 100)
        amount = np.full(100, 50.0)
        t, loss_opt, loss_half = _assert_optimizer_beats_half(y, prob, amount, "no_fraud")
        # With no fraud, alerting is pure cost — threshold should be high
        assert t > 0.5

    @pytest.mark.parametrize("seed", range(10))
    def test_randomized_property(self, seed):
        """Property test across 10 random seeds — guarantee holds universally."""
        rng = np.random.default_rng(seed + 1000)
        n = rng.integers(50, 2000)
        fraud_rate = rng.uniform(0.005, 0.15)
        y = (rng.random(n) < fraud_rate).astype(int)
        if y.sum() == 0:
            y[0] = 1  # ensure at least one fraud
        # Scores with variable quality
        noise = rng.uniform(0.1, 0.9)
        prob = np.clip(noise * rng.random(n) + (1 - noise) * y, 0, 1)
        amount = rng.lognormal(3, 1.5, n)
        _assert_optimizer_beats_half(y, prob, amount, f"random_seed_{seed}")
