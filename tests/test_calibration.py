"""Tests for calibration and Brier decomposition."""
from __future__ import annotations

import numpy as np

from sentinel.risk_quant.calibration import brier_decomposition, reliability_curve


class TestReliabilityCurve:
    def test_returns_correct_structure(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        result = reliability_curve(y, prob, n_bins=5)
        assert "bin_edges" in result
        assert "mean_predicted" in result
        assert "fraction_positive" in result
        assert "bin_counts" in result
        assert len(result["bin_edges"]) == 6  # n_bins + 1
        assert len(result["mean_predicted"]) == 5

    def test_perfect_calibration(self):
        """If prob == y, fraction_positive should match mean_predicted."""
        rng = np.random.default_rng(0)
        n = 1000
        prob = rng.uniform(0, 1, n)
        y = (rng.random(n) < prob).astype(int)
        result = reliability_curve(y, prob, n_bins=10)
        # Approximate: for large n, they should be close
        for mp, fp, count in zip(result["mean_predicted"],
                                  result["fraction_positive"],
                                  result["bin_counts"],
                                  strict=False):
            if count > 30:
                assert abs(mp - fp) < 0.15  # within 15% for stochastic check


class TestBrierDecomposition:
    def test_components_sum_to_brier(self):
        """Brier = reliability - resolution + uncertainty."""
        rng = np.random.default_rng(1)
        y = (rng.random(500) < 0.2).astype(float)
        prob = np.clip(y + rng.normal(0, 0.2, 500), 0, 1)
        result = brier_decomposition(y, prob, n_bins=10)
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        # Should be close to Brier (approximate due to binning)
        assert abs(reconstructed - result["brier_score"]) < 0.01

    def test_perfect_model_has_zero_reliability(self):
        """A perfect model (prob==y) has reliability near 0."""
        y = np.array([0.0]*50 + [1.0]*50)
        prob = np.array([0.0]*50 + [1.0]*50)
        result = brier_decomposition(y, prob)
        assert result["brier_score"] == 0.0
        assert result["reliability"] == 0.0

    def test_uncertainty_equals_base_rate_variance(self):
        """Uncertainty = p*(1-p) where p is base rate."""
        y = np.array([0]*80 + [1]*20, dtype=float)
        prob = np.full(100, 0.5)
        result = brier_decomposition(y, prob)
        expected_uncertainty = 0.2 * 0.8  # base_rate * (1 - base_rate)
        assert abs(result["uncertainty"] - expected_uncertainty) < 1e-10
