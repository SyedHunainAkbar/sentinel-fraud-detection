"""Tests for quant-risk analytics."""
import numpy as np

from sentinel.risk_quant.backtest import bootstrap_ci
from sentinel.risk_quant.loss_model import expected_shortfall, value_at_risk
from sentinel.risk_quant.stability import psi


def test_es_at_least_var():
    rng = np.random.default_rng(0)
    losses = rng.gamma(2, 100, 5000)
    var = value_at_risk(losses, 0.95)
    es = expected_shortfall(losses, 0.95)
    assert es >= var  # expected shortfall is a tail mean beyond VaR


def test_psi_zero_for_identical_and_positive_for_shifted():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 5000)
    same = rng.normal(0, 1, 5000)
    shifted = rng.normal(3, 1, 5000)
    assert psi(base, same) < 0.1
    assert psi(base, shifted) > 0.25


def test_bootstrap_ci_brackets_mean():
    vals = [100.0, 120.0, 90.0, 110.0, 105.0]
    ci = bootstrap_ci(vals, n_boot=500)
    assert ci["lower"] <= ci["mean"] <= ci["upper"]
