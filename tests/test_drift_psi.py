"""Tests for drift PSI computation and severity classification."""
import numpy as np

from sentinel.drift.drift_psi import (
    classify_severity,
    compute_feature_psi,
    compute_score_psi,
)


def test_classify_severity_stable():
    assert classify_severity(0.05) == "stable"
    assert classify_severity(0.0) == "stable"
    assert classify_severity(0.099) == "stable"


def test_classify_severity_warning():
    assert classify_severity(0.10) == "warning"
    assert classify_severity(0.15) == "warning"
    assert classify_severity(0.249) == "warning"


def test_classify_severity_critical():
    assert classify_severity(0.25) == "critical"
    assert classify_severity(0.50) == "critical"
    assert classify_severity(1.0) == "critical"


def test_compute_feature_psi_identical_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, (5000, 3))
    live = rng.normal(0, 1, (5000, 3))
    names = ["feat_a", "feat_b", "feat_c"]
    results = compute_feature_psi(ref, live, names)
    assert len(results) == 3
    for r in results:
        assert r["severity"] == "stable"
        assert r["psi"] < 0.10


def test_compute_feature_psi_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, (5000, 2))
    live = np.column_stack([
        rng.normal(0, 1, 5000),   # stable
        rng.normal(5, 1, 5000),   # heavily shifted
    ])
    names = ["stable_feat", "shifted_feat"]
    results = compute_feature_psi(ref, live, names)
    assert results[0]["severity"] == "stable"
    assert results[1]["severity"] == "critical"
    assert results[1]["psi"] > 0.25


def test_compute_score_psi_self_comparison():
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, 3000)
    result = compute_score_psi(scores, scores)
    assert result["psi"] < 0.01
    assert result["severity"] == "stable"


def test_compute_score_psi_shifted():
    rng = np.random.default_rng(42)
    ref = rng.beta(2, 5, 3000)
    live = rng.beta(5, 2, 3000)
    result = compute_score_psi(ref, live)
    assert result["psi"] > 0.25
    assert result["severity"] == "critical"
