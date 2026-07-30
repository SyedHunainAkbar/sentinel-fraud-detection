"""Tests for drift rolling-window performance and degradation detection."""
import numpy as np

from sentinel.drift.drift_performance import detect_degradation, rolling_window_metrics


def _make_test_data(n=600, seed=42):
    """Generate synthetic test data with known structure."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.02).astype(int)  # ~2% fraud
    # Good model: frauds get high scores, legit get low
    y_prob = np.where(y_true == 1, rng.uniform(0.6, 0.99, n), rng.uniform(0.0, 0.3, n))
    amount = rng.exponential(100, n)
    unix_time = np.arange(n, dtype=float)  # sequential
    return y_true, y_prob, amount, unix_time


def test_rolling_window_metrics_returns_correct_windows():
    y_true, y_prob, amount, unix_time = _make_test_data(n=600)
    results = rolling_window_metrics(
        y_true, y_prob, amount, unix_time, threshold=0.5, n_windows=3
    )
    assert len(results) == 3
    for wm in results:
        assert "window" in wm
        assert "n_transactions" in wm
        assert "pr_auc" in wm
        assert "expected_loss" in wm
        assert "recall_at_budget" in wm
        assert wm["n_transactions"] == 200


def test_rolling_window_metrics_respects_time_order():
    y_true, y_prob, amount, unix_time = _make_test_data(n=400)
    # Shuffle all arrays together; results should be identical to sorted input
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(y_true))
    results_fwd = rolling_window_metrics(
        y_true, y_prob, amount, unix_time, threshold=0.5, n_windows=2
    )
    results_shuf = rolling_window_metrics(
        y_true[perm], y_prob[perm], amount[perm], unix_time[perm],
        threshold=0.5, n_windows=2,
    )
    # Should produce same results regardless of input order
    assert results_fwd[0]["expected_loss"] == results_shuf[0]["expected_loss"]
    assert results_fwd[1]["expected_loss"] == results_shuf[1]["expected_loss"]


def test_detect_degradation_no_alerts_when_healthy():
    window_metrics = [
        {"window": 1, "pr_auc": 0.90, "expected_loss": 100, "recall_at_budget": 0.8},
        {"window": 2, "pr_auc": 0.85, "expected_loss": 120, "recall_at_budget": 0.75},
    ]
    alerts = detect_degradation(window_metrics, baseline_pr_auc=0.90, degradation_ratio=0.80)
    assert alerts == []


def test_detect_degradation_flags_low_pr_auc():
    window_metrics = [
        {"window": 1, "pr_auc": 0.90, "expected_loss": 100, "recall_at_budget": 0.8},
        {"window": 2, "pr_auc": 0.60, "expected_loss": 500, "recall_at_budget": 0.4},
        {"window": 3, "pr_auc": 0.50, "expected_loss": 800, "recall_at_budget": 0.3},
    ]
    alerts = detect_degradation(window_metrics, baseline_pr_auc=0.90, degradation_ratio=0.80)
    assert len(alerts) == 2
    assert alerts[0]["window"] == 2
    assert alerts[0]["type"] == "performance_degradation"
    assert alerts[1]["window"] == 3


def test_detect_degradation_handles_none_pr_auc():
    window_metrics = [
        {"window": 1, "pr_auc": None, "expected_loss": 100, "recall_at_budget": 0.0},
    ]
    alerts = detect_degradation(window_metrics, baseline_pr_auc=0.90)
    assert alerts == []
