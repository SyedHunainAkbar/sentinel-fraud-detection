"""Tests for copilot evaluation (agreement with labels)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.copilot.evaluate_copilot import build_holdout_set, evaluate_copilot


def _make_sample_df(n=100, fraud_rate=0.1, seed=42):
    """Create a minimal transaction DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "trans_num": [f"t{i:04d}" for i in range(n)],
        "category": rng.choice(["grocery", "shopping_net"], n),
        "amt": rng.lognormal(3, 1, n),
        "is_fraud": (rng.random(n) < fraud_rate).astype(int),
        "cc_num": rng.integers(1000000, 9999999, n),
    })


class TestBuildHoldoutSet:
    def test_returns_correct_size(self):
        df = _make_sample_df(200, fraud_rate=0.2)
        holdout = build_holdout_set(df, n_flagged=30)
        assert len(holdout) <= 30

    def test_contains_mix_of_fraud_and_legit(self):
        df = _make_sample_df(200, fraud_rate=0.2)
        holdout = build_holdout_set(df, n_flagged=30)
        assert holdout["is_fraud"].sum() > 0
        assert (holdout["is_fraud"] == 0).sum() > 0

    def test_deterministic_with_same_seed(self):
        df = _make_sample_df(200, fraud_rate=0.2)
        h1 = build_holdout_set(df, n_flagged=20, seed=7)
        h2 = build_holdout_set(df, n_flagged=20, seed=7)
        pd.testing.assert_frame_equal(h1, h2)


class TestEvaluateCopilot:
    def test_returns_valid_metrics(self):
        df = _make_sample_df(50, fraud_rate=0.3, seed=0)
        # High-prob scorer for fraud transactions
        def score_fn(txn):
            return 0.95 if txn.get("is_fraud", 0) else 0.05

        metrics = evaluate_copilot(df, score_fn)
        assert "agreement_rate" in metrics
        assert "precision_escalate" in metrics
        assert "recall_escalate" in metrics
        assert "f1_escalate" in metrics
        assert "confusion" in metrics
        assert 0 <= metrics["agreement_rate"] <= 1
        assert metrics["n_evaluated"] == len(df)

    def test_perfect_scorer_has_high_agreement(self):
        df = _make_sample_df(30, fraud_rate=0.4, seed=1)
        # Perfect scorer gives high prob for fraud
        def score_fn(txn):
            return 0.99 if txn.get("is_fraud", 0) else 0.01

        metrics = evaluate_copilot(df, score_fn)
        # With perfect scoring, escalate should align with fraud
        assert metrics["agreement_rate"] >= 0.7

    def test_confusion_matrix_sums_to_n(self):
        df = _make_sample_df(40, fraud_rate=0.25, seed=3)

        def score_fn(txn):
            return 0.5

        metrics = evaluate_copilot(df, score_fn)
        cm = metrics["confusion"]
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(df)

    def test_recommendation_distribution_sums_to_n(self):
        df = _make_sample_df(25, fraud_rate=0.2, seed=5)

        def score_fn(txn):
            return min(0.99, txn.get("amt", 0) / 500.0)

        metrics = evaluate_copilot(df, score_fn)
        dist = metrics["recommendation_distribution"]
        total = dist["escalate"] + dist["clear"] + dist["request_info"]
        assert total == len(df)
