"""Tests for model training module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models import train_baseline, train_isolation_forest, train_xgboost


@pytest.fixture
def training_data():
    """Minimal training data with required features."""
    rng = np.random.default_rng(42)
    n = 200
    df_dict = {
        "distance_km": rng.exponential(10, n),
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "is_night": rng.integers(0, 2, n),
        "age": rng.integers(18, 80, n),
        "log_amt": rng.normal(3, 1, n),
        "amt_z_by_cat": rng.normal(0, 1, n),
        "velocity_24h": rng.integers(0, 10, n),
        "city_pop_log": rng.normal(10, 2, n),
        "home_deviation_km": rng.exponential(50, n),
        "category": rng.choice(["grocery", "shopping", "gas"], n),
        "gender": rng.choice(["M", "F"], n),
        "state": rng.choice(["TX", "CA", "NY"], n),
    }
    X = pd.DataFrame(df_dict)
    y = pd.Series((rng.random(n) < 0.1).astype(int))
    return X, y


class TestTrainBaseline:
    def test_returns_model_with_predict_proba(self, training_data):
        X, y = training_data
        model = train_baseline(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_probabilities_sum_to_one(self, training_data):
        X, y = training_data
        model = train_baseline(X, y)
        probs = model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class TestTrainXGBoost:
    def test_returns_model_with_predict_proba(self, training_data):
        X, y = training_data
        model = train_xgboost(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_handles_pure_legit_gracefully(self, training_data):
        X, _ = training_data
        y = pd.Series(np.zeros(len(X), dtype=int))
        y.iloc[0] = 1  # need at least one positive
        model = train_xgboost(X, y)
        assert model.predict_proba(X).shape[1] == 2


class TestTrainIsolationForest:
    def test_returns_model_and_scorer(self, training_data):
        X, _ = training_data
        model, score_fn = train_isolation_forest(X)
        scores = score_fn(X)
        assert len(scores) == len(X)

    def test_scores_in_zero_one(self, training_data):
        X, _ = training_data
        _, score_fn = train_isolation_forest(X)
        scores = score_fn(X)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_anomalies_score_higher(self, training_data):
        """Injected outliers should generally score higher."""
        X, _ = training_data
        # Add obvious outliers
        outliers = X.iloc[:5].copy()
        outliers["distance_km"] = 5000.0
        outliers["log_amt"] = 10.0
        X_aug = pd.concat([X, outliers], ignore_index=True)
        _, score_fn = train_isolation_forest(X_aug)
        scores = score_fn(X_aug)
        # Outlier scores should be above median
        assert scores[-5:].mean() > np.median(scores)
