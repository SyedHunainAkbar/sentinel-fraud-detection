"""Tests for hyperparameter search module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.hyperparam import PARAM_DISTRIBUTIONS, hyperparameter_search


@pytest.fixture
def small_training_data():
    """Small dataset for fast search tests."""
    rng = np.random.default_rng(42)
    n = 150
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
        "category": rng.choice(["grocery", "shopping"], n),
        "gender": rng.choice(["M", "F"], n),
        "state": rng.choice(["TX", "CA"], n),
    }
    X = pd.DataFrame(df_dict)
    y = pd.Series((rng.random(n) < 0.15).astype(int))
    return X, y


class TestHyperparamSearch:
    def test_returns_valid_result_structure(self, small_training_data):
        X, y = small_training_data
        result = hyperparameter_search(X, y, n_iter=2, cv=2, n_jobs=1, verbose=0)
        assert "best_params" in result
        assert "best_score" in result
        assert "best_estimator" in result
        assert "top5" in result
        assert isinstance(result["best_score"], float)
        assert 0 <= result["best_score"] <= 1

    def test_best_score_is_positive(self, small_training_data):
        X, y = small_training_data
        result = hyperparameter_search(X, y, n_iter=2, cv=2, n_jobs=1, verbose=0)
        assert result["best_score"] > 0

    def test_param_distributions_has_expected_keys(self):
        expected = {"clf__n_estimators", "clf__max_depth", "clf__learning_rate",
                    "clf__subsample", "clf__colsample_bytree"}
        assert expected.issubset(set(PARAM_DISTRIBUTIONS.keys()))
