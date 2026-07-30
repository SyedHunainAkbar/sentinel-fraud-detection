"""Tests for live scoring module."""
from __future__ import annotations

import pandas as pd
import pytest

from sentinel.serving.live import (
    EXAMPLES,
    heuristic_score,
    load_scorer,
    score_frame,
    validate_columns,
)


class TestLoadScorer:
    def test_committed_model_loads(self):
        """The app_assets/ model loads successfully."""
        model, fb = load_scorer()
        assert model is not None, "xgboost.joblib failed to load from app_assets/"
        assert fb is not None, "feature_builder.joblib failed to load from app_assets/"

    def test_model_has_predict_proba(self):
        model, _ = load_scorer()
        assert hasattr(model, "predict_proba")


class TestScoreFrame:
    def test_high_risk_scores_higher_than_low_risk(self):
        """The high-risk example must score strictly higher than the low-risk one."""
        model, fb = load_scorer()
        if model is None:
            pytest.skip("Model not available in app_assets/")

        low_txn = EXAMPLES[0]["txn"]   # Low risk
        high_txn = EXAMPLES[2]["txn"]  # High risk

        df = pd.DataFrame([low_txn, high_txn])
        probs = score_frame(df, model, fb)

        assert probs[1] > probs[0], (
            f"High-risk example ({probs[1]:.4f}) should score higher than "
            f"low-risk example ({probs[0]:.4f})"
        )

    def test_returns_probabilities_in_unit_interval(self):
        model, fb = load_scorer()
        if model is None:
            pytest.skip("Model not available in app_assets/")

        df = pd.DataFrame([ex["txn"] for ex in EXAMPLES])
        probs = score_frame(df, model, fb)
        assert all(0 <= p <= 1 for p in probs)
        assert len(probs) == len(EXAMPLES)


class TestHeuristicScore:
    def test_higher_amount_scores_higher(self):
        low = heuristic_score({"amt": 10, "trans_date_trans_time": "2020-01-01 14:00:00"})
        high = heuristic_score({"amt": 900, "trans_date_trans_time": "2020-01-01 14:00:00"})
        assert high > low

    def test_night_time_adds_risk(self):
        day = heuristic_score({"amt": 100, "trans_date_trans_time": "2020-01-01 14:00:00"})
        night = heuristic_score({"amt": 100, "trans_date_trans_time": "2020-01-01 02:00:00"})
        assert night > day

    def test_bounded_zero_one(self):
        score = heuristic_score({"amt": 50000, "trans_date_trans_time": "2020-01-01 03:00:00"})
        assert 0 < score <= 0.99


class TestValidateColumns:
    def test_complete_dataframe_passes(self):
        df = pd.DataFrame([EXAMPLES[0]["txn"]])
        missing = validate_columns(df)
        assert missing == []

    def test_missing_columns_detected(self):
        df = pd.DataFrame({"amt": [100], "category": ["grocery"]})
        missing = validate_columns(df)
        assert "cc_num" in missing
        assert "trans_date_trans_time" in missing
        assert len(missing) > 5
