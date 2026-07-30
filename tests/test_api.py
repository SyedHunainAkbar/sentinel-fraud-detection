"""Tests for the FastAPI scoring endpoint."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from sentinel.serving.api import app

client = TestClient(app)

# A minimal valid transaction payload matching the Transaction schema
VALID_TXN = {
    "trans_date_trans_time": "2019-06-21 12:14:00",
    "cc_num": 2291163933867244,
    "category": "grocery_pos",
    "amt": 45.0,
    "gender": "F",
    "state": "TX",
    "lat": 33.9659,
    "long": -80.9355,
    "city_pop": 333497,
    "dob": "1968-03-19",
    "unix_time": 1371816840,
    "merch_lat": 33.986391,
    "merch_long": -81.200714,
}


class _FakeModel:
    """Minimal mock that mimics pipeline predict_proba."""

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else 1
        # Return a fixed probability of 0.72 for the fraud class
        return np.array([[0.28, 0.72]] * n)


class _FakeFeatureBuilder:
    """Minimal mock that mimics FeatureBuilder.transform."""

    def transform(self, df):
        n = len(df)
        X = np.zeros((n, 5))
        return X, None, None


def _mock_load():
    """Return fake model and feature builder."""
    return _FakeModel(), _FakeFeatureBuilder()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestScoreEndpoint:
    """Tests for POST /score."""

    @patch("sentinel.serving.api._load", side_effect=_mock_load)
    def test_score_returns_probability_and_decision(self, mock):
        resp = client.post("/score", json=VALID_TXN)
        assert resp.status_code == 200
        body = resp.json()
        assert "probability" in body
        assert "decision" in body
        assert "threshold" in body
        assert isinstance(body["probability"], float)
        assert 0.0 <= body["probability"] <= 1.0
        assert body["decision"] in ("alert", "accept")

    @patch("sentinel.serving.api._load", side_effect=_mock_load)
    def test_score_respects_custom_threshold(self, mock):
        # With threshold=0.9, prob 0.72 should be "accept"
        resp = client.post("/score?threshold=0.9", json=VALID_TXN)
        assert resp.status_code == 200
        assert resp.json()["decision"] == "accept"

    @patch("sentinel.serving.api._load", side_effect=_mock_load)
    def test_score_alerts_above_threshold(self, mock):
        # With threshold=0.5, prob 0.72 should be "alert"
        resp = client.post("/score?threshold=0.5", json=VALID_TXN)
        assert resp.status_code == 200
        assert resp.json()["decision"] == "alert"

    def test_score_missing_field_returns_422(self):
        incomplete = {k: v for k, v in VALID_TXN.items() if k != "amt"}
        resp = client.post("/score", json=incomplete)
        assert resp.status_code == 422

    @patch("sentinel.serving.api._load", side_effect=FileNotFoundError("no model"))
    def test_score_no_model_returns_503(self, mock):
        resp = client.post("/score", json=VALID_TXN)
        assert resp.status_code == 503
