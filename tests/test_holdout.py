"""Tests for external holdout evaluation module."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sentinel.holdout import _holdout_path, main


class TestHoldoutPath:
    def test_defaults_to_data_raw_fraudtest(self, monkeypatch):
        monkeypatch.delenv("SENTINEL_HOLDOUT", raising=False)
        path = _holdout_path()
        assert path.name == "fraudTest.csv"
        assert "raw" in str(path)

    def test_respects_env_override(self, monkeypatch):
        monkeypatch.setenv("SENTINEL_HOLDOUT", "/tmp/custom.csv")
        path = _holdout_path()
        assert path == Path("/tmp/custom.csv")


class TestHoldoutMain:
    def test_raises_if_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SENTINEL_HOLDOUT", str(tmp_path / "nope.csv"))
        with pytest.raises(FileNotFoundError, match="Hold-out file not found"):
            main()

    def test_raises_if_model_missing(self, monkeypatch, tmp_path):
        # Create a fake CSV so file-not-found is for model
        from sentinel import config
        fake_csv = tmp_path / "fake.csv"
        # Minimal valid CSV
        cols = config.REQUIRED_COLUMNS
        df = pd.DataFrame({c: [0] for c in cols})
        df["trans_date_trans_time"] = "2020-01-01 00:00:00"
        df["dob"] = "1990-01-01"
        df.to_csv(fake_csv, index=False)
        monkeypatch.setenv("SENTINEL_HOLDOUT", str(fake_csv))
        # Point models to empty dir
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "no_models")
        (tmp_path / "no_models").mkdir()
        with pytest.raises(FileNotFoundError, match="Model artifacts not found"):
            main()
