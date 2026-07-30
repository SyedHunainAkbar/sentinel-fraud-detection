"""Tests for sentinel.ingest: schema validation and temporal split."""
from __future__ import annotations

import pandas as pd
import pytest

from sentinel import config
from sentinel.ingest import load_transactions, temporal_split


class TestLoadTransactions:
    """Tests for load_transactions schema validation."""

    def test_loads_sample_successfully(self):
        """Sample CSV loads without error."""
        df = load_transactions(config.SAMPLE_CSV)
        assert len(df) > 0
        for col in config.REQUIRED_COLUMNS:
            assert col in df.columns

    def test_rejects_missing_column(self, tmp_path):
        """Raises ValueError when a required column is absent."""
        # Create a CSV missing the 'amt' column
        df = pd.DataFrame({
            col: [0] for col in config.REQUIRED_COLUMNS if col != "amt"
        })
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="Missing required columns.*amt"):
            load_transactions(path)

    def test_rejects_multiple_missing_columns(self, tmp_path):
        """Raises ValueError listing all missing columns."""
        df = pd.DataFrame({"trans_date_trans_time": ["2019-01-01"], "cc_num": [123]})
        path = tmp_path / "sparse.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_transactions(path)

    def test_deduplicates_by_trans_num(self, tmp_path):
        """Duplicate trans_num rows are removed."""
        df = load_transactions(config.SAMPLE_CSV)
        # Duplicate first row
        doubled = pd.concat([df.head(5), df.head(5)], ignore_index=True)
        path = tmp_path / "dupes.csv"
        doubled.to_csv(path, index=False)

        result = load_transactions(path)
        assert len(result) == 5


class TestTemporalSplit:
    """Tests for temporal_split time ordering."""

    @pytest.fixture()
    def sample_df(self):
        return load_transactions(config.SAMPLE_CSV)

    def test_split_preserves_time_order(self, sample_df):
        """Train max time <= test min time (no temporal overlap)."""
        train, test = temporal_split(sample_df, train_frac=0.7)
        assert train["unix_time"].max() <= test["unix_time"].min()

    def test_split_sizes_sum_to_original(self, sample_df):
        """No rows lost in the split."""
        train, test = temporal_split(sample_df, train_frac=0.7)
        assert len(train) + len(test) == len(sample_df)

    def test_train_is_sorted(self, sample_df):
        """Train partition is time-sorted."""
        train, _ = temporal_split(sample_df, train_frac=0.7)
        assert (train["unix_time"].diff().dropna() >= 0).all()

    def test_test_is_sorted(self, sample_df):
        """Test partition is time-sorted."""
        _, test = temporal_split(sample_df, train_frac=0.7)
        assert (test["unix_time"].diff().dropna() >= 0).all()
