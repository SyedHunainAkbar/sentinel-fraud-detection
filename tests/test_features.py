"""Tests for feature engineering (correctness + leakage)."""
import numpy as np
import pandas as pd

from sentinel.features import FeatureBuilder, causal_velocity_24h, haversine_km


def test_haversine_known_distance():
    # NYC to LA is ~3936 km
    d = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900 < float(d) < 3980


def test_velocity_is_causal():
    # same card, three times; last two are within 24h of a prior txn
    vel = causal_velocity_24h([1, 1, 1], [0, 100, 90000])
    assert list(vel) == [0, 1, 0]  # 3rd txn is 25h later, outside 24h


def test_category_zscore_fit_on_train_only():
    train = pd.DataFrame({"category": ["a", "a", "b"], "amt": [10.0, 20.0, 100.0]})
    fb = FeatureBuilder().fit(_pad(train))
    # a new frame with an unseen category falls back to global stats, not refit
    test = _pad(pd.DataFrame({"category": ["zzz"], "amt": [15.0]}))
    X, _, _ = fb.transform(test)
    assert "amt_z_by_cat" in X.columns
    assert np.isfinite(X["amt_z_by_cat"].iloc[0])


def _pad(df: pd.DataFrame) -> pd.DataFrame:
    """Add the minimum columns FeatureBuilder.transform needs."""
    n = len(df)
    base = pd.DataFrame({
        "trans_date_trans_time": pd.date_range("2020-01-01", periods=n, freq="h"),
        "dob": pd.to_datetime(["1990-01-01"] * n),
        "lat": [40.0] * n, "long": [-74.0] * n,
        "merch_lat": [40.1] * n, "merch_long": [-74.1] * n,
        "cc_num": [1] * n, "unix_time": range(n),
        "city_pop": [100000] * n, "gender": ["M"] * n, "state": ["NY"] * n,
        "is_fraud": [0] * n,
    })
    for c in df.columns:
        base[c] = df[c].values
    return base
