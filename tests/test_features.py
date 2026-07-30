"""Tests for feature engineering (correctness + leakage)."""
import numpy as np
import pandas as pd

from sentinel.features import (
    FeatureBuilder,
    causal_home_deviation,
    causal_velocity_24h,
    haversine_km,
)


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


def test_no_leakage_transform_uses_fitted_stats_only():
    """FeatureBuilder.transform must use stats from fit(), not from the new data.

    If we fit on a train set with category 'a' having mean=10, then transform a
    test set where category 'a' has mean=1000, the z-score must be computed using
    the *train* mean (10), not the test mean (1000).
    """
    train = _pad(pd.DataFrame({"category": ["a", "a", "a"], "amt": [10.0, 10.0, 10.0]}))
    fb = FeatureBuilder().fit(train)

    # Test data has very different amounts — if leaking, z-score would be ~0
    test = _pad(pd.DataFrame({"category": ["a"], "amt": [1000.0]}))
    X, _, _ = fb.transform(test)

    # z-score should be large because (1000 - 10) / std is huge, not near 0
    z = X["amt_z_by_cat"].iloc[0]
    assert abs(z) > 5.0, f"Expected large z-score (no leakage), got {z}"


def test_no_future_data_in_velocity():
    """Causal velocity must not count future transactions."""
    # 5 transactions for same card at 1-hour intervals
    times = [0, 3600, 7200, 10800, 14400]
    vel = causal_velocity_24h([1, 1, 1, 1, 1], times)

    # First txn has no prior → 0; each subsequent counts only past within 24h
    assert vel[0] == 0
    # Each subsequent can see all prior (all within 24h = 86400s)
    for i in range(1, len(vel)):
        assert vel[i] == i, f"vel[{i}] should be {i}, got {vel[i]}"


def test_haversine_zero_distance():
    """Same point should return 0 km."""
    d = haversine_km(51.5074, -0.1278, 51.5074, -0.1278)
    assert abs(float(d)) < 1e-6


def test_haversine_antipodal():
    """Antipodal points should be ~20000 km (half circumference)."""
    # North pole to south pole
    d = haversine_km(90.0, 0.0, -90.0, 0.0)
    assert 20000 < float(d) < 20100


def test_haversine_vectorized():
    """Haversine accepts arrays and returns correct shape."""
    lats1 = np.array([0.0, 51.5074])
    lons1 = np.array([0.0, -0.1278])
    lats2 = np.array([0.0, 48.8566])
    lons2 = np.array([1.0, 2.3522])
    result = haversine_km(lats1, lons1, lats2, lons2)
    assert len(result) == 2
    # First: equator, 1 degree longitude ~ 111 km
    assert 110 < float(result[0]) < 112
    # Second: London to Paris ~ 340 km
    assert 330 < float(result[1]) < 350


# --- Tests for causal_home_deviation ---


def test_home_deviation_3_transaction_card():
    """Exact distances on a 3-transaction card.

    Card 1 at times [1, 2, 3]:
    - Txn 0: first txn → fallback to haversine(home, merchant0)
    - Txn 1: mean of prior merchants = merchant0 → haversine(merchant0, merchant1)
    - Txn 2: mean of prior merchants = midpoint(merchant0, merchant1)
             → haversine(midpoint, merchant2)
    """
    cc_num = [1, 1, 1]
    unix_time = [100, 200, 300]
    # Home at (0, 0); merchants at (0,1), (0,2), (0,3) degrees longitude on equator
    home_lat = [0.0, 0.0, 0.0]
    home_long = [0.0, 0.0, 0.0]
    merch_lat = [0.0, 0.0, 0.0]
    merch_long = [1.0, 2.0, 3.0]

    result = causal_home_deviation(cc_num, unix_time, merch_lat, merch_long,
                                   home_lat, home_long)

    # Txn 0: haversine((0,0), (0,1)) ~ 111 km
    assert 110 < result[0] < 112

    # Txn 1: prior mean = (0,1); haversine((0,1), (0,2)) ~ 111 km
    assert 110 < result[1] < 112

    # Txn 2: prior mean = (0, 1.5); haversine((0,1.5), (0,3)) ~ 167 km
    assert 165 < result[2] < 168


def test_home_deviation_causality():
    """Adding a later transaction must not change earlier values."""
    cc_num_short = [1, 1, 1]
    unix_time_short = [100, 200, 300]
    merch_lat = [0.0, 0.0, 0.0]
    merch_long = [1.0, 2.0, 3.0]
    home_lat = [0.0, 0.0, 0.0]
    home_long = [0.0, 0.0, 0.0]

    result_short = causal_home_deviation(cc_num_short, unix_time_short,
                                         merch_lat, merch_long,
                                         home_lat, home_long)

    # Now add a 4th transaction
    cc_num_long = [1, 1, 1, 1]
    unix_time_long = [100, 200, 300, 400]
    merch_lat_long = [0.0, 0.0, 0.0, 0.0]
    merch_long_long = [1.0, 2.0, 3.0, 10.0]
    home_lat_long = [0.0, 0.0, 0.0, 0.0]
    home_long_long = [0.0, 0.0, 0.0, 0.0]

    result_long = causal_home_deviation(cc_num_long, unix_time_long,
                                        merch_lat_long, merch_long_long,
                                        home_lat_long, home_long_long)

    # First 3 values must be identical — causality preserved
    np.testing.assert_array_almost_equal(result_short, result_long[:3], decimal=6)
