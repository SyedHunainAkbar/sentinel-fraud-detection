"""Leakage-free feature engineering for card-transaction fraud.

The :class:`FeatureBuilder` fits per-category statistics on the training split only and
applies them to later data. Rolling velocity is causal (prior transactions only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (vectorized)."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def causal_velocity_24h(cc_num, unix_time):
    """Count of prior same-card transactions within trailing 24h (causal)."""
    cc = np.asarray(cc_num)
    t = np.asarray(unix_time, dtype=float)
    out = np.zeros(len(t), dtype=int)
    order = np.lexsort((t, cc))
    for card in np.unique(cc):
        idx = order[cc[order] == card]
        times = t[idx]
        left = np.searchsorted(times, times - 86400.0, side="left")
        out[idx] = np.arange(len(times)) - left
    return out


class FeatureBuilder:
    """Fit-once/transform-many feature builder with no train->test leakage."""

    def __init__(self) -> None:
        self.cat_stats_: dict[str, tuple[float, float]] = {}
        self.global_amt_stats_: tuple[float, float] = (0.0, 1.0)

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        g = df.groupby("category")["amt"]
        self.cat_stats_ = {
            k: (float(v.mean()), float(v.std(ddof=0) or 1.0)) for k, v in g
        }
        self.global_amt_stats_ = (float(df["amt"].mean()), float(df["amt"].std(ddof=0) or 1.0))
        return self

    def _amt_z(self, row_cat, amt):
        mean, std = self.cat_stats_.get(row_cat, self.global_amt_stats_)
        return (amt - mean) / (std or 1.0)

    def transform(self, df: pd.DataFrame):
        """Return (X, y, amount). ``y``/``amount`` are None if is_fraud absent."""
        out = pd.DataFrame(index=df.index)
        ts = pd.to_datetime(df["trans_date_trans_time"])
        dob = pd.to_datetime(df["dob"])

        out["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
        out["hour"] = ts.dt.hour
        out["day_of_week"] = ts.dt.dayofweek
        out["is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(int)
        out["age"] = ((ts - dob).dt.days / 365.25).round(1)
        out["log_amt"] = np.log1p(df["amt"])
        out["amt_z_by_cat"] = [
            self._amt_z(c, a)
            for c, a in zip(df["category"], df["amt"], strict=False)
        ]
        out["velocity_24h"] = causal_velocity_24h(df["cc_num"].values, df["unix_time"].values)
        out["city_pop_log"] = np.log1p(df["city_pop"])
        for c in config.CATEGORICAL_FEATURES:
            out[c] = df[c].astype("category")

        y = df["is_fraud"].astype(int) if "is_fraud" in df else None
        amount = df["amt"].astype(float) if "amt" in df else None
        return out, y, amount

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)
