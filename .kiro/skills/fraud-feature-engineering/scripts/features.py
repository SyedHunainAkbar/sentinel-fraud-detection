"""Reusable, leakage-free feature helpers for card-transaction fraud models.

These are standalone reference implementations with doctests. The production versions
live in ``src/sentinel/features.py``; keep them consistent.
"""
from __future__ import annotations

import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (vectorized).

    >>> round(float(haversine_km(40.7128, -74.0060, 34.0522, -118.2437)), 0)
    3936.0
    """
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def causal_velocity_24h(cc_num, unix_time):
    """Trailing-24h prior-transaction count per card (causal; excludes current row).

    Parameters are equal-length arrays. Returns an int array of the same length.

    >>> list(map(int, causal_velocity_24h([1, 1, 1], [0, 100, 90000])))
    [0, 1, 0]
    """
    cc = np.asarray(cc_num)
    t = np.asarray(unix_time, dtype=float)
    order = np.lexsort((t, cc))
    out = np.zeros(len(t), dtype=int)
    for card in np.unique(cc):
        idx = order[cc[order] == card]
        times = t[idx]
        # for each position, count earlier times within 86400s
        left = np.searchsorted(times, times - 86400.0, side="left")
        pos = np.arange(len(times))
        out[idx] = pos - left
    return out


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
