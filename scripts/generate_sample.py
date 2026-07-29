"""Generate a tiny synthetic sample with the Sparkov schema for tests/CI.

This is NOT the real dataset — it exists only so the pipeline, tests, and CI run without
the ~350 MB Kaggle download. Fetch the real data with `make data`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N = 1200
FRAUD_RATE = 0.02

rng = np.random.default_rng(SEED)
categories = ["grocery_pos", "gas_transport", "misc_net", "shopping_net",
              "entertainment", "food_dining", "travel"]
states = ["CA", "NY", "TX", "FL", "WA", "IL"]

start = pd.Timestamp("2020-01-01")
times = start + pd.to_timedelta(np.sort(rng.uniform(0, 60, N)), unit="D")
is_fraud = (rng.random(N) < FRAUD_RATE).astype(int)

# fraud transactions skew toward larger amounts, odd hours, far distances
amt = np.where(is_fraud == 1,
               rng.gamma(3.0, 120.0, N),
               rng.gamma(2.0, 30.0, N)).round(2)
lat = rng.uniform(33, 47, N)
lon = rng.uniform(-122, -73, N)
merch_lat = lat + np.where(is_fraud == 1, rng.normal(0, 3, N), rng.normal(0, 0.3, N))
merch_lon = lon + np.where(is_fraud == 1, rng.normal(0, 3, N), rng.normal(0, 0.3, N))
cc_num = rng.integers(10**15, 10**16, 40)  # 40 cards

df = pd.DataFrame({
    "trans_date_trans_time": times.strftime("%Y-%m-%d %H:%M:%S"),
    "cc_num": rng.choice(cc_num, N),
    "merchant": ["fraud_Kirlin" if f else "legit_Store" for f in is_fraud],
    "category": rng.choice(categories, N),
    "amt": amt,
    "first": "Sample", "last": "User",
    "gender": rng.choice(["M", "F"], N),
    "street": "1 Main St",
    "city": "Springfield",
    "state": rng.choice(states, N),
    "zip": rng.integers(10000, 99999, N),
    "lat": lat.round(4), "long": lon.round(4),
    "city_pop": rng.integers(1000, 3_000_000, N),
    "job": "Analyst",
    "dob": pd.to_datetime("1970-01-01") + pd.to_timedelta(rng.integers(0, 18000, N), unit="D"),
    "trans_num": [f"t{i:06d}" for i in range(N)],
    "unix_time": (times.view("int64") // 10**9),
    "merch_lat": merch_lat.round(4), "merch_long": merch_lon.round(4),
    "is_fraud": is_fraud,
})

out = "data/sample/transactions_sample.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} ({len(df)} rows, {is_fraud.sum()} fraud)")
