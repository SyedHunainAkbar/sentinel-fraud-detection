"""Single source of truth for paths, seed, and cost parameters."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SAMPLE_CSV = DATA_DIR / "sample" / "transactions_sample.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Determinism
RANDOM_SEED = 42

# Cost model (see .kiro/steering/risk.md)
REVIEW_COST = 3.00  # fixed dollar cost per alert (analyst time + friction)

# Alert budget: fraction of transactions that can be reviewed per period
ALERT_BUDGET_FRAC = 0.005  # top 0.5% by score

# Temporal split
TRAIN_FRAC = 0.7

# Expected raw schema for the Sparkov dataset
REQUIRED_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "city", "state", "zip", "lat", "long", "city_pop", "job", "dob",
    "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud",
]

CATEGORICAL_FEATURES = ["category", "gender", "state"]
NUMERIC_FEATURES = [
    "distance_km", "hour", "day_of_week", "is_night", "age",
    "log_amt", "amt_z_by_cat", "velocity_24h", "city_pop_log",
    "home_deviation_km",
]

# Drift monitoring
PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25
DEGRADATION_RATIO = 0.80  # flag if window PR-AUC < 80% of baseline
DRIFT_N_WINDOWS = 6


def data_path() -> Path:
    """Full dataset path if provided via SENTINEL_DATA, else the committed sample."""
    env = os.environ.get("SENTINEL_DATA")
    return Path(env) if env else SAMPLE_CSV


for _d in (MODELS_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)
