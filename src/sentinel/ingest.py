"""Load, validate, and temporally split transaction data."""
from __future__ import annotations

import pandas as pd

from . import config


def load_transactions(path=None) -> pd.DataFrame:
    """Load transactions, validate schema, coerce dtypes, and drop duplicates.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    path = path or config.data_path()
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["is_fraud"] = df["is_fraud"].astype(int)
    df = df.drop_duplicates(subset="trans_num").reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame, train_frac: float | None = None):
    """Split by time: earliest ``train_frac`` to train, remainder to test."""
    train_frac = config.TRAIN_FRAC if train_frac is None else train_frac
    df = df.sort_values("unix_time").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
