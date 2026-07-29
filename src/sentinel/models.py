"""Model training with class-imbalance handling.

Three models: a calibrated logistic-regression baseline, an XGBoost model with
``scale_pos_weight``, and an isolation-forest anomaly baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )


def train_baseline(X: pd.DataFrame, y: pd.Series):
    """Calibrated, class-balanced logistic regression."""
    base = Pipeline([
        ("pre", _preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=config.RANDOM_SEED)),
    ])
    n_splits = min(3, int(y.sum())) if y.sum() >= 2 else 2
    model = CalibratedClassifierCV(base, method="sigmoid", cv=n_splits)
    model.fit(X, y)
    return model


def train_xgboost(X: pd.DataFrame, y: pd.Series):
    """XGBoost with scale_pos_weight for imbalance."""
    from xgboost import XGBClassifier

    pos = max(int(y.sum()), 1)
    neg = len(y) - pos
    model = Pipeline([
        ("pre", _preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=neg / pos, eval_metric="aucpr",
            tree_method="hist", random_state=config.RANDOM_SEED,
        )),
    ])
    model.fit(X, y)
    return model


def train_isolation_forest(X: pd.DataFrame):
    """Unsupervised anomaly baseline. Returns (model, score_fn)."""
    model = Pipeline([
        ("pre", _preprocessor()),
        ("clf", IsolationForest(n_estimators=200, contamination="auto",
                                random_state=config.RANDOM_SEED)),
    ])
    model.fit(X)

    def score(frame: pd.DataFrame) -> np.ndarray:
        # higher = more anomalous, mapped to [0, 1]
        raw = -model.named_steps["clf"].decision_function(
            model.named_steps["pre"].transform(frame)
        )
        return (raw - raw.min()) / (np.ptp(raw) or 1.0)

    return model, score
