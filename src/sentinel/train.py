"""Orchestrate: ingest -> features -> train models -> persist artifacts."""
from __future__ import annotations

import joblib

from . import config, ingest, models
from .features import FeatureBuilder


def main() -> None:
    df = ingest.load_transactions()
    train_df, test_df = ingest.temporal_split(df)

    fb = FeatureBuilder()
    X_train, y_train, _ = fb.fit_transform(train_df)
    X_test, y_test, amt_test = fb.transform(test_df)

    baseline = models.train_baseline(X_train, y_train)
    xgb = models.train_xgboost(X_train, y_train)
    iso, iso_score = models.train_isolation_forest(X_train)

    preds = {
        "baseline": baseline.predict_proba(X_test)[:, 1],
        "xgboost": xgb.predict_proba(X_test)[:, 1],
        "isolation_forest": iso_score(X_test),
    }

    # Persist numeric training features as reference for drift monitoring
    X_train_numeric = X_train[config.NUMERIC_FEATURES].to_numpy()

    artifacts = {
        "feature_builder": fb,
        "models": {"baseline": baseline, "xgboost": xgb, "isolation_forest": iso},
        "y_test": y_test.to_numpy(),
        "amount_test": amt_test.to_numpy(),
        "time_test": test_df["unix_time"].to_numpy(),
        "preds": preds,
        "X_train_numeric": X_train_numeric,
        "X_test_numeric": X_test[config.NUMERIC_FEATURES].to_numpy(),
        "time_range": (
            f"{df['trans_date_trans_time'].min()} .. {df['trans_date_trans_time'].max()}"
        ),
        "fraud_rate": float(df["is_fraud"].mean()),
    }
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(artifacts, config.MODELS_DIR / "artifacts.joblib")
    joblib.dump(xgb, config.MODELS_DIR / "xgboost.joblib")
    joblib.dump(fb, config.MODELS_DIR / "feature_builder.joblib")
    print(f"Trained on {len(train_df)} / tested on {len(test_df)} rows. "
          f"Fraud rate: {artifacts['fraud_rate']:.3%}. Artifacts saved to {config.MODELS_DIR}.")


if __name__ == "__main__":
    main()
