"""Evaluate the trained XGBoost model on a TRUE external hold-out set.

The Sparkov dataset ships as two files: fraudTrain.csv (used for temporal train/test
split during development) and fraudTest.csv (never seen during training or threshold
tuning). This script loads fraudTest.csv, applies the saved FeatureBuilder, scores with
the persisted XGBoost model, and writes a full metric report to
reports/holdout_eval.json.

Usage:
    python -m sentinel.holdout                        # defaults to data/raw/fraudTest.csv
    SENTINEL_HOLDOUT=path/to/file.csv python -m sentinel.holdout
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from . import config, evaluation, ingest


def _holdout_path() -> Path:
    """Resolve the hold-out CSV path from env or convention."""
    env = os.environ.get("SENTINEL_HOLDOUT")
    if env:
        return Path(env)
    return config.DATA_DIR / "raw" / "fraudTest.csv"


def main() -> None:
    """Run external hold-out evaluation and persist results."""
    holdout_csv = _holdout_path()
    if not holdout_csv.exists():
        raise FileNotFoundError(
            f"Hold-out file not found: {holdout_csv}. "
            "Download the Sparkov dataset with `make data` or set SENTINEL_HOLDOUT."
        )

    # Load saved artifacts (trained on fraudTrain.csv temporal split)
    model_path = config.MODELS_DIR / "xgboost.joblib"
    fb_path = config.MODELS_DIR / "feature_builder.joblib"
    if not model_path.exists() or not fb_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `make train` first."
        )

    model = joblib.load(model_path)
    fb = joblib.load(fb_path)

    # Load and transform external hold-out
    print(f"Loading external hold-out: {holdout_csv}")
    df = ingest.load_transactions(holdout_csv)
    X, y, amount = fb.transform(df)

    # Score
    y_prob = model.predict_proba(X)[:, 1]

    # Full metric evaluation (same as in-sample evaluate.py)
    metrics = evaluation.evaluate(y.to_numpy(), y_prob, amount.to_numpy())

    # Dollars saved vs naive (no-model) baseline
    naive_loss = metrics["naive_loss"]
    dollars_saved = naive_loss - metrics["expected_loss"]

    # Also load in-sample results for comparison context
    eval_path = config.REPORTS_DIR / "evaluation.json"
    in_sample = None
    if eval_path.exists():
        in_sample_report = json.loads(eval_path.read_text())
        best_model = in_sample_report.get("best_model", "xgboost")
        in_sample = in_sample_report.get("models", {}).get(best_model)

    report = {
        "version": "0.1.0",
        "dataset": "Sparkov fraudTest.csv (external hold-out, never seen during training)",
        "holdout_file": str(holdout_csv.name),
        "n_transactions": len(df),
        "n_frauds": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "review_cost": config.REVIEW_COST,
        "alert_budget_frac": config.ALERT_BUDGET_FRAC,
        "model": "xgboost",
        "metrics": {k: v for k, v in metrics.items() if k != "cost_curve"},
        "naive_loss": naive_loss,
        "dollars_saved": dollars_saved,
        "cost_curve": metrics["cost_curve"],
    }

    # Add comparison if in-sample metrics are available
    if in_sample:
        report["comparison_vs_temporal_split"] = {
            "note": "Temporal split = train/test from fraudTrain.csv; "
                    "hold-out = entirely separate fraudTest.csv",
            "temporal_split": {
                "pr_auc": in_sample.get("pr_auc"),
                "roc_auc": in_sample.get("roc_auc"),
                "ks": in_sample.get("ks"),
                "brier": in_sample.get("brier"),
                "expected_loss": in_sample.get("expected_loss"),
                "optimal_threshold": in_sample.get("optimal_threshold"),
                "precision_at_k": in_sample.get("precision_at_k"),
                "recall_at_budget": in_sample.get("recall_at_budget"),
            },
            "external_holdout": {
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
                "ks": metrics.get("ks"),
                "brier": metrics.get("brier"),
                "expected_loss": metrics.get("expected_loss"),
                "optimal_threshold": metrics.get("optimal_threshold"),
                "precision_at_k": metrics.get("precision_at_k"),
                "recall_at_budget": metrics.get("recall_at_budget"),
            },
        }

    out = config.REPORTS_DIR / "holdout_eval.json"
    out.write_text(json.dumps(report, indent=2))

    print("External hold-out evaluation complete.")
    print(f"  Transactions: {len(df):,} | Frauds: {int(y.sum()):,} "
          f"({y.mean():.3%})")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} "
          f"| KS: {metrics['ks']:.3f}")
    print(f"  Optimal threshold: {metrics['optimal_threshold']:.4f}")
    print(f"  Expected loss: ${metrics['expected_loss']:,.2f}")
    print(f"  Dollars saved vs naive: ${dollars_saved:,.2f}")
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
