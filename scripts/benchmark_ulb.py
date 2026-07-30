"""Benchmark Sentinel's XGBoost pipeline on the ULB PCA creditcard dataset.

The ULB dataset (Kaggle: "Credit Card Fraud Detection") uses PCA-anonymized features
(V1-V28) plus Time and Amount. This provides external credibility — showing the same
cost-sensitive approach works on an independent, widely-cited fraud dataset.

Usage:
    python scripts/benchmark_ulb.py                           # default path
    ULB_DATA=path/to/creditcard.csv python scripts/benchmark_ulb.py

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Output: reports/benchmark_ulb.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Add src to path so we can use sentinel evaluation utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel import config  # noqa: E402
from sentinel.evaluation import (  # noqa: E402
    expected_dollar_loss,
    ks_statistic,
    optimize_threshold,
    precision_recall_at_budget,
)


def load_ulb(path: Path) -> pd.DataFrame:
    """Load and validate the ULB creditcard.csv dataset.

    Parameters
    ----------
    path : Path
        Path to creditcard.csv.

    Returns
    -------
    DataFrame
        Validated dataset with expected columns.
    """
    df = pd.read_csv(path)
    required = ["Time", "Amount", "Class"]
    pca_cols = [f"V{i}" for i in range(1, 29)]
    missing = [c for c in required + pca_cols if c not in df.columns]
    if missing:
        raise ValueError(f"ULB dataset missing columns: {missing}")
    return df


def main() -> None:
    """Run the ULB benchmark end-to-end."""
    # Resolve data path
    env_path = os.environ.get("ULB_DATA")
    if env_path:
        data_path = Path(env_path)
    else:
        data_path = config.DATA_DIR / "raw" / "creditcard.csv"

    if not data_path.exists():
        print(
            f"ERROR: ULB dataset not found at {data_path}.\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place as data/raw/creditcard.csv or set ULB_DATA env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading ULB dataset: {data_path}")
    df = load_ulb(data_path)
    print(f"  {len(df):,} transactions | {df['Class'].sum():,} frauds "
          f"({df['Class'].mean():.3%} fraud rate)")

    # Feature columns: V1-V28 + scaled Amount + Time
    pca_cols = [f"V{i}" for i in range(1, 29)]
    feature_cols = pca_cols + ["Amount_scaled", "Time_scaled"]

    # Scale Amount and Time (PCA features are already scaled)
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])

    X = df[feature_cols].values
    y = df["Class"].values
    amount = df["Amount"].values

    # Temporal split: use first 70% as train (data is ordered by Time)
    cut = int(len(df) * 0.7)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]
    amount_test = amount[cut:]

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train fraud rate: {y_train.mean():.4%} | Test fraud rate: {y_test.mean():.4%}")

    # Train XGBoost with scale_pos_weight (same approach as Sentinel)
    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    # Score
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate with the same cost-sensitive framework
    print("Evaluating...")
    t_star, cost_curve = optimize_threshold(y_test, y_prob, amount_test)
    loss_at_optimal = expected_dollar_loss(y_test, y_prob, amount_test, t_star)
    naive_loss = expected_dollar_loss(y_test, y_prob, amount_test, threshold=1.1)
    dollars_saved = naive_loss - loss_at_optimal

    pr_auc = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    ks = ks_statistic(y_test, y_prob)
    brier = float(brier_score_loss(y_test, y_prob))
    prec_k, rec_b = precision_recall_at_budget(y_test, y_prob)

    # Fraud caught at optimal threshold
    alerts = y_prob >= t_star
    fraud_caught = int((alerts & (y_test == 1)).sum())
    total_fraud = int(y_test.sum())

    report = {
        "dataset": "ULB Credit Card Fraud (PCA anonymized)",
        "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "n_transactions": len(df),
        "n_test": len(X_test),
        "fraud_rate_test": float(y_test.mean()),
        "model": "XGBoost (same architecture as Sentinel)",
        "metrics": {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "ks": round(ks, 4),
            "brier": round(brier, 6),
            "precision_at_k": round(prec_k, 4),
            "recall_at_budget": round(rec_b, 4),
            "optimal_threshold": round(t_star, 4),
            "expected_loss": round(loss_at_optimal, 2),
            "naive_loss": round(naive_loss, 2),
            "dollars_saved": round(dollars_saved, 2),
            "fraud_caught": f"{fraud_caught}/{total_fraud}",
        },
        "review_cost": config.REVIEW_COST,
        "notes": "Temporal split (first 70% train, last 30% test by time). "
                 "Same XGBoost hyperparameters as Sentinel Sparkov pipeline. "
                 "Demonstrates approach generalizes to an independent, widely-cited dataset.",
    }

    out = config.REPORTS_DIR / "benchmark_ulb.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print("ULB Benchmark Results")
    print(f"{'='*60}")
    print(f"  PR-AUC:          {pr_auc:.4f}")
    print(f"  ROC-AUC:         {roc_auc:.4f}")
    print(f"  KS:              {ks:.4f}")
    print(f"  Brier:           {brier:.6f}")
    print(f"  Precision@k:     {prec_k:.4f}")
    print(f"  Recall@budget:   {rec_b:.4f}")
    print(f"  Threshold:       {t_star:.4f}")
    print(f"  Expected loss:   ${loss_at_optimal:,.2f}")
    print(f"  Dollars saved:   ${dollars_saved:,.2f}")
    print(f"  Fraud caught:    {fraud_caught}/{total_fraud} "
          f"({fraud_caught/max(total_fraud,1):.1%})")
    print(f"{'='*60}")
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
