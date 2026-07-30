"""Evaluate copilot recommendation agreement with ground-truth labels.

Runs the investigation agent on a held-out set of flagged transactions and
measures how well the recommendation (escalate/clear/request_info) aligns with
the actual is_fraud label.

Metrics:
- Agreement rate: fraction where escalate -> actual fraud, clear -> actual legit
- Precision of escalate recommendations
- Recall of escalate recommendations (vs actual fraud)
- Confusion matrix breakdown

Usage:
    python -m sentinel.copilot.evaluate_copilot
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config, ingest
from .investigate import investigate
from .retriever import get_retriever


def build_holdout_set(
    df: pd.DataFrame,
    n_flagged: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a held-out evaluation set of flagged transactions.

    Selects a mix of true fraud and borderline legit transactions that
    an analyst would review — mimics the decision boundary where the
    copilot matters most.

    Parameters
    ----------
    df : DataFrame
        Full transaction dataset.
    n_flagged : int
        Number of transactions to include in the evaluation set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Held-out transactions with ground-truth labels.
    """
    rng = np.random.default_rng(seed)  # noqa: F841 — seed used by pandas .sample()

    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]

    # Take ~60% fraud, ~40% legit (simulates alerts that are a mix)
    n_fraud = min(int(n_flagged * 0.6), len(fraud))
    n_legit = min(n_flagged - n_fraud, len(legit))

    fraud_sample = fraud.sample(n=n_fraud, random_state=seed)
    # Pick high-amount legit (borderline, analyst-worthy)
    legit_sorted = legit.sort_values("amt", ascending=False)
    legit_sample = legit_sorted.head(n_legit * 3).sample(n=n_legit, random_state=seed)

    holdout = pd.concat([fraud_sample, legit_sample], ignore_index=True)
    return holdout.sample(frac=1, random_state=seed).reset_index(drop=True)


def evaluate_copilot(
    holdout: pd.DataFrame,
    score_fn,
    retriever=None,
) -> dict:
    """Evaluate copilot recommendations against ground-truth labels.

    Parameters
    ----------
    holdout : DataFrame
        Transactions with is_fraud labels.
    score_fn : callable
        Scoring function for the model.
    retriever : optional
        Retriever instance; defaults to TF-IDF.

    Returns
    -------
    dict
        Agreement metrics, confusion breakdown, per-transaction results.
    """
    retriever = retriever or get_retriever()
    results = []

    for _, row in holdout.iterrows():
        txn = row.to_dict()
        actual_fraud = int(txn.get("is_fraud", 0))

        inv = investigate(txn, score_fn, history_df=None, retriever=retriever)

        # Map recommendation to binary decision
        rec_is_fraud = inv.recommendation == "escalate"
        agreed = rec_is_fraud == bool(actual_fraud)

        results.append({
            "trans_num": str(txn.get("trans_num", "")),
            "actual_fraud": actual_fraud,
            "recommendation": inv.recommendation,
            "confidence": inv.confidence,
            "probability": inv.probability,
            "agreed": agreed,
        })

    df_results = pd.DataFrame(results)

    # Compute metrics
    n = len(df_results)
    agreement_rate = float(df_results["agreed"].mean())

    # Precision/Recall of "escalate" recommendation
    escalated = df_results[df_results["recommendation"] == "escalate"]
    actual_fraud_total = df_results["actual_fraud"].sum()

    precision_escalate = (
        float(escalated["actual_fraud"].sum() / len(escalated))
        if len(escalated) > 0 else 0.0
    )
    recall_escalate = (
        float(escalated["actual_fraud"].sum() / actual_fraud_total)
        if actual_fraud_total > 0 else 0.0
    )

    # Confusion breakdown
    esc = df_results["recommendation"] == "escalate"
    fraud_mask = df_results["actual_fraud"] == 1
    tp = int((esc & fraud_mask).sum())
    fp = int((esc & ~fraud_mask).sum())
    fn = int((~esc & fraud_mask).sum())
    tn = int((~esc & ~fraud_mask).sum())

    # F1 for escalate
    f1_escalate = (
        2 * precision_escalate * recall_escalate / (precision_escalate + recall_escalate)
        if (precision_escalate + recall_escalate) > 0 else 0.0
    )

    return {
        "n_evaluated": n,
        "n_fraud": int(actual_fraud_total),
        "n_legit": int(n - actual_fraud_total),
        "agreement_rate": round(agreement_rate, 4),
        "precision_escalate": round(precision_escalate, 4),
        "recall_escalate": round(recall_escalate, 4),
        "f1_escalate": round(f1_escalate, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "recommendation_distribution": {
            "escalate": int((df_results["recommendation"] == "escalate").sum()),
            "clear": int((df_results["recommendation"] == "clear").sum()),
            "request_info": int((df_results["recommendation"] == "request_info").sum()),
        },
        "mean_confidence": round(float(df_results["confidence"].mean()), 4),
        "per_transaction": results,
    }


def main() -> None:
    """Run copilot evaluation and write report."""
    import joblib

    print("Loading data and model...")
    df = ingest.load_transactions()

    # Use trained model if available, else heuristic
    try:
        model = joblib.load(config.MODELS_DIR / "xgboost.joblib")
        fb = joblib.load(config.MODELS_DIR / "feature_builder.joblib")

        def score_fn(txn: dict) -> float:
            X, _, _ = fb.transform(pd.DataFrame([txn]))
            return float(model.predict_proba(X)[:, 1][0])
    except FileNotFoundError:
        def score_fn(txn: dict) -> float:
            return min(0.99, txn.get("amt", 0) / 1000.0)

        print("  (using heuristic scorer — run `make train` for model-based)")

    # Build held-out evaluation set
    _, test_df = ingest.temporal_split(df)
    holdout = build_holdout_set(test_df, n_flagged=min(50, len(test_df)))
    print(f"  Evaluating {len(holdout)} transactions "
          f"({holdout['is_fraud'].sum()} fraud, {(~holdout['is_fraud'].astype(bool)).sum()} legit)")

    # Run evaluation
    metrics = evaluate_copilot(holdout, score_fn)

    # Write report
    report = {
        "evaluation_type": "copilot_agreement",
        "description": "Measures how well copilot recommendations align with ground-truth labels",
        **{k: v for k, v in metrics.items() if k != "per_transaction"},
        "per_transaction_sample": metrics["per_transaction"][:10],  # first 10 for readability
    }

    out = config.REPORTS_DIR / "copilot_evaluation.json"
    out.write_text(json.dumps(report, indent=2))

    print("\nCopilot Evaluation Results:")
    print(f"  Agreement rate:      {metrics['agreement_rate']:.1%}")
    print(f"  Escalate precision:  {metrics['precision_escalate']:.1%}")
    print(f"  Escalate recall:     {metrics['recall_escalate']:.1%}")
    print(f"  Escalate F1:         {metrics['f1_escalate']:.1%}")
    print(f"  Confusion: TP={metrics['confusion']['tp']} FP={metrics['confusion']['fp']} "
          f"FN={metrics['confusion']['fn']} TN={metrics['confusion']['tn']}")
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()
