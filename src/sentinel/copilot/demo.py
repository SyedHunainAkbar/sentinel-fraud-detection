"""Demo: run the investigation agent on the riskiest sample transactions."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from .. import config, ingest
from ..features import FeatureBuilder
from .investigate import investigate


def _make_score_fn():
    """Use the trained model if present; else a transparent heuristic."""
    try:
        model = joblib.load(config.MODELS_DIR / "xgboost.joblib")
        fb = joblib.load(config.MODELS_DIR / "feature_builder.joblib")

        def score(txn: dict) -> float:
            X, _, _ = fb.transform(pd.DataFrame([txn]))
            return float(model.predict_proba(X)[:, 1][0])
        return score
    except FileNotFoundError:
        def score(txn: dict) -> float:
            return min(0.99, txn.get("amt", 0) / 1000.0)
        return score


def main(n: int = 5) -> None:
    df = ingest.load_transactions()
    fb = FeatureBuilder().fit(df)
    X, _, _ = fb.transform(df)
    df = df.assign(_prob_proxy=X["log_amt"] + X["distance_km"] / 100.0)
    top = df.sort_values("_prob_proxy", ascending=False).head(n)

    score_fn = _make_score_fn()
    results = []
    for _, row in top.iterrows():
        txn = row.to_dict()
        history = df[df["cc_num"] == row["cc_num"]]
        inv = investigate(txn, score_fn, history_df=history)
        results.append(inv.to_dict())
        print(f"{inv.transaction_id}: {inv.recommendation} "
              f"(p={inv.probability:.2f}, cites={inv.citations})")

    out = config.REPORTS_DIR / "investigations.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
