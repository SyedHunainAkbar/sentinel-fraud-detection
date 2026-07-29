"""Load artifacts, compute the full report, and write reports/evaluation.json."""
from __future__ import annotations

import json

import joblib

from . import config, evaluation


def main() -> None:
    artifacts = joblib.load(config.MODELS_DIR / "artifacts.joblib")
    y_test = artifacts["y_test"]
    amount = artifacts["amount_test"]

    results = {name: evaluation.evaluate(y_test, prob, amount)
               for name, prob in artifacts["preds"].items()}

    # Best model = lowest expected dollar loss at its optimal threshold
    best = min(results, key=lambda k: results[k]["expected_loss"])
    naive_loss = results[best]["naive_loss"]
    dollars_saved = naive_loss - results[best]["expected_loss"]

    report = {
        "version": "0.1.0",
        "dataset": "Sparkov simulated transactions (Kaggle)",
        "time_range": artifacts.get("time_range"),
        "fraud_rate": artifacts.get("fraud_rate"),
        "review_cost": config.REVIEW_COST,
        "alert_budget_frac": config.ALERT_BUDGET_FRAC,
        "best_model": best,
        "naive_loss": naive_loss,
        "dollars_saved": dollars_saved,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "cost_curve"}
                   for k, v in results.items()},
        "cost_curve": results[best]["cost_curve"],
    }
    out = config.REPORTS_DIR / "evaluation.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"Best model: {best} | expected loss ${results[best]['expected_loss']:,.2f} "
          f"| saved ${dollars_saved:,.2f} vs naive. Wrote {out}.")
    for name, m in results.items():
        print(f"  {name:16s} PR-AUC={m['pr_auc']}  KS={m['ks']:.3f}  "
              f"loss=${m['expected_loss']:,.2f}")


if __name__ == "__main__":
    main()
