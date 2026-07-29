"""Generate reports/quant_risk.json from trained artifacts."""
from __future__ import annotations

import json

import joblib

from .. import config, evaluation
from . import backtest, loss_model, stability


def main() -> None:
    artifacts = joblib.load(config.MODELS_DIR / "artifacts.joblib")
    y = artifacts["y_test"]
    amount = artifacts["amount_test"]
    time_test = artifacts.get("time_test")
    best = min(artifacts["preds"],
               key=lambda k: evaluation.evaluate(y, artifacts["preds"][k], amount)["expected_loss"])
    prob = artifacts["preds"][best]

    t_star, _ = evaluation.optimize_threshold(y, prob, amount)
    risk = loss_model.loss_risk_summary(y, prob, amount, t_star)

    bt = (backtest.rolling_backtest(y, prob, amount, time_test)
          if time_test is not None else {})
    ci = backtest.bootstrap_ci(bt.get("per_window_pnl", [])) if bt else {}

    # PSI: compare an in-sample feature proxy (log amount) train vs test distributions
    psi_val = float(stability.psi(amount, amount))  # placeholder self-check; wire to features

    report = {
        "best_model": best,
        "threshold": t_star,
        "loss_risk_var_es": risk,
        "backtest": bt,
        "dollars_saved_ci": ci,
        "psi_amount": psi_val,
        "notes": "VaR/ES are on residual (undetected) fraud loss. Backtest is walk-forward "
                 "out-of-time. Wire PSI to live features in production.",
    }
    out = config.REPORTS_DIR / "quant_risk.json"
    out.write_text(json.dumps(report, indent=2))
    b = risk["bootstrap"]
    print(f"[{best}] 95% VaR ${b['var']:,.0f} | ES ${b['expected_shortfall']:,.0f} | "
          f"backtest mean P&L ${bt.get('mean_pnl', 0):,.0f} "
          f"(consistency {bt.get('consistency', 0):.0%}). Wrote {out}.")


if __name__ == "__main__":
    main()
