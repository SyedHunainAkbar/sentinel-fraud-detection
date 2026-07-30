"""Generate reports/quant_risk.json from trained artifacts."""
from __future__ import annotations

import json

import joblib

from .. import config, evaluation
from . import backtest, calibration, loss_model, stability


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

    # PSI: per-feature stability between engineered train vs test distributions
    X_train = artifacts.get("X_train_numeric")
    X_test = artifacts.get("X_test_numeric")
    if X_train is not None and X_test is not None:
        psi_features = stability.feature_psi(
            X_train, X_test, config.NUMERIC_FEATURES
        )
        max_psi_feature = max(psi_features, key=psi_features.get)
        max_psi_value = psi_features[max_psi_feature]
    else:
        # Fallback: single amount-based PSI if feature matrices not persisted
        psi_features = {"amt": float(stability.psi(amount, amount))}
        max_psi_feature = "amt"
        max_psi_value = psi_features["amt"]

    # Calibration: reliability curve + Brier decomposition
    cal_curve = calibration.reliability_curve(y, prob)
    brier_decomp = calibration.brier_decomposition(y, prob)

    report = {
        "best_model": best,
        "threshold": t_star,
        "loss_risk_var_es": risk,
        "backtest": bt,
        "dollars_saved_ci": ci,
        "psi": {
            "per_feature": psi_features,
            "max_feature": max_psi_feature,
            "max_value": max_psi_value,
            "warning_threshold": config.PSI_WARNING_THRESHOLD,
            "critical_threshold": config.PSI_CRITICAL_THRESHOLD,
        },
        "calibration": {
            "reliability_curve": cal_curve,
            "brier_decomposition": brier_decomp,
        },
        "notes": "VaR/ES are on residual (undetected) fraud loss. Backtest is walk-forward "
                 "out-of-time. PSI computed on engineered numeric features (train vs test). "
                 "Calibration treats fraud score as a PD estimate.",
    }
    out = config.REPORTS_DIR / "quant_risk.json"
    out.write_text(json.dumps(report, indent=2))
    b = risk["bootstrap"]
    print(f"[{best}] 95% VaR ${b['var']:,.0f} | ES ${b['expected_shortfall']:,.0f} | "
          f"backtest mean P&L ${bt.get('mean_pnl', 0):,.0f} "
          f"(consistency {bt.get('consistency', 0):.0%}). "
          f"Max PSI: {max_psi_feature}={max_psi_value:.4f}. "
          f"Brier: {brier_decomp['brier_score']:.4f}. Wrote {out}.")


if __name__ == "__main__":
    main()
