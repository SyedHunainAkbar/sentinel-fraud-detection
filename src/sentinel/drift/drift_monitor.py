"""Drift monitor orchestrator — generates reports/drift.json."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from .. import config
from ..evaluation import expected_dollar_loss
from .drift_performance import detect_degradation, rolling_window_metrics
from .drift_psi import compute_feature_psi, compute_score_psi


def _fast_optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    n_grid: int = 200,
) -> float:
    """Quick threshold optimization using a fixed linspace grid (no per-prediction sweep).

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    amount : np.ndarray
        Transaction amounts.
    n_grid : int
        Number of candidate thresholds to evaluate.

    Returns
    -------
    float
        Cost-optimal threshold from the grid.
    """
    grid = np.linspace(0, 1, n_grid)
    costs = [expected_dollar_loss(y_true, y_prob, amount, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def run_drift_monitor() -> dict:
    """Execute the full drift monitoring pipeline.

    Loads trained artifacts, computes feature/score PSI, evaluates rolling-window
    performance, classifies alerts, and writes ``reports/drift.json``.

    Returns
    -------
    dict
        The complete drift report.
    """
    artifact_path = config.MODELS_DIR / "artifacts.joblib"
    if not artifact_path.exists():
        print(
            f"ERROR: {artifact_path} not found. Run `make train` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    artifacts = joblib.load(artifact_path)

    # Validate required keys
    required = ["y_test", "amount_test", "time_test", "preds",
                "X_train_numeric", "X_test_numeric"]
    missing = [k for k in required if k not in artifacts]
    if missing:
        print(
            f"ERROR: artifacts missing keys {missing}. Re-run `make train`.",
            file=sys.stderr,
        )
        sys.exit(1)

    y_test = artifacts["y_test"]
    amount_test = artifacts["amount_test"]
    time_test = artifacts["time_test"]
    X_train_numeric = artifacts["X_train_numeric"]
    X_test_numeric = artifacts["X_test_numeric"]
    feature_names = config.NUMERIC_FEATURES

    # Select best model (fast: evaluate at a coarse grid rather than full sweep)
    best = min(
        artifacts["preds"],
        key=lambda k: _fast_expected_loss(
            y_test, artifacts["preds"][k], amount_test
        ),
    )
    prob = artifacts["preds"][best]

    # Cost-optimal threshold (coarse grid — fast for monitoring)
    t_star = _fast_optimize_threshold(y_test, prob, amount_test)

    # Baseline PR-AUC (full test set)
    baseline_pr_auc = float(average_precision_score(y_test, prob))

    # --- PSI ---
    feature_psi = compute_feature_psi(
        X_train_numeric, X_test_numeric, feature_names
    )

    # Score PSI: compare train-set predictions vs test-set predictions
    # Use the best model's train probabilities if available, else use test as proxy
    ref_scores = prob  # self-comparison placeholder if train preds unavailable
    if "preds_train" in artifacts:
        ref_scores = artifacts["preds_train"].get(best, prob)
    score_psi = compute_score_psi(ref_scores, prob)

    # --- Rolling performance ---
    window_metrics = rolling_window_metrics(
        y_test, prob, amount_test, time_test, threshold=t_star
    )
    degradation_alerts = detect_degradation(window_metrics, baseline_pr_auc)

    # --- Collect alerts ---
    alerts: list[dict] = []
    for fp in feature_psi:
        if fp["severity"] == "critical":
            alerts.append({
                "type": "psi_critical",
                "feature": fp["feature"],
                "psi": fp["psi"],
                "message": (
                    f"Material drift detected in {fp['feature']} "
                    f"(PSI={fp['psi']:.3f} > {config.PSI_CRITICAL_THRESHOLD})"
                ),
            })
        elif fp["severity"] == "warning":
            alerts.append({
                "type": "psi_warning",
                "feature": fp["feature"],
                "psi": fp["psi"],
                "message": (
                    f"Moderate drift in {fp['feature']} "
                    f"(PSI={fp['psi']:.3f}, threshold={config.PSI_WARNING_THRESHOLD})"
                ),
            })
    if score_psi["severity"] in ("critical", "warning"):
        alerts.append({
            "type": f"score_psi_{score_psi['severity']}",
            "psi": score_psi["psi"],
            "message": (
                f"Score distribution drift: PSI={score_psi['psi']:.3f} "
                f"({score_psi['severity']})"
            ),
        })
    alerts.extend(degradation_alerts)

    # --- Build report ---
    report = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "best_model": best,
            "threshold": round(t_star, 6),
            "reference_period": "train split (first 70% by time)",
            "live_period": "test split (last 30% by time)",
            "psi_warning_threshold": config.PSI_WARNING_THRESHOLD,
            "psi_critical_threshold": config.PSI_CRITICAL_THRESHOLD,
            "degradation_ratio": config.DEGRADATION_RATIO,
            "n_windows": config.DRIFT_N_WINDOWS,
        },
        "feature_psi": feature_psi,
        "score_psi": score_psi,
        "rolling_performance": window_metrics,
        "baseline_pr_auc": round(baseline_pr_auc, 6),
        "alerts": alerts,
    }

    # --- Write output ---
    out = config.REPORTS_DIR / "drift.json"
    out.write_text(json.dumps(report, indent=2))
    n_alerts = len(alerts)
    status = "CLEAN" if n_alerts == 0 else f"{n_alerts} ALERT(S)"
    print(
        f"[drift-monitor] {status} | model={best} | threshold={t_star:.4f} | "
        f"baseline PR-AUC={baseline_pr_auc:.4f} | Wrote {out}"
    )
    return report


def _fast_expected_loss(
    y: np.ndarray, prob: np.ndarray, amount: np.ndarray
) -> float:
    """Fast expected loss at a coarse-grid optimal threshold for model comparison."""
    t = _fast_optimize_threshold(y, prob, amount, n_grid=50)
    return expected_dollar_loss(y, prob, amount, t)


def main() -> None:
    """CLI entry point for ``python -m sentinel.drift.drift_monitor``."""
    run_drift_monitor()


if __name__ == "__main__":
    main()
