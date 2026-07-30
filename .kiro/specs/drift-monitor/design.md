# Design — Production Drift Monitor

## Overview

The drift monitor is a batch-mode module that loads trained artifacts, computes feature-
and score-level PSI against the training reference, evaluates rolling-window performance,
and emits `reports/drift.json` with alerts. It is designed for periodic execution (cron,
CI, or manual `make drift`) rather than real-time streaming.

---

## Data-Flow Diagram

```
┌─────────────────────┐
│ models/             │
│  artifacts.joblib   │──────────┐
└─────────────────────┘          │
                                 ▼
                    ┌────────────────────────┐
                    │   drift_monitor.py     │
                    │   (orchestrator)       │
                    │                        │
                    │  1. Load artifacts     │
                    │  2. Extract ref/live   │
                    │  3. Dispatch PSI       │
                    │  4. Dispatch perf      │
                    │  5. Classify alerts    │
                    │  6. Write report       │
                    └───┬──────────┬─────────┘
                        │          │
            ┌───────────┘          └───────────┐
            ▼                                  ▼
┌───────────────────────┐        ┌──────────────────────────┐
│  drift_psi.py         │        │  drift_performance.py    │
│                       │        │                          │
│  compute_feature_psi()│        │  rolling_window_metrics()│
│  classify_severity()  │        │  detect_degradation()    │
└───────────────────────┘        └──────────────────────────┘
            │                                  │
            │    uses                          │    uses
            ▼                                  ▼
┌───────────────────────┐        ┌──────────────────────────┐
│  risk_quant/          │        │  evaluation.py           │
│   stability.psi()     │        │   pr_auc, expected_loss  │
└───────────────────────┘        │   recall_at_budget       │
                                 └──────────────────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
                ┌──────────────────────┐
                │  reports/drift.json  │
                └──────────────────────┘
```

---

## Module Responsibilities

### `src/sentinel/drift/drift_psi.py`

Computes per-feature PSI and classifies drift severity.

```python
def compute_feature_psi(
    ref_features: pd.DataFrame,
    live_features: pd.DataFrame,
    feature_names: list[str],
    bins: int = 10,
) -> list[dict]:
    """PSI for each feature; returns list of {feature, psi, severity}."""

def compute_score_psi(
    ref_scores: np.ndarray,
    live_scores: np.ndarray,
    bins: int = 10,
) -> dict:
    """PSI on predicted probability distribution."""

def classify_severity(psi_value: float) -> str:
    """'stable' | 'warning' | 'critical' based on 0.10/0.25 thresholds."""
```

Delegates the actual PSI calculation to `risk_quant.stability.psi()` — no reimplementation.

### `src/sentinel/drift/drift_performance.py`

Evaluates model performance over rolling time windows.

```python
def rolling_window_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    unix_time: np.ndarray,
    threshold: float,
    n_windows: int = 6,
) -> list[dict]:
    """Per-window PR-AUC, expected loss, recall@budget."""

def detect_degradation(
    window_metrics: list[dict],
    baseline_pr_auc: float,
    degradation_ratio: float = 0.80,
) -> list[dict]:
    """Windows where PR-AUC < degradation_ratio * baseline."""
```

Uses `evaluation.py` functions for metric computation — no duplication.

### `src/sentinel/drift/drift_monitor.py`

Orchestrator; the `__main__` entry point.

```python
def run_drift_monitor() -> dict:
    """Full pipeline: load -> PSI -> performance -> alerts -> write JSON."""

def main() -> None:
    """CLI entry point invoked by `make drift`."""
```

### `src/sentinel/drift/__init__.py`

Package marker; exports `run_drift_monitor` for programmatic use.

---

## Report Schema (`reports/drift.json`)

```json
{
  "metadata": {
    "generated_at": "2026-07-30T02:30:00Z",
    "reference_period": "train split (first 70% by time)",
    "live_period": "test split (last 30% by time)",
    "psi_critical_threshold": 0.25,
    "psi_warning_threshold": 0.10,
    "degradation_ratio": 0.80,
    "n_windows": 6
  },
  "feature_psi": [
    {"feature": "distance_km", "psi": 0.03, "severity": "stable"},
    {"feature": "velocity_24h", "psi": 0.28, "severity": "critical"}
  ],
  "score_psi": {"psi": 0.05, "severity": "stable"},
  "rolling_performance": [
    {
      "window": 1,
      "n_transactions": 64800,
      "pr_auc": 0.89,
      "expected_loss": 12450.0,
      "recall_at_budget": 0.72
    }
  ],
  "baseline_pr_auc": 0.905,
  "alerts": [
    {
      "type": "psi_critical",
      "feature": "velocity_24h",
      "psi": 0.28,
      "message": "Material drift detected in velocity_24h (PSI=0.28 > 0.25)"
    },
    {
      "type": "performance_degradation",
      "window": 4,
      "pr_auc": 0.68,
      "message": "PR-AUC dropped to 0.68 (< 80% of baseline 0.905)"
    }
  ]
}
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Separate `drift_psi` and `drift_performance` modules | Single responsibility; each is independently testable (R9) |
| Reuse `stability.psi()` | No reimplementation; proven function from quant-risk layer |
| Reuse `evaluation.py` metrics | Consistent metric definitions across pipeline |
| New `src/sentinel/drift/` package | Keeps drift logic separate from the existing `risk_quant/` (which models loss distributions, not monitoring) |
| Rolling windows by time sort | Respects temporal ordering (no leakage); consistent with backtest design |
| Alerts as a flat array | Easy to filter/count programmatically; dashboard can render directly |
| Reference = train features from artifacts | No leakage: reference is always the distribution the model was validated against |
| Batch mode, not streaming | Matches current pipeline architecture; streaming is a future extension |

---

## Integration Points

- **Config**: New constants in `config.py` for drift thresholds (`PSI_WARNING = 0.10`,
  `PSI_CRITICAL = 0.25`, `DEGRADATION_RATIO = 0.80`, `DRIFT_N_WINDOWS = 6`).
- **Artifacts**: Requires `artifacts.joblib` to contain `X_train_features` (reference
  distributions) in addition to the existing `y_test`, `preds`, `amount_test`, `time_test`.
  This means `train.py` must persist the training feature matrix.
- **Makefile**: New `drift` target calling `python -m sentinel.drift.drift_monitor`.
- **Tests**: `tests/test_drift_psi.py` and `tests/test_drift_performance.py`.

---

## What This Is Not

- Not a real-time streaming monitor (batch only).
- Not a retraining trigger (flags for human review; humans decide).
- Not a replacement for `quant_risk.json` (complementary: drift detects *when* to
  revalidate; quant-risk quantifies *what* the loss looks like).
