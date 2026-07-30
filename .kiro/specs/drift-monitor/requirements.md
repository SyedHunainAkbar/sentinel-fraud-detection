# Requirements — Production Drift Monitor

## Introduction

A production drift monitor that continuously tracks Population Stability Index (PSI) and
rolling-window model performance for the deployed fraud model. Detects distribution shift
early so the operations team can trigger revalidation before the model silently degrades.
Aligns with the quant-risk steering: PSI > 0.25 signals material drift and must trigger
review.

## Definitions

- **Reference distribution**: feature and score distributions from the training set (the
  baseline the model was validated against).
- **Live distribution**: feature and score distributions from new incoming transactions
  (simulated via the test split or a rolling window in batch mode).
- **PSI**: Population Stability Index — symmetric KL-divergence discretized into bins.
- **Rolling window**: a configurable time-based or count-based sliding window over which
  performance metrics are recomputed.

---

## R1 — Per-feature PSI computation (EARS: ubiquitous)

The system SHALL compute PSI for every numeric feature in `config.NUMERIC_FEATURES` plus
the model's predicted probability, comparing the reference (train) distribution against
the live (current window) distribution.

## R2 — PSI alerting threshold (EARS: event-driven)

WHEN PSI for any tracked feature exceeds 0.25, the system SHALL flag that feature as
materially drifted in the output report with severity "critical".

WHEN PSI for any tracked feature is between 0.10 and 0.25, the system SHALL flag that
feature with severity "warning".

## R3 — Rolling-window performance tracking (EARS: ubiquitous)

The system SHALL compute PR-AUC, expected dollar loss, and recall at the alert budget
over a configurable rolling window (default: each of `n_windows` sequential time-ordered
segments of the test set).

## R4 — Performance degradation detection (EARS: event-driven)

WHEN rolling-window PR-AUC drops below 80% of the full-test-set PR-AUC, the system SHALL
emit a degradation alert in the report.

## R5 — Report output (EARS: ubiquitous)

The system SHALL write `reports/drift.json` containing:
- Per-feature PSI values and their severity classification.
- Per-window performance metrics (PR-AUC, expected loss, recall at budget).
- A top-level `alerts` array listing all triggered drift/degradation flags.
- Metadata: timestamp, reference period, live period, threshold configuration.

## R6 — Determinism (EARS: ubiquitous)

The system SHALL use `config.RANDOM_SEED` for any stochastic operation (bootstrap, etc.)
so results are reproducible across runs.

## R7 — Integration with existing artifacts (EARS: state-driven)

IF trained artifacts exist in `models/artifacts.joblib`, the system SHALL load them and
derive reference distributions from the training features persisted therein.

IF artifacts do not exist, the system SHALL exit with a clear error message directing the
user to run `make train` first.

## R8 — Make target (EARS: ubiquitous)

The system SHALL be runnable via `make drift` so behavior is consistent with the existing
`make train`, `make evaluate`, and `make quant-risk` targets.

## R9 — Testability (EARS: ubiquitous)

The system SHALL expose pure functions for PSI computation and window-based metric
calculation that are independently testable with the sample CSV.

---

## Non-functional constraints

| Constraint          | Detail                                                     |
|---------------------|------------------------------------------------------------|
| Language            | Python 3.11, type hints on public functions                |
| Libraries           | numpy, pandas, scikit-learn only (no new dependencies)     |
| Persistence         | JSON for reports (`reports/drift.json`)                    |
| Config              | All paths and thresholds via `config.py` constants         |
| No leakage          | Reference distributions derived from train split only      |
| Git hygiene         | `reports/drift.json` is gitignored (generated artifact)    |
