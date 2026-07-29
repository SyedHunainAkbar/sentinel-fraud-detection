# Requirements — Fraud Detection Pipeline

## Introduction
Sentinel scores credit card transactions to detect fraud while minimizing expected dollar
loss. This document captures requirements as user stories with EARS-style acceptance
criteria. Personas: Data Scientist (DS), Quant (Q), Business Analyst (BA).

---

## R1 — Data ingestion & validation
**Story:** As a DS, I want validated, schema-checked transaction data so downstream code
can assume clean inputs.
- WHEN a transactions CSV is loaded, the system SHALL verify the expected columns exist
  and coerce dtypes, raising a clear error on mismatch.
- WHEN duplicate `trans_num` values exist, the system SHALL drop duplicates.
- IF a required column is missing, THEN the system SHALL fail fast with the column name.

## R2 — Leakage-free temporal split
**Story:** As a Q, I want a time-ordered train/test split so evaluation reflects reality.
- WHEN splitting, the system SHALL order by transaction time and assign the earliest
  fraction to train and the latest to test, with no overlap across the cutoff.

## R3 — Feature engineering (causal, leakage-free)
**Story:** As a DS, I want interpretable engineered features fit only on training data.
- The system SHALL derive: haversine distance (customer↔merchant), hour, day-of-week,
  is_night, customer age, log amount, per-category amount z-score, and per-card
  transaction velocity (trailing 24h count).
- Category z-score statistics SHALL be fit on train only and applied to test.
- Per-card velocity SHALL use only prior transactions (causal).

## R4 — Model training with imbalance handling
**Story:** As a DS, I want multiple models with class-imbalance handling to compare.
- The system SHALL train a calibrated logistic-regression baseline, an XGBoost model with
  `scale_pos_weight`, and an isolation-forest anomaly baseline.
- All stochastic components SHALL use the shared random seed.

## R5 — Cost-sensitive threshold optimization
**Story:** As a Q, I want the decision threshold chosen to minimize expected dollar loss.
- The system SHALL compute total expected dollar loss across candidate thresholds using
  the cost matrix in `risk.md` and select the loss-minimizing threshold.
- The system SHALL persist the full threshold-vs-cost curve for reporting.

## R6 — Evaluation metrics
**Story:** As a DS/Q, I want imbalanced-appropriate metrics.
- The system SHALL report PR-AUC, ROC-AUC, Precision@k, Recall at a fixed alert budget,
  KS statistic, Brier score, and expected dollar loss at the optimal threshold.

## R7 — Explainability
**Story:** As a BA/regulator, I want to know why a transaction was flagged.
- The system SHALL produce SHAP global importance and support per-transaction local
  explanations for the tree model.

## R8 — Serving
**Story:** As an engineer, I want to score transactions via an API and view results.
- The system SHALL expose a FastAPI `/score` endpoint returning probability + decision.
- The system SHALL provide a Streamlit dashboard reading `reports/evaluation.json`.

## R9 — Business reporting
**Story:** As a BA, I want a plain-English executive summary.
- The system SHALL generate an executive summary that leads with dollars saved vs. a
  naive always-legit rule at the chosen alert budget, with supporting charts.

## R10 — Reproducibility, tests, governance
**Story:** As a hiring reviewer, I want to reproduce and trust the results.
- The full pipeline SHALL run via documented `make` targets on the committed sample.
- The system SHALL have pytest coverage on `src/` (target >80%) and pass ruff lint.
- The system SHALL emit a model card documenting use, metrics, cost, and limitations.
- No real data or secrets SHALL be committed.
