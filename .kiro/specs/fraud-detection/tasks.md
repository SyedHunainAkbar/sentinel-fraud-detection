# Tasks — Fraud Detection Pipeline

Work top to bottom. Each task is small and independently verifiable. Check items off as
Kiro completes them. Scaffolded reference implementations already exist for many of
these — extend and harden them rather than rewriting from scratch (saves credits).

## 1. Foundations
- [x] 1.1 `config.py` with paths, `RANDOM_SEED`, `REVIEW_COST`, feature lists
- [x] 1.2 `ingest.load_transactions` with schema + dtype validation and dedupe
- [x] 1.3 `ingest.temporal_split` (time-ordered, no overlap)
- [ ] 1.4 Unit test: loader rejects a missing column; split preserves time order

## 2. Features (R3)
- [x] 2.1 Haversine distance helper (tested against known coordinates)
- [x] 2.2 Temporal features (hour, dow, is_night) and age from dob
- [x] 2.3 Per-category amount z-score, fit on train only
- [x] 2.4 Causal per-card 24h velocity
- [x] 2.5 `FeatureBuilder.fit/transform` assembling X, y, amount
- [ ] 2.6 Test: no-leakage check (transform uses only fitted stats); haversine correctness

## 3. Models (R4)
- [x] 3.1 Calibrated logistic-regression baseline
- [x] 3.2 XGBoost with `scale_pos_weight`
- [x] 3.3 Isolation-forest anomaly baseline
- [ ] 3.4 Persist models to `models/` via joblib

## 4. Cost-sensitive evaluation (R5, R6)
- [x] 4.1 `expected_dollar_loss` and `optimize_threshold` (min-cost sweep)
- [x] 4.2 PR-AUC, ROC-AUC, Precision@k, Recall@budget, KS, Brier
- [ ] 4.3 Test: optimizer never worse than the 0.5 default on a toy set
- [ ] 4.4 `evaluate.py` writes `reports/evaluation.json` (single source of truth)

## 5. Explainability (R7)
- [ ] 5.1 SHAP global importance saved to `reports/`
- [ ] 5.2 Local explanation helper for a single transaction

## 6. Serving (R8)
- [x] 6.1 FastAPI `/score` endpoint (loads model + FeatureBuilder)
- [x] 6.2 Streamlit dashboard reading `reports/evaluation.json`
- [ ] 6.3 API test: well-formed request returns probability + decision

## 7. Business reporting (R9)
- [ ] 7.1 `model-card` skill: generate `reports/model_card.md` from `evaluation.json`
- [ ] 7.2 Executive summary: dollars saved vs. naive rule at the alert budget, with charts

## 8. Governance & polish (R10)
- [ ] 8.1 Raise test coverage on `src/` past 80%
- [ ] 8.2 Ensure ruff is clean; CI green on the committed sample
- [ ] 8.3 Fill results into README tables from `evaluation.json`
- [ ] 8.4 Final pass: confirm no data/secrets committed; seeds fixed; runs are reproducible

## 9. Stretch (only if credits remain — hand to Kiro Web)
- [ ] 9.1 Temporal drift monitor over rolling windows (mirrors production fraud systems)
- [ ] 9.2 Hyperparameter search for XGBoost via autonomous run
- [ ] 9.3 Benchmark the same pipeline on the ULB PCA dataset for external credibility
