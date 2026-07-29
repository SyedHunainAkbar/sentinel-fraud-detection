---
inclusion: always
---

# Technology Steering

## Language & runtime
- Python 3.11. Use type hints on all public functions. NumPy-style docstrings.

## Core libraries (do not introduce others without updating this file)
- Data: pandas, polars (for heavy IO), numpy
- ML: scikit-learn, xgboost, imbalanced-learn
- Explainability: shap
- Serving: fastapi + uvicorn (scoring API), streamlit (executive dashboard)
- Quality: pytest, pytest-cov, ruff
- Persistence: joblib for models; JSON for metrics/reports

## Engineering standards
- **Determinism**: a single `RANDOM_SEED` (see `src/sentinel/config.py`) is passed to
  every stochastic component (splits, samplers, model `random_state`). No unseeded
  randomness anywhere.
- **Reproducibility**: `requirements.txt` is pinned enough to reproduce results. The full
  pipeline must run from `make setup && make train && make evaluate`.
- **No leakage**: any statistic used as a feature (means, encoders, scalers) is `fit` on
  the training split only and applied to validation/test. Temporal split respects time.
- **No data or secrets in git**: `data/` (except `data/sample/`) and `models/` are
  gitignored. Never hardcode credentials; the Kaggle download uses the user's CLI config.

## Metrics that matter (imbalanced + cost-sensitive)
Report all of these; never report bare accuracy as a headline:
- PR-AUC (primary discrimination metric under imbalance)
- ROC-AUC (secondary)
- Precision@k and Recall at a fixed daily **alert budget**
- KS statistic (separation)
- Calibration (reliability curve + Brier score)
- **Expected dollar loss at the cost-optimal threshold** (the headline)

## Repository commands
Prefer `make` targets over ad-hoc commands so behavior is consistent across IDE, CLI,
and Kiro Web. See the `Makefile`.
