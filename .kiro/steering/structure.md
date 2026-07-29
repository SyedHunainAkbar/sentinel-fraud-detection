---
inclusion: always
---

# Structure Steering

## Directory layout (keep code in the right place)
```
src/sentinel/
  config.py        # paths, seed, cost parameters, feature lists — single source of truth
  ingest.py        # load + validate transactions; temporal train/test split
  features.py      # FeatureBuilder: fit/transform, leakage-free engineered features
  models.py        # train_baseline / train_xgboost / train_isolation_forest
  evaluation.py    # metrics, cost curve, threshold optimization, report JSON
  train.py         # orchestrates: ingest -> features -> train -> persist artifacts
  evaluate.py      # loads artifacts -> full report -> reports/evaluation.json + summary
  serving/
    api.py         # FastAPI /score endpoint
    dashboard.py   # Streamlit executive dashboard
tests/             # pytest; mirror src module names (test_features.py, ...)
reports/           # generated: evaluation.json, exec_summary.md, model_card.md, plots
data/
  sample/          # tiny committed CSV so tests/CI run without the Kaggle download
  README.md        # how to fetch the full dataset
```

## Conventions
- Module names are lowercase; classes are `CapWords`; functions/vars are `snake_case`.
- Public functions take explicit arguments; no hidden global state except `config`.
- Every new `src` module gets a matching `tests/test_<module>.py`.
- Feature engineering functions are **pure** where possible and independently testable.
- Anything that reads or writes a path uses `config.py` constants, not string literals.

## Artifact contract
- Trained models are saved to `models/` (gitignored) as joblib files.
- `train.py` writes test-set predictions + amounts so `evaluate.py` needs no retraining.
- `evaluate.py` is the only writer of `reports/evaluation.json`; the dashboard and model
  card read from it. This keeps a single source of truth for reported numbers.
