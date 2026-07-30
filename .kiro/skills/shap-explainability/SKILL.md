---
name: shap-explainability
description: Produce SHAP-based global and local explanations for the Sentinel tree model — summary bar plot, beeswarm, top-N local force explanations, and a JSON summary of feature importances. Use when generating model explanations, preparing governance documentation, or investigating individual alert decisions.
---

# SHAP Explainability

Financial models operate under regulatory scrutiny. Provide SHAP-based global and local
explanations so every alert can be traced to the features that drove the decision.

## Outputs
1. **Global importance** — mean |SHAP value| per feature, saved as
   `reports/shap_summary.json` and `reports/shap_global_bar.png`.
2. **Beeswarm plot** — full feature-value interaction view, saved as
   `reports/shap_beeswarm.png`.
3. **Local explanations** — top-N highest-risk transactions with per-feature SHAP
   contributions, saved in the `local_explanations` key of `reports/shap_summary.json`.

## Design constraints
- Uses `shap.TreeExplainer` (exact, fast for tree models).
- Runs on the test set (or a subsample for large datasets, capped at 5000 rows).
- Feature names come from `config.NUMERIC_FEATURES`.
- Deterministic: subsampling uses `config.RANDOM_SEED`.
- No additional dependencies beyond `shap` (already in `requirements.txt`).

## Usage
```
python .kiro/skills/shap-explainability/scripts/generate_shap.py
```
Or via the Makefile:
```
make shap
```

## Integration
- `reports/shap_summary.json` is the machine-readable output; the dashboard and model
  card can reference it.
- Plots are standalone PNGs for inclusion in reports or slide decks.
- The script loads `models/artifacts.joblib` — requires `make train` to have been run.
