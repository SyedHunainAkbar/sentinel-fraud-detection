---
name: model-card
description: Generate a standardized model card for a fraud/risk model from an evaluation JSON — intended use, training data, metrics, cost analysis, limitations, and ethical considerations. Use when documenting a trained model, preparing a model for review, or producing governance/regulatory documentation for the Sentinel pipeline.
---

# Model Card Generator

Financial models face regulatory scrutiny, so every trained model ships with a model card.
Generate it deterministically from `reports/evaluation.json` so the card and the reported
numbers never drift apart.

## Required sections
1. **Model details** — name, version, date, model type, owner.
2. **Intended use** — near-real-time transaction fraud scoring; who uses it and how.
3. **Training data** — source dataset, time range, class balance, preprocessing.
4. **Metrics** — PR-AUC, ROC-AUC, KS, Brier, Precision@k, Recall at the alert budget.
5. **Cost analysis** — cost matrix, chosen threshold, expected dollar loss, dollars saved
   vs. a naive always-legit rule at the alert budget.
6. **Limitations** — imbalance, synthetic-data caveats, concept drift, geography/coverage.
7. **Ethical considerations** — customer friction from false positives, fairness across
   demographics, need for human review of alerts.

## Usage
Run `python .kiro/skills/model-card/scripts/generate_model_card.py \
  reports/evaluation.json reports/model_card.md`. The script fills the template from the
evaluation JSON and leaves narrative sections as clearly-marked prompts to complete.
