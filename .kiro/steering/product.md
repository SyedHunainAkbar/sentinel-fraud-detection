---
inclusion: always
---

# Product Steering — Sentinel

## What this project is
Sentinel is a **cost-sensitive credit card fraud detection platform**. It scores card
transactions in near-real-time and flags likely fraud, optimizing for **minimized
expected dollar loss** rather than raw accuracy.

## Why it exists (the framing that must appear everywhere)
Fraud detection is an **imbalanced, cost-sensitive** problem. Two errors have very
different costs:
- A **missed fraud** (false negative) costs the full transaction amount.
- A **false alarm** (false positive) costs analyst review time plus customer friction.

Therefore *every* modeling decision is justified in **dollars of expected loss**, never
in accuracy or AUC alone. Any output, chart, or report that does not connect back to
dollar impact is incomplete.

## Three audiences this project must serve simultaneously
This is a portfolio project targeting three roles. Each deliverable should speak to at
least one persona; the README and reports must speak to all three.

1. **Data Scientist** — rigor: imbalance handling, model comparison, calibration,
   leakage-free temporal validation, explainability (SHAP), tests, reproducibility.
2. **Quant** — economics: an explicit cost matrix, expected-loss derivation,
   threshold optimization, backtesting over time, calibrated probabilities.
3. **Business Analyst** — communication: an executive summary in plain English,
   dollars saved vs. a naive rule, alert-budget tradeoffs, clear visuals.

## Definition of success
- A reproducible pipeline runnable via `make` targets end to end.
- A cost-optimal decision threshold, chosen to minimize expected dollar loss.
- A leakage-free temporal backtest (train on earlier period, test on later).
- An executive summary that leads with a dollar figure, not a metric.
- A model card documenting intended use, metrics, cost analysis, and limitations.

## Non-goals
- Chasing state-of-the-art deep/graph models for marginal AUC gains.
- Any use of real cardholder PII. Only public/synthetic data; data never committed.
