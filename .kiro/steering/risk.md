---
inclusion: always
---

# Risk & Domain Steering (fraud economics)

This file encodes the domain reasoning that makes the project read as senior. Kiro should
apply it whenever generating modeling, evaluation, or reporting code.

## The cost matrix (dollars)
For a decision at threshold t, per transaction:

| Actual \ Predicted | Legit (t) | Fraud (t) |
|--------------------|-----------|-----------|
| **Legit**          | 0         | REVIEW_COST (false alarm: analyst time + friction) |
| **Fraud**          | AMOUNT (missed — full loss) | REVIEW_COST (caught — loss prevented) |

- `REVIEW_COST` is a fixed per-alert cost (default $3.00 in `config.py`).
- The cost of a missed fraud is the **transaction amount**, so cost is amount-weighted.
- Total cost at threshold t = (REVIEW_COST x alerts) + (sum of amounts of missed frauds).

## Threshold selection
Do **not** use the default 0.5 cutoff. Sweep candidate thresholds over the predicted-
probability distribution and pick the one that **minimizes total expected dollar loss**
on the validation split. Report the full cost-vs-threshold curve, not just the optimum.

## Class imbalance
Fraud is well under 1% of transactions. Handle with class weighting / `scale_pos_weight`
and evaluate with PR-AUC and alert-budget metrics — never headline accuracy, which is
trivially ~99% by predicting "legit" always.

## Leakage & temporal validity
Fraud data is a time series. Split by time (earlier = train, later = test). Never shuffle
across the cutoff. Features that use aggregates (per-card velocity, category z-scores)
must be computed causally (only past information) and fit on train only.

## Explainability & governance
Financial models operate under regulatory scrutiny. Provide SHAP-based global and local
explanations, and emit a **model card** (intended use, training data, metrics, cost
analysis, limitations, ethical considerations). Log the alert budget and expected loss so
decisions are auditable.

## Alert budget
Analysts can only review so many alerts per day. Report Precision@k and Recall at a fixed
budget (e.g., top 0.5% of transactions by score) so the model is evaluated the way an
operations team would actually deploy it.
