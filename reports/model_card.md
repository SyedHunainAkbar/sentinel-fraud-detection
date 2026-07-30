# Model Card — Sentinel Fraud Detection

## Model details
- **Name:** Sentinel fraud scorer
- **Best model:** xgboost
- **Version:** 0.1.0
- **Date:** 2026-07-30

## Intended use
Near-real-time scoring of credit card transactions to flag likely fraud for analyst
review. Outputs a calibrated probability and a cost-optimal accept/alert decision.
_Not_ intended as a sole automated decline mechanism without human review.

## Training data
- **Source:** Sparkov simulated transactions (Kaggle)
- **Time range:** 2019-01-01 00:00:18 .. 2020-06-21 12:13:37
- **Fraud prevalence:** 0.58%

## Metrics (xgboost)
| Metric | Value |
|---|---|
| PR-AUC | 0.9085 |
| ROC-AUC | 0.9983 |
| KS statistic | 0.9630 |
| Brier score | 0.0049 |
| Precision@k | 0.9368 |
| Recall @ alert budget | 0.7639 |

## Cost analysis
- **Cost-optimal threshold:** 0.2027
- **Expected dollar loss @ threshold:** $28,856.03
- **Naive (always-legit) loss:** $1,279,077.70
- **Dollars saved vs. naive:** $1,250,221.67

## Limitations
Severe class imbalance; results on simulated data may not transfer to production
distributions; susceptible to concept drift; geographic/coverage bias possible.
_Add project-specific limitations here._

## Ethical considerations
False positives create customer friction and analyst load; monitor fairness across
demographic segments; all alerts should receive human review before adverse action.
