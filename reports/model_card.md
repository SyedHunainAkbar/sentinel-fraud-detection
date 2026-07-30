# Model Card — Sentinel Fraud Detection

## Model details
- **Name:** Sentinel fraud scorer
- **Best model:** baseline
- **Version:** 0.1.0
- **Date:** 2026-07-30

## Intended use
Near-real-time scoring of credit card transactions to flag likely fraud for analyst
review. Outputs a calibrated probability and a cost-optimal accept/alert decision.
_Not_ intended as a sole automated decline mechanism without human review.

## Training data
- **Source:** Sparkov simulated transactions (Kaggle)
- **Time range:** 2020-01-01 00:44:49 .. 2020-02-29 23:10:03
- **Fraud prevalence:** 1.33%

## Metrics (baseline)
| Metric | Value |
|---|---|
| PR-AUC | 1.0000 |
| ROC-AUC | 1.0000 |
| KS statistic | 1.0000 |
| Brier score | 0.0005 |
| Precision@k | 1.0000 |
| Recall @ alert budget | 0.2000 |

## Cost analysis
- **Cost-optimal threshold:** 0.0300
- **Expected dollar loss @ threshold:** $15.00
- **Naive (always-legit) loss:** $1,669.49
- **Dollars saved vs. naive:** $1,654.49

## Limitations
Severe class imbalance; results on simulated data may not transfer to production
distributions; susceptible to concept drift; geographic/coverage bias possible.
_Add project-specific limitations here._

## Ethical considerations
False positives create customer friction and analyst load; monitor fairness across
demographic segments; all alerts should receive human review before adverse action.
