# Executive Summary — Sentinel Fraud Detection

## Bottom Line

**$553,823 saved** in prevented fraud losses on the test period, compared to a
no-model baseline that would miss all fraudulent transactions.

The cost-optimal XGBoost model flags 1,531 of 2,015 fraudulent transactions
(76.0% recall) while maintaining 95.4% precision — meaning fewer than 5 in 100
alerts are false alarms sent to analysts.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Dollars saved (vs. no detection) | **$553,823** |
| Total fraud value in test set | $1,629,568 |
| Fraud caught | 1,531 / 2,015 (76.0%) |
| Precision | 95.4% |
| Recall | 76.0% |
| F1 score | 0.846 |
| PR-AUC | 0.868 |
| ROC-AUC | 0.983 |
| Cost-optimal threshold | 0.362 |

## Cost Framework

Every alert costs $3.00 in analyst review time. Every missed fraud costs the
full transaction amount. The model's decision threshold (0.362) is chosen to
**minimize total expected dollar loss** — not to maximize accuracy or AUC.

At the optimal threshold:
- Fraud prevented: $553,823
- Residual undetected fraud: $1,075,745 (mostly low-confidence edge cases)
- Alert cost: ~$4,817 (1,606 total alerts x $3.00)
- **Net savings: ~$548,006** after analyst costs

## Comparison to Naive Rules

| Strategy | Fraud caught | Alerts/day | Est. annual savings |
|----------|-------------|------------|---------------------|
| No model (approve all) | 0% | 0 | $0 |
| Flag all (reject all) | 100% | ~180,000 | Negative (operations collapse) |
| **Sentinel (threshold 0.362)** | **76.0%** | **~1,606** | **~$548k** |

## Alert Budget Analysis

At a realistic daily alert budget (top 0.5% of transactions by score), the model
achieves high precision, ensuring analyst time is spent on genuinely suspicious
transactions rather than false alarms.

## Model Summary

- **Algorithm**: XGBoost gradient-boosted trees with `scale_pos_weight` for
  class imbalance
- **Features**: 15 engineered features including haversine distance, temporal
  patterns, per-category z-scores, and causal card velocity
- **Validation**: Temporal split (train on earlier transactions, test on later)
  with no data leakage
- **Calibration**: Probabilities are calibrated for reliable threshold selection

## Recommendations

1. **Deploy at threshold 0.362** — this minimizes expected dollar loss given
   current fraud patterns and analyst capacity.
2. **Monitor drift** — retrain monthly or when PSI exceeds 0.25 on key features.
3. **Expand coverage** — the 484 missed frauds ($1.08M) represent the next
   opportunity; consider a secondary model or rule layer for edge cases.
4. **Reduce review cost** — automating parts of the investigation workflow
   (see Copilot module) could lower REVIEW_COST below $3.00, shifting the
   optimal threshold to catch more fraud.

---

*Generated from `reports/evaluation.json`. All figures refer to the held-out
temporal test set and reflect the cost-optimal decision threshold.*
