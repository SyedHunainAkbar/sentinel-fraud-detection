# Executive Summary — Sentinel Fraud Detection

## Bottom Line

**$1,654 saved** in prevented fraud losses on the test period, compared to a
no-model baseline that would miss all fraudulent transactions.

The cost-optimal model flags all 5 fraudulent transactions in the test set
(100% recall at the optimal threshold) while maintaining perfect precision —
zero false alarms sent to analysts.

---

## Key Metrics (sample dataset)

| Metric | Value |
|--------|-------|
| Dollars saved (vs. no detection) | **$1,654** |
| Total fraud value in test set | $1,669 |
| Best model | Calibrated logistic regression |
| Fraud caught | 5 / 5 (100%) |
| Precision@k | 1.000 |
| Recall@budget (top 0.5%) | 0.200 |
| PR-AUC | 1.000 |
| ROC-AUC | 1.000 |
| KS statistic | 1.000 |
| Brier score | 0.0005 |
| Cost-optimal threshold | 0.030 |
| Expected loss @ optimal | $15.00 |

*Note: The committed sample (1,200 rows, 16 frauds) is synthetic and intentionally
easy — perfect separation is expected. Run on the full Sparkov dataset (~1.3M
transactions) for production-realistic numbers.*

## Cost Framework

Every alert costs $3.00 in analyst review time. Every missed fraud costs the
full transaction amount. The model's decision threshold (0.030) is chosen to
**minimize total expected dollar loss** — not to maximize accuracy or AUC.

At the optimal threshold:
- Fraud prevented: $1,654
- Residual undetected fraud: $0 (sample is fully separable)
- Alert cost: $15.00 (5 alerts x $3.00)
- **Net savings: $1,654** after analyst costs

## Comparison to Naive Rules

| Strategy | Fraud caught | Alerts | Expected loss |
|----------|-------------|--------|---------------|
| No model (approve all) | 0% | 0 | $1,669 |
| Flag all (reject all) | 100% | 360 | $1,080 |
| **Sentinel (threshold 0.030)** | **100%** | **5** | **$15** |

## Model Comparison

| Model | PR-AUC | KS | Threshold | Expected loss |
|-------|--------|----|-----------|---------------|
| Logistic (calibrated) | 1.000 | 1.000 | 0.030 | $15.00 |
| XGBoost | 1.000 | 1.000 | 0.730 | $15.00 |
| Isolation Forest | 0.851 | 0.983 | 0.521 | $33.00 |

## Quant Risk Summary

- **95% VaR**: $0 (no residual loss on sample — all fraud detected)
- **Expected Shortfall**: $0
- **Backtest consistency**: 75% of windows profitable
- **Mean P&L per window**: $282
- **Max PSI drift**: amt_z_by_cat = 0.061 (stable, below 0.10 threshold)

## Recommendations

1. **Deploy on full dataset** — run `make train evaluate` with `SENTINEL_DATA`
   pointing to the full Sparkov CSV for production-scale validation.
2. **Monitor drift** — `make drift` produces feature PSI alerts; retrain
   monthly or when PSI exceeds 0.25 on key features.
3. **Run external holdout** — `make holdout` validates on the entirely separate
   fraudTest.csv for the strongest generalization evidence.
4. **Tune hyperparameters** — `make hyperparam` searches for better XGBoost
   params optimizing PR-AUC.

---

*Generated from `reports/evaluation.json` and `reports/quant_risk.json`.
All figures refer to the held-out temporal test set (sample data) and reflect
the cost-optimal decision threshold.*
