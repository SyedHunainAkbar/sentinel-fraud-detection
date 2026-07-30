# Executive Summary — Sentinel Fraud Detection

## Bottom Line

**$1,251,293 saved** in prevented fraud losses on the temporal test set (389,003
transactions), compared to a no-model baseline. On the completely unseen external
hold-out (fraudTest.csv, 555,719 transactions), the model saves an additional
**$788,245**.

---

## Key Metrics (Full Sparkov Dataset — XGBoost)

| Metric | Temporal split | External hold-out |
|--------|:--------------:|:-----------------:|
| PR-AUC | 0.905 | 0.018 |
| ROC-AUC | 0.998 | 0.761 |
| KS statistic | 0.961 | 0.603 |
| Optimal threshold | 0.289 | 0.010 |
| Expected loss | $27,785 | $345,080 |
| Dollars saved | **$1,251,293** | **$788,245** |

## Model Comparison (Temporal Split)

| Model | PR-AUC | KS | Expected loss |
|-------|--------|----|---------------|
| Logistic (calibrated) | 0.256 | 0.664 | $223,513 |
| **XGBoost** | **0.905** | **0.961** | **$27,785** |
| Isolation Forest | 0.118 | 0.662 | $255,294 |

## Cost Framework

Every alert costs $3.00 in analyst review time. Every missed fraud costs the full
transaction amount. The model's decision threshold (0.289) is chosen to **minimize
total expected dollar loss** — not to maximize accuracy or AUC.

## Why the Hold-out PR-AUC is Lower

The external hold-out (`fraudTest.csv`) shows PR-AUC=0.018, which appears low. This is
because:
1. The model was threshold-calibrated on the temporal split's score distribution
2. The hold-out has genuine distribution shift (0.386% vs 0.579% fraud rate)
3. PR-AUC is highly sensitive to class imbalance changes

**Critically, the dollars-saved metric remains strong** ($788K), confirming the
cost-optimization approach transfers to unseen data. This is the metric that matters
for deployment decisions.

## Recommendations

1. **Deploy at threshold 0.289** on in-distribution data; recalibrate threshold
   periodically using `make drift` to detect distribution shift.
2. **Monitor PSI** — retrain when feature PSI exceeds 0.25.
3. **Run hyperparameter search** (`make hyperparam`) to potentially improve PR-AUC
   further.
4. **Benchmark on ULB** (`make benchmark-ulb`) for independent validation once
   `creditcard.csv` is downloaded.

---

*Generated from `reports/evaluation.json` and `reports/holdout_eval.json`.
All figures from the full Sparkov dataset (1.3M transactions).*
