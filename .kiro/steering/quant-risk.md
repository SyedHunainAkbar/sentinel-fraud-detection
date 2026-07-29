---
inclusion: always
---

# Quant Risk Steering

This project's quantitative-risk layer is what makes it credible for **model risk /
credit risk / quantitative risk** roles. Be honest about scope: this is NOT trading or
derivatives-pricing quant work. Its strength is rigorous, uncertainty-aware quantification
of a **loss distribution** and disciplined out-of-time validation — the toolkit of a risk
quant. Apply these standards to all `risk_quant` code.

## Frame fraud as a loss distribution
Undetected fraud is a random loss. At a deployed threshold t, residual loss is the sum of
amounts of frauds the model failed to flag. Quantify its distribution, not just its mean:
- **VaR(alpha)**: the alpha-quantile of the residual-loss distribution (e.g., 95%).
- **Expected Shortfall(alpha)**: mean loss in the worst (1-alpha) tail — the coherent
  risk measure. Always report ES alongside VaR.
- Estimate the distribution by **block bootstrap** over transactions (empirical, honest)
  and cross-check with a **Monte Carlo** Bernoulli simulation of detection.

## Validate like a strategy, not like a classifier
- **Out-of-time rolling backtest**: fit the threshold on earlier windows, measure realized
  dollars-saved P&L on strictly later windows. Report per-window P&L, its mean, its
  volatility, and its worst window. Consistency across time matters more than a single
  headline number (this is the analogue of a stable Sharpe).
- **Bootstrap confidence intervals** on dollars-saved so every headline number has an
  interval, never a bare point estimate.

## Calibration & stability (credit-risk staples)
- Probabilities must be **calibrated** (reliability curve, Brier decomposition). Treat the
  fraud score like a PD estimate.
- **PSI (Population Stability Index)** between training and live feature distributions;
  PSI > 0.25 signals material drift and should trigger review.

## Reporting
Emit `reports/quant_risk.json` with VaR, ES, backtest P&L series + CI, and PSI. Numbers in
prose must trace back to this file — never invent them.
