# Requirements — Quant Risk Analytics

## Introduction
Quantify the risk of undetected fraud as a loss distribution and validate the deployed
policy like a strategy. Targets model-risk / credit-risk / quantitative-risk roles.

## R1 — Loss distribution & tail risk
- The system SHALL estimate the residual (undetected) fraud loss distribution via block
  bootstrap and cross-check with a Monte Carlo Bernoulli-detection simulation.
- The system SHALL report VaR(alpha) and Expected Shortfall(alpha), with ES >= VaR.

## R2 — Out-of-time backtest
- The system SHALL run a walk-forward backtest: threshold fit on earlier windows, realized
  dollars-saved measured on strictly later windows.
- The system SHALL report per-window P&L, mean, volatility, worst window, and consistency.

## R3 — Confidence intervals
- Headline dollars-saved SHALL carry a bootstrap confidence interval, never a bare point.

## R4 — Stability
- The system SHALL compute PSI between reference and live distributions; PSI > 0.25 SHALL
  be flagged as material drift.

## R5 — Reporting
- The system SHALL write `reports/quant_risk.json` as the single source of quant numbers.
