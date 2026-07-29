# Design — Quant Risk Analytics

## Modules
- `risk_quant/loss_model.py` — VaR, Expected Shortfall, bootstrap + Monte Carlo loss.
- `risk_quant/backtest.py` — walk-forward out-of-time backtest + bootstrap CI.
- `risk_quant/stability.py` — Population Stability Index.
- `risk_quant/quant_report.py` — orchestration -> `reports/quant_risk.json`.

## Residual loss
At threshold t, a fraud is undetected if its predicted probability < t. Residual loss is
the sum of undetected fraud amounts. Bootstrap resamples transactions into synthetic
periods to build the loss distribution; Monte Carlo simulates fraud occurrence from the
model's probabilities as a cross-check.

## Backtest
Sort by time, split into k sequential windows; for each, fit the cost-optimal threshold on
all earlier data and measure dollars-saved on the window. Stability of P&L across windows
is the risk-quant analogue of a consistent strategy return.

## Positioning (honest)
This is quantitative RISK work (loss modeling, tail risk, backtesting, calibration, PSI),
not trading/derivatives quant. State that plainly in interviews.
