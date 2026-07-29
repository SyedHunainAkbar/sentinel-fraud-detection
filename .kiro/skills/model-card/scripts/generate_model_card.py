"""Generate a Markdown model card from an evaluation JSON.

Usage:
    python generate_model_card.py reports/evaluation.json reports/model_card.md
"""
from __future__ import annotations

import json
import sys
from datetime import date


def _fmt(v, kind="num"):
    if v is None:
        return "_TBD_"
    if kind == "usd":
        return f"${v:,.2f}"
    if kind == "pct":
        return f"{v:.2%}"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def build_card(ev: dict) -> str:
    best = ev.get("best_model", "TBD")
    m = ev.get("models", {}).get(best, {})
    return f"""# Model Card — Sentinel Fraud Detection

## Model details
- **Name:** Sentinel fraud scorer
- **Best model:** {best}
- **Version:** {ev.get('version', '0.1.0')}
- **Date:** {date.today().isoformat()}

## Intended use
Near-real-time scoring of credit card transactions to flag likely fraud for analyst
review. Outputs a calibrated probability and a cost-optimal accept/alert decision.
_Not_ intended as a sole automated decline mechanism without human review.

## Training data
- **Source:** {ev.get('dataset', 'Sparkov simulated transactions (Kaggle)')}
- **Time range:** {ev.get('time_range', '_TBD_')}
- **Fraud prevalence:** {_fmt(ev.get('fraud_rate'), 'pct')}

## Metrics ({best})
| Metric | Value |
|---|---|
| PR-AUC | {_fmt(m.get('pr_auc'))} |
| ROC-AUC | {_fmt(m.get('roc_auc'))} |
| KS statistic | {_fmt(m.get('ks'))} |
| Brier score | {_fmt(m.get('brier'))} |
| Precision@k | {_fmt(m.get('precision_at_k'))} |
| Recall @ alert budget | {_fmt(m.get('recall_at_budget'))} |

## Cost analysis
- **Cost-optimal threshold:** {_fmt(m.get('optimal_threshold'))}
- **Expected dollar loss @ threshold:** {_fmt(m.get('expected_loss'), 'usd')}
- **Naive (always-legit) loss:** {_fmt(ev.get('naive_loss'), 'usd')}
- **Dollars saved vs. naive:** {_fmt(ev.get('dollars_saved'), 'usd')}

## Limitations
Severe class imbalance; results on simulated data may not transfer to production
distributions; susceptible to concept drift; geographic/coverage bias possible.
_Add project-specific limitations here._

## Ethical considerations
False positives create customer friction and analyst load; monitor fairness across
demographic segments; all alerts should receive human review before adverse action.
"""


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "reports/evaluation.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "reports/model_card.md"
    with open(src) as fh:
        ev = json.load(fh)
    with open(dst, "w") as fh:
        fh.write(build_card(ev))
    print(f"Wrote model card to {dst}")


if __name__ == "__main__":
    main()
