"""Streamlit executive dashboard reading reports/evaluation.json."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from .. import config

st.set_page_config(page_title="Sentinel — Fraud Detection", layout="wide")
st.title("Sentinel — Cost-Sensitive Fraud Detection")

path = config.REPORTS_DIR / "evaluation.json"
if not path.exists():
    st.warning("No evaluation found. Run `make train && make evaluate` first.")
    st.stop()

rep = json.loads(path.read_text())
best = rep["best_model"]
m = rep["models"][best]

c1, c2, c3 = st.columns(3)
c1.metric("Dollars saved vs naive rule", f"${rep['dollars_saved']:,.0f}")
c2.metric("Best model", best)
c3.metric("Expected loss @ optimal threshold", f"${m['expected_loss']:,.0f}")

st.subheader("Model comparison")
st.dataframe(pd.DataFrame(rep["models"]).T)

st.subheader("Cost vs. decision threshold")
curve = pd.DataFrame(rep["cost_curve"]).set_index("threshold")
st.line_chart(curve)

st.caption(f"Dataset: {rep['dataset']} | fraud rate {rep['fraud_rate']:.3%} "
           f"| alert budget {rep['alert_budget_frac']:.1%}")
