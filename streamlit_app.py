"""Streamlit Community Cloud dashboard — self-contained, reads from reports/demo/.

This entry point does NOT import the sentinel package, so it works on Streamlit Cloud
without needing xgboost, shap, scikit-learn, etc. installed. It only needs streamlit
and pandas.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sentinel — Fraud Detection", layout="wide")
st.title("Sentinel — Cost-Sensitive Fraud Detection")

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
DEMO = REPORTS / "demo"


def _load_json(name: str) -> dict | None:
    """Load a report JSON: prefer live, fall back to demo."""
    for directory in (REPORTS, DEMO):
        path = directory / name
        if path.exists():
            return json.loads(path.read_text())
    return None


tab_exec, tab_risk = st.tabs(["Executive", "Risk"])

# ---------------------------------------------------------------------------
# Tab 1: Executive
# ---------------------------------------------------------------------------
with tab_exec:
    rep = _load_json("evaluation.json")
    if not rep:
        st.warning("No evaluation data. Run `make train && make evaluate` first.")
    else:
        best = rep["best_model"]
        m = rep["models"][best]

        c1, c2, c3 = st.columns(3)
        c1.metric("Dollars saved vs naive rule", f"${rep['dollars_saved']:,.0f}")
        c2.metric("Best model", best)
        c3.metric("Expected loss @ optimal threshold", f"${m['expected_loss']:,.0f}")

        st.subheader("Model comparison")
        st.dataframe(pd.DataFrame(rep["models"]).T)

        st.subheader("Cost vs. decision threshold")
        if "cost_curve" in rep:
            curve = pd.DataFrame(rep["cost_curve"]).set_index("threshold")
            st.line_chart(curve)

        st.caption(
            f"Dataset: {rep.get('dataset', 'N/A')} | "
            f"fraud rate {rep.get('fraud_rate', 0):.3%} | "
            f"alert budget {rep.get('alert_budget_frac', 0.005):.1%}"
        )

# ---------------------------------------------------------------------------
# Tab 2: Risk
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Quantitative Risk Analytics")
    qr = _load_json("quant_risk.json")
    if not qr:
        st.warning("No quant risk report. Run `make quant-risk` first.")
    else:
        risk_data = qr.get("loss_risk_var_es", {})
        boot = risk_data.get("bootstrap", {})
        mc = risk_data.get("monte_carlo", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("95% VaR (bootstrap)", f"${boot.get('var', 0):,.0f}")
        c2.metric("95% ES (bootstrap)", f"${boot.get('expected_shortfall', 0):,.0f}")
        c3.metric("95% VaR (Monte Carlo)", f"${mc.get('var', 0):,.0f}")
        c4.metric("95% ES (Monte Carlo)", f"${mc.get('expected_shortfall', 0):,.0f}")

        # Backtest P&L
        st.subheader("Backtest: Dollars Saved by Window")
        bt = qr.get("backtest", {})
        pnl = bt.get("per_window_pnl", [])
        if pnl:
            pnl_df = pd.DataFrame({
                "Window": list(range(1, len(pnl) + 1)),
                "Dollars Saved ($)": pnl,
            }).set_index("Window")
            st.bar_chart(pnl_df, use_container_width=True)

            ci = qr.get("dollars_saved_ci", {})
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Mean P&L", f"${bt.get('mean_pnl', 0):,.0f}")
            sc2.metric("Worst Window", f"${bt.get('worst_window', 0):,.0f}")
            sc3.metric("Consistency", f"{bt.get('consistency', 0):.0%}")
            sc4.metric(
                "95% CI",
                f"${ci.get('lower', 0):,.0f} – ${ci.get('upper', 0):,.0f}",
            )

        # PSI
        st.subheader("Feature Stability (PSI)")
        psi_data = qr.get("psi", {})
        per_feature = psi_data.get("per_feature", {})
        if per_feature:
            psi_df = pd.DataFrame({
                "Feature": list(per_feature.keys()),
                "PSI": list(per_feature.values()),
            }).sort_values("PSI", ascending=False)
            st.bar_chart(psi_df.set_index("Feature"), use_container_width=True)

            max_feat = psi_data.get("max_feature", "")
            max_val = psi_data.get("max_value", 0)
            crit = psi_data.get("critical_threshold", 0.25)
            warn = psi_data.get("warning_threshold", 0.10)
            if max_val > crit:
                st.error(f"CRITICAL: {max_feat} PSI = {max_val:.4f} > {crit}")
            elif max_val > warn:
                st.warning(f"WARNING: {max_feat} PSI = {max_val:.4f} > {warn}")
            else:
                st.success(f"All features stable (max PSI: {max_feat} = {max_val:.4f})")

        # Calibration
        st.subheader("Calibration")
        cal = qr.get("calibration", {})
        brier = cal.get("brier_decomposition", {})
        if brier:
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Brier Score", f"{brier.get('brier_score', 0):.4f}")
            bc2.metric("Reliability", f"{brier.get('reliability', 0):.4f}")
            bc3.metric("Resolution", f"{brier.get('resolution', 0):.4f}")
            bc4.metric("Uncertainty", f"{brier.get('uncertainty', 0):.4f}")

        rel_curve = cal.get("reliability_curve", {})
        if rel_curve:
            cal_df = pd.DataFrame({
                "Mean Predicted": rel_curve.get("mean_predicted", []),
                "Observed Frequency": rel_curve.get("fraction_positive", []),
            })
            cal_df["Perfect"] = cal_df["Mean Predicted"]
            st.line_chart(cal_df.set_index("Mean Predicted"), use_container_width=True)
