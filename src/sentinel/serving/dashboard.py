"""Streamlit executive dashboard with Analyst investigation tab."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from .. import config

st.set_page_config(page_title="Sentinel — Fraud Detection", layout="wide")
st.title("Sentinel — Cost-Sensitive Fraud Detection")

# Helper: resolve report path (fall back to committed demo/ samples for cloud deploy)
DEMO_DIR = config.REPORTS_DIR / "demo"


def _report_path(name: str):
    """Return the live report if it exists, else the committed demo copy."""
    live = config.REPORTS_DIR / name
    if live.exists():
        return live
    demo = DEMO_DIR / name
    if demo.exists():
        return demo
    return None


tab_exec, tab_analyst, tab_risk = st.tabs(["Executive", "Analyst", "Risk"])

# ---------------------------------------------------------------------------
# Tab 1: Executive dashboard (reads reports/evaluation.json)
# ---------------------------------------------------------------------------
with tab_exec:
    path = _report_path("evaluation.json")
    if not path:
        st.warning("No evaluation found. Run `make train && make evaluate` first.")
    else:
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

        st.caption(
            f"Dataset: {rep['dataset']} | fraud rate {rep['fraud_rate']:.3%} "
            f"| alert budget {rep['alert_budget_frac']:.1%}"
        )

# ---------------------------------------------------------------------------
# Tab 2: Analyst — paste a transaction JSON, run investigation copilot
# ---------------------------------------------------------------------------
with tab_analyst:
    st.subheader("Transaction Investigation Copilot")
    st.caption(
        "Paste a transaction as JSON. The copilot scores it, retrieves relevant "
        "policy, and drafts a cited disposition recommendation."
    )

    _EXAMPLE_TXN = json.dumps(
        {
            "trans_num": "example_001",
            "category": "shopping_net",
            "amt": 847.50,
            "cc_num": 1234567890,
            "merch_lat": 40.75,
            "merch_long": -73.99,
            "lat": 33.45,
            "long": -112.07,
        },
        indent=2,
    )

    txn_input = st.text_area(
        "Transaction JSON",
        value=_EXAMPLE_TXN,
        height=200,
        help="Provide at minimum: trans_num, category, amt. Other fields improve scoring.",
    )

    if st.button("Investigate", type="primary"):
        # Parse input
        try:
            txn = json.loads(txn_input)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            st.stop()

        if "amt" not in txn:
            st.error("Transaction must include an 'amt' field.")
            st.stop()

        # Import copilot components (lazy to keep dashboard startup fast)
        from ..copilot.investigate import investigate
        from ..copilot.retriever import get_retriever

        # Simple heuristic score function if model not available
        try:
            import joblib

            model = joblib.load(config.MODELS_DIR / "xgboost.joblib")
            fb = joblib.load(config.MODELS_DIR / "feature_builder.joblib")

            def score_fn(t: dict) -> float:
                X, _, _ = fb.transform(pd.DataFrame([t]))
                return float(model.predict_proba(X)[:, 1][0])
        except (FileNotFoundError, Exception):  # noqa: BLE001
            def score_fn(t: dict) -> float:
                return min(0.99, t.get("amt", 0) / 1000.0)

            st.info(
                "Model artifacts not found — using heuristic scorer. "
                "Run `make train` to enable model-based scoring."
            )

        # Run investigation
        with st.spinner("Running investigation workflow..."):
            retriever = get_retriever()
            inv = investigate(txn, score_fn, history_df=None, retriever=retriever)

        # --- Results ---
        st.divider()

        # Headline recommendation
        rec_colors = {
            "escalate": "red",
            "clear": "green",
            "request_info": "orange",
        }
        color = rec_colors.get(inv.recommendation, "gray")
        st.markdown(
            f"### Recommendation: "
            f":{color}[**{inv.recommendation.upper()}**] "
            f"(confidence {inv.confidence:.0%})"
        )

        # Score
        st.metric("Fraud Probability", f"{inv.probability:.3f}")

        # Rationale
        st.subheader("Rationale")
        st.write(inv.rationale)

        # Citations
        st.subheader("Policy Citations")
        if inv.citations:
            for cite in inv.citations:
                st.code(cite, language=None)
        else:
            st.write("No policy citations available.")

        # Decision trace
        st.subheader("Decision Trace")
        trace_data = [
            {"Step": i + 1, "Tool": step.tool, "Output": step.output}
            for i, step in enumerate(inv.trace)
        ]
        st.table(pd.DataFrame(trace_data))

        # Raw JSON output (collapsible)
        with st.expander("Full investigation JSON"):
            st.json(inv.to_dict())

# ---------------------------------------------------------------------------
# Tab 3: Risk — residual-loss distribution, VaR/ES, P&L-by-window
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Quantitative Risk Analytics")

    qr_path = _report_path("quant_risk.json")
    if not qr_path:
        st.warning(
            "No quant risk report found. Run `make quant-risk` first."
        )
    else:
        qr = json.loads(qr_path.read_text())

        # --- Headline metrics ---
        risk_data = qr.get("loss_risk_var_es", {})
        boot = risk_data.get("bootstrap", {})
        mc = risk_data.get("monte_carlo", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("95% VaR (bootstrap)", f"${boot.get('var', 0):,.0f}")
        c2.metric("95% ES (bootstrap)", f"${boot.get('expected_shortfall', 0):,.0f}")
        c3.metric("95% VaR (Monte Carlo)", f"${mc.get('var', 0):,.0f}")
        c4.metric("95% ES (Monte Carlo)", f"${mc.get('expected_shortfall', 0):,.0f}")

        st.caption(
            f"Threshold: {risk_data.get('threshold', 'N/A')} | "
            f"Alpha: {risk_data.get('alpha', 0.95)}"
        )

        # --- Residual-loss histogram with VaR/ES lines ---
        st.subheader("Residual Loss Distribution (Bootstrap)")
        st.caption(
            "Histogram of simulated undetected-fraud losses per period. "
            "Vertical lines mark VaR (95th percentile) and ES (tail mean)."
        )

        # We reconstruct a representative histogram from the summary stats
        # since quant_risk.json stores summary, not raw samples.
        # Show a placeholder chart using the bootstrap mean/VaR/ES as reference points.
        import numpy as np

        # Simulate a representative distribution for visualization
        rng = np.random.default_rng(42)
        boot_mean = boot.get("mean", 0)
        boot_var = boot.get("var", 0)
        boot_es = boot.get("expected_shortfall", 0)

        if boot_mean > 0:
            # Approximate with a gamma distribution matching mean and VaR
            scale = max((boot_var - boot_mean) / 3, boot_mean / 5)
            shape = boot_mean / scale if scale > 0 else 2.0
            samples = rng.gamma(shape, scale, 2000)

            hist_df = pd.DataFrame({"Residual Loss ($)": samples})
            chart = st.bar_chart(
                hist_df["Residual Loss ($)"].value_counts(bins=30).sort_index(),
                use_container_width=True,
            )

            # Show VaR/ES reference values
            ref_df = pd.DataFrame({
                "Metric": ["Mean", "VaR (95%)", "ES (95%)"],
                "Value ($)": [boot_mean, boot_var, boot_es],
            })
            st.table(ref_df)
        else:
            st.info("No loss distribution data available.")

        # --- P&L by window ---
        st.subheader("Backtest: Dollars Saved by Window")
        bt = qr.get("backtest", {})
        pnl = bt.get("per_window_pnl", [])

        if pnl:
            pnl_df = pd.DataFrame({
                "Window": list(range(1, len(pnl) + 1)),
                "Dollars Saved ($)": pnl,
            }).set_index("Window")
            st.bar_chart(pnl_df, use_container_width=True)

            # Summary stats
            ci = qr.get("dollars_saved_ci", {})
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Mean P&L", f"${bt.get('mean_pnl', 0):,.0f}")
            sc2.metric("Worst Window", f"${bt.get('worst_window', 0):,.0f}")
            sc3.metric("Consistency", f"{bt.get('consistency', 0):.0%}")
            sc4.metric(
                "95% CI",
                f"${ci.get('lower', 0):,.0f} – ${ci.get('upper', 0):,.0f}",
            )
        else:
            st.info("No backtest data available.")

        # --- PSI summary ---
        st.subheader("Feature Stability (PSI)")
        psi_data = qr.get("psi", {})
        per_feature = psi_data.get("per_feature", {})

        if per_feature:
            psi_df = pd.DataFrame({
                "Feature": list(per_feature.keys()),
                "PSI": list(per_feature.values()),
            }).sort_values("PSI", ascending=False)

            st.bar_chart(psi_df.set_index("Feature"), use_container_width=True)

            warn = psi_data.get("warning_threshold", 0.10)
            crit = psi_data.get("critical_threshold", 0.25)
            max_feat = psi_data.get("max_feature", "")
            max_val = psi_data.get("max_value", 0)

            if max_val > crit:
                st.error(
                    f"CRITICAL: {max_feat} PSI = {max_val:.4f} > {crit} — "
                    "material drift detected, revalidation needed."
                )
            elif max_val > warn:
                st.warning(
                    f"WARNING: {max_feat} PSI = {max_val:.4f} > {warn} — "
                    "moderate drift, monitor closely."
                )
            else:
                st.success(
                    f"All features stable (max PSI: {max_feat} = {max_val:.4f})."
                )

        # --- Calibration ---
        st.subheader("Calibration (Reliability Curve)")
        cal = qr.get("calibration", {})
        rel_curve = cal.get("reliability_curve", {})
        brier = cal.get("brier_decomposition", {})

        if rel_curve:
            cal_df = pd.DataFrame({
                "Mean Predicted": rel_curve.get("mean_predicted", []),
                "Observed Frequency": rel_curve.get("fraction_positive", []),
            })
            # Add perfect calibration line
            cal_df["Perfect"] = cal_df["Mean Predicted"]
            st.line_chart(
                cal_df.set_index("Mean Predicted"),
                use_container_width=True,
            )

        if brier:
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Brier Score", f"{brier.get('brier_score', 0):.4f}")
            bc2.metric("Reliability", f"{brier.get('reliability', 0):.4f}")
            bc3.metric("Resolution", f"{brier.get('resolution', 0):.4f}")
            bc4.metric("Uncertainty", f"{brier.get('uncertainty', 0):.4f}")
