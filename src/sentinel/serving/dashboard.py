"""Streamlit dashboard with interactive live scoring + read-only analytics tabs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .. import config
from .live import (
    APP_ASSETS,
    EXAMPLES,
    heuristic_score,
    load_scorer,
    score_frame,
    validate_columns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEMO_DIR = config.REPORTS_DIR / "demo"


def _report_path(name: str) -> Path | None:
    """Return the live report if it exists, else the committed demo copy."""
    live = config.REPORTS_DIR / name
    if live.exists():
        return live
    demo = DEMO_DIR / name
    if demo.exists():
        return demo
    return None


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------


def render() -> None:  # noqa: C901 — complex but single dashboard entry point
    """Render the full Streamlit dashboard."""
    st.set_page_config(page_title="Sentinel — Fraud Detection", layout="wide")
    st.title("Sentinel — Cost-Sensitive Fraud Detection")

    tab_live, tab_exec, tab_analyst, tab_risk = st.tabs(
        ["Try it live", "Executive", "Analyst", "Risk"]
    )

    # Load model once (cached across reruns)
    model, fb = _get_scorer()

    # ===================================================================
    # TAB: Try it live
    # ===================================================================
    with tab_live:
        _render_live_tab(model, fb)

    # ===================================================================
    # TAB: Executive
    # ===================================================================
    with tab_exec:
        _render_executive_tab()

    # ===================================================================
    # TAB: Analyst (copilot investigation)
    # ===================================================================
    with tab_analyst:
        _render_analyst_tab(model, fb)

    # ===================================================================
    # TAB: Risk
    # ===================================================================
    with tab_risk:
        _render_risk_tab()


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_scorer():
    return load_scorer()


# ---------------------------------------------------------------------------
# Try it live tab
# ---------------------------------------------------------------------------


def _render_live_tab(model, fb) -> None:
    st.subheader("Interactive Fraud Scoring")

    if model is None:
        st.info(
            "Using heuristic scorer (model assets not found). "
            "The app will still score transactions based on amount + time signals."
        )

    mode = st.radio(
        "Mode",
        ["Investigate a transaction", "Score a CSV"],
        horizontal=True,
    )

    if mode == "Investigate a transaction":
        _render_investigate_mode(model, fb)
    else:
        _render_csv_mode(model, fb)


def _render_investigate_mode(model, fb) -> None:
    st.caption(
        "Pick an example or edit the fields. The agentic copilot scores the "
        "transaction, retrieves relevant fraud policy, and drafts a cited "
        "disposition with a full decision trace."
    )

    # Example selector
    example_names = [ex["name"] for ex in EXAMPLES]
    selected = st.selectbox("Example transaction", example_names)
    example_txn = next(ex["txn"] for ex in EXAMPLES if ex["name"] == selected)

    # Editable fields
    col1, col2, col3 = st.columns(3)
    with col1:
        amt = st.number_input("Amount ($)", value=float(example_txn["amt"]), min_value=0.01)
    with col2:
        category = st.selectbox(
            "Category",
            ["grocery_pos", "shopping_net", "shopping_pos", "gas_transport",
             "entertainment", "food_dining", "misc_net", "misc_pos", "travel"],
            index=["grocery_pos", "shopping_net", "shopping_pos", "gas_transport",
                   "entertainment", "food_dining", "misc_net", "misc_pos",
                   "travel"].index(example_txn.get("category", "shopping_net")),
        )
    with col3:
        far_merchant = st.toggle("Merchant far from home", value=(
            abs(example_txn.get("merch_lat", 0) - example_txn.get("lat", 0)) > 5
        ))

    # Build transaction dict
    txn = dict(example_txn)
    txn["amt"] = amt
    txn["category"] = category
    if far_merchant:
        txn["merch_lat"] = txn["lat"] + 18.0  # ~2000km away
        txn["merch_long"] = txn["long"] + 15.0
    else:
        txn["merch_lat"] = txn["lat"] + 0.01
        txn["merch_long"] = txn["long"] + 0.01

    if st.button("Score & Investigate", type="primary"):
        # Build score function
        if model is not None and fb is not None:
            def score_fn(t: dict) -> float:
                X, _, _ = fb.transform(pd.DataFrame([t]))
                return float(model.predict_proba(X)[:, 1][0])
        else:
            score_fn = heuristic_score

        # Run the full agentic investigation
        from ..copilot.investigate import investigate
        from ..copilot.retriever import get_retriever

        with st.spinner("Running agentic investigation..."):
            retriever = get_retriever()
            inv = investigate(txn, score_fn, history_df=None, retriever=retriever)

        st.divider()

        # Headline
        rec_colors = {"escalate": "red", "clear": "green", "request_info": "orange"}
        color = rec_colors.get(inv.recommendation, "gray")
        st.markdown(
            f"### :{color}[**{inv.recommendation.upper()}**] "
            f"(confidence {inv.confidence:.0%})"
        )
        st.metric("Fraud Probability", f"{inv.probability:.4f}")

        # Rationale
        st.subheader("Rationale")
        st.write(inv.rationale)

        # Citations
        st.subheader("Policy Citations")
        if inv.citations:
            for cite in inv.citations:
                st.code(cite, language=None)
        else:
            st.caption("No matching policy citations.")

        # Decision trace (each tool call)
        st.subheader("Decision Trace")
        trace_df = pd.DataFrame([
            {"Step": i + 1, "Tool": s.tool, "Output": s.output}
            for i, s in enumerate(inv.trace)
        ])
        st.table(trace_df)

        with st.expander("Full investigation JSON"):
            st.json(inv.to_dict())


def _render_csv_mode(model, fb) -> None:
    st.caption(
        "Upload a CSV of transactions to score. The model assigns fraud "
        "probabilities and flags alerts above the threshold you set."
    )

    # Template + demo downloads
    col_dl1, col_dl2 = st.columns(2)
    template_path = APP_ASSETS / "upload_template.csv"
    if template_path.exists():
        with col_dl1:
            st.download_button(
                "Download the CSV template",
                data=template_path.read_bytes(),
                file_name="upload_template.csv",
                mime="text/csv",
            )

    demo_path = APP_ASSETS / "demo_upload.csv"
    use_demo = False
    if demo_path.exists():
        with col_dl2:
            use_demo = st.button(
                "Score a bundled sample (300 real transactions)",
            )

    uploaded = st.file_uploader(
        "Upload transactions CSV *(max file size: 5 GB)*", type=["csv"]
    )

    # Determine data source
    if use_demo and demo_path.exists():
        df = pd.read_csv(demo_path)
        n_fraud = int(df.get("is_fraud", pd.Series([0])).sum())
        st.write(f"Loaded bundled sample: **{len(df):,}** rows ({n_fraud} known fraud)")
    elif uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Uploaded **{len(df):,}** rows")
    else:
        return

    # Validate schema
    missing = validate_columns(df)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.info("Download the template above to see the expected schema.")
        return

    # Limit to 50k rows
    if len(df) > 50_000:
        st.warning("Limiting to first 50,000 rows for performance.")
        df = df.head(50_000)

    # Threshold slider
    threshold = st.slider(
        "Alert threshold", 0.01, 0.99, 0.30,
        help="Transactions scoring above this are flagged as potential fraud.",
    )

    # Score
    if model is not None and fb is not None:
        with st.spinner(f"Scoring {len(df):,} transactions..."):
            probs = score_frame(df, model, fb)
        source = "XGBoost model"
    else:
        probs = np.array([heuristic_score(row.to_dict()) for _, row in df.iterrows()])
        source = "heuristic (no model loaded)"

    df = df.copy()
    df["fraud_probability"] = probs
    df["flagged"] = probs >= threshold

    # Summary metrics
    n_flagged = int(df["flagged"].sum())
    dollars_at_risk = float(df.loc[df["flagged"], "amt"].sum()) if "amt" in df else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions scored", f"{len(df):,}")
    c2.metric("Flagged as fraud", f"{n_flagged:,}")
    c3.metric("Total $ flagged", f"${dollars_at_risk:,.0f}")
    st.caption(f"Scored with: {source} | Threshold: {threshold:.2f}")

    # Top flagged rows
    st.subheader("Top flagged transactions")
    top = df.sort_values("fraud_probability", ascending=False).head(20)
    display_cols = ["trans_num", "category", "amt", "fraud_probability", "flagged"]
    display_cols = [c for c in display_cols if c in top.columns]
    st.dataframe(top[display_cols], use_container_width=True)

    # Download scored CSV
    csv_out = df.to_csv(index=False).encode()
    st.download_button(
        "Download scored CSV",
        data=csv_out,
        file_name="scored_transactions.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Executive tab (read-only, from reports)
# ---------------------------------------------------------------------------


def _render_executive_tab() -> None:
    path = _report_path("evaluation.json")
    if not path:
        st.warning("No evaluation found. Run `make train && make evaluate` first.")
        return

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
    if "cost_curve" in rep:
        curve = pd.DataFrame(rep["cost_curve"]).set_index("threshold")
        st.line_chart(curve)

    st.caption(
        f"Dataset: {rep.get('dataset', 'N/A')} | "
        f"fraud rate {rep.get('fraud_rate', 0):.3%} | "
        f"alert budget {rep.get('alert_budget_frac', 0.005):.1%}"
    )


# ---------------------------------------------------------------------------
# Analyst tab (copilot — preserved from existing)
# ---------------------------------------------------------------------------


def _render_analyst_tab(model, fb) -> None:
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
        help="Provide at minimum: trans_num, category, amt.",
    )

    if st.button("Investigate", type="primary", key="analyst_btn"):
        try:
            txn = json.loads(txn_input)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return

        if "amt" not in txn:
            st.error("Transaction must include an 'amt' field.")
            return

        from ..copilot.investigate import investigate
        from ..copilot.retriever import get_retriever

        if model is not None and fb is not None:
            def score_fn(t: dict) -> float:
                X, _, _ = fb.transform(pd.DataFrame([t]))
                return float(model.predict_proba(X)[:, 1][0])
        else:
            score_fn = heuristic_score

        with st.spinner("Running investigation workflow..."):
            retriever = get_retriever()
            inv = investigate(txn, score_fn, history_df=None, retriever=retriever)

        st.divider()
        rec_colors = {"escalate": "red", "clear": "green", "request_info": "orange"}
        color = rec_colors.get(inv.recommendation, "gray")
        st.markdown(
            f"### :{color}[**{inv.recommendation.upper()}**] "
            f"(confidence {inv.confidence:.0%})"
        )
        st.metric("Fraud Probability", f"{inv.probability:.3f}")

        st.subheader("Rationale")
        st.write(inv.rationale)

        st.subheader("Policy Citations")
        if inv.citations:
            for cite in inv.citations:
                st.code(cite, language=None)
        else:
            st.write("No policy citations available.")

        st.subheader("Decision Trace")
        trace_data = [
            {"Step": i + 1, "Tool": step.tool, "Output": step.output}
            for i, step in enumerate(inv.trace)
        ]
        st.table(pd.DataFrame(trace_data))

        with st.expander("Full investigation JSON"):
            st.json(inv.to_dict())


# ---------------------------------------------------------------------------
# Risk tab (read-only, from reports)
# ---------------------------------------------------------------------------


def _render_risk_tab() -> None:
    st.subheader("Quantitative Risk Analytics")

    qr_path = _report_path("quant_risk.json")
    if not qr_path:
        st.warning("No quant risk report. Run `make quant-risk` first.")
        return

    qr = json.loads(qr_path.read_text())
    risk_data = qr.get("loss_risk_var_es", {})
    boot = risk_data.get("bootstrap", {})
    mc = risk_data.get("monte_carlo", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("95% VaR (bootstrap)", f"${boot.get('var', 0):,.0f}")
    c2.metric("95% ES (bootstrap)", f"${boot.get('expected_shortfall', 0):,.0f}")
    c3.metric("95% VaR (Monte Carlo)", f"${mc.get('var', 0):,.0f}")
    c4.metric("95% ES (Monte Carlo)", f"${mc.get('expected_shortfall', 0):,.0f}")

    # Backtest
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
        sc4.metric("95% CI", f"${ci.get('lower', 0):,.0f} – ${ci.get('upper', 0):,.0f}")

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
