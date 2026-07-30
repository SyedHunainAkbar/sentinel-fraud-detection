"""Generate SHAP global and local explanations for the Sentinel tree model.

Usage:
    python .kiro/skills/shap-explainability/scripts/generate_shap.py

Outputs:
    reports/shap_summary.json   — feature importances + local explanations
    reports/shap_global_bar.png — global mean |SHAP| bar chart
    reports/shap_beeswarm.png   — beeswarm plot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap

# Ensure we can run headless
matplotlib.use("Agg")

# Resolve project root (works whether called from repo root or scripts dir)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from sentinel import config, ingest  # noqa: E402
from sentinel.features import FeatureBuilder  # noqa: E402

# --- Configuration ---
MAX_SAMPLES = 5000  # cap test set for SHAP computation speed
TOP_N_LOCAL = 5     # number of top-risk transactions to explain locally


def main() -> None:
    """Run SHAP analysis and persist outputs."""
    artifact_path = config.MODELS_DIR / "artifacts.joblib"
    if not artifact_path.exists():
        print(f"ERROR: {artifact_path} not found. Run `make train` first.", file=sys.stderr)
        sys.exit(1)

    print("[shap] Loading artifacts...")
    artifacts = joblib.load(artifact_path)

    # Load the full pipeline and extract components
    model_pipeline = artifacts["models"]["xgboost"]
    feature_builder: FeatureBuilder = artifacts["feature_builder"]

    # Extract tree model and preprocessor from pipeline
    if hasattr(model_pipeline, "named_steps"):
        tree_model = model_pipeline.named_steps["clf"]
        preprocessor = model_pipeline.named_steps["pre"]
    else:
        tree_model = model_pipeline
        preprocessor = None

    # Reload and transform test data to get full feature DataFrame
    print("[shap] Loading and transforming test data...")
    df = ingest.load_transactions()
    _, test_df = ingest.temporal_split(df)
    X_test_df, y_test, _ = feature_builder.transform(test_df)

    # Subsample for speed
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = len(X_test_df)
    if n > MAX_SAMPLES:
        idx = rng.choice(n, size=MAX_SAMPLES, replace=False)
        X_sample_df = X_test_df.iloc[idx].reset_index(drop=True)
        y_sample = y_test.iloc[idx].to_numpy() if hasattr(y_test, "iloc") else y_test[idx]
        print(f"[shap] Subsampled {MAX_SAMPLES} from {n} test rows.")
    else:
        X_sample_df = X_test_df
        y_sample = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

    # Transform through the ColumnTransformer to get what the tree actually sees
    if preprocessor is not None:
        X_transformed = preprocessor.transform(X_sample_df)
        all_feature_names = list(preprocessor.get_feature_names_out())
    else:
        X_transformed = X_sample_df.to_numpy()
        all_feature_names = list(X_sample_df.columns)

    # Ensure dense array
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed, dtype=np.float32)

    # --- SHAP values ---
    print("[shap] Computing TreeExplainer SHAP values...")
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class (fraud)

    # --- Global importance ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    n_feats = len(all_feature_names)
    importance_order = np.argsort(mean_abs_shap)[::-1]

    global_importance = [
        {"feature": all_feature_names[i], "mean_abs_shap": round(float(mean_abs_shap[i]), 6)}
        for i in importance_order
    ]

    # For plots and local explanations, focus on the top features for readability
    # Use only numeric features (first 9) for the focused beeswarm
    numeric_idx = list(range(len(config.NUMERIC_FEATURES)))
    numeric_names = [all_feature_names[i] for i in numeric_idx]

    # --- Global bar plot (top 15 features) ---
    print("[shap] Generating global bar plot...")
    top_n_plot = min(15, n_feats)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_order = importance_order[:top_n_plot]
    y_pos = np.arange(top_n_plot)
    sorted_vals = mean_abs_shap[plot_order]
    sorted_names = [all_feature_names[i] for i in plot_order]
    ax.barh(y_pos, sorted_vals, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP) — Top 15")
    fig.tight_layout()
    bar_path = config.REPORTS_DIR / "shap_global_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"[shap] Saved {bar_path}")

    # --- Beeswarm plot (numeric features only for clarity) ---
    print("[shap] Generating beeswarm plot...")
    shap.summary_plot(
        shap_values[:, numeric_idx],
        X_transformed[:, numeric_idx],
        feature_names=numeric_names,
        show=False,
    )
    beeswarm_path = config.REPORTS_DIR / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[shap] Saved {beeswarm_path}")

    # --- Local explanations (top-N highest risk) ---
    print(f"[shap] Generating top-{TOP_N_LOCAL} local explanations...")
    probs = tree_model.predict_proba(X_transformed)[:, 1]
    top_idx = np.argsort(probs)[::-1][:TOP_N_LOCAL]

    # For local explanations, report only numeric feature contributions (most interpretable)
    local_explanations = []
    for rank, i in enumerate(top_idx, 1):
        contributions = {
            numeric_names[j]: round(float(shap_values[i, numeric_idx[j]]), 6)
            for j in range(len(numeric_idx))
        }
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val = base_val[1]
        local_explanations.append({
            "rank": rank,
            "sample_index": int(i),
            "predicted_probability": round(float(probs[i]), 6),
            "actual_label": int(y_sample[i]),
            "base_value": round(float(base_val), 6),
            "shap_contributions": contributions,
        })

    # --- Write JSON summary ---
    summary = {
        "model": "xgboost",
        "n_samples": int(len(X_sample_df)),
        "n_features_total": n_feats,
        "feature_names_numeric": list(config.NUMERIC_FEATURES),
        "global_importance": global_importance,
        "local_explanations": local_explanations,
    }
    json_path = config.REPORTS_DIR / "shap_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[shap] Saved {json_path}")

    # --- Summary to stdout ---
    print("\n[shap] Top features by mean |SHAP|:")
    for item in global_importance[:5]:
        print(f"  {item['feature']:25s}  {item['mean_abs_shap']:.4f}")
    print(f"\n[shap] Done. Outputs in {config.REPORTS_DIR}/")


if __name__ == "__main__":
    main()
