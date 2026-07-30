"""XGBoost hyperparameter search optimizing PR-AUC.

Uses RandomizedSearchCV with stratified k-fold, scoring on average_precision (PR-AUC).
Persists the best parameters and retrained model to models/.

Usage:
    python -m sentinel.hyperparam            # runs on full dataset
    SENTINEL_DATA=data/sample/... python -m sentinel.hyperparam  # sample
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from . import config
from .models import _preprocessor

# Search space — covers the meaningful XGBoost knobs
PARAM_DISTRIBUTIONS: dict[str, Any] = {
    "clf__n_estimators": randint(100, 800),
    "clf__max_depth": randint(3, 10),
    "clf__learning_rate": uniform(0.01, 0.29),  # [0.01, 0.30]
    "clf__subsample": uniform(0.6, 0.4),         # [0.6, 1.0]
    "clf__colsample_bytree": uniform(0.5, 0.5),  # [0.5, 1.0]
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0, 5),
    "clf__reg_alpha": uniform(0, 2),
    "clf__reg_lambda": uniform(0.5, 4.5),        # [0.5, 5.0]
}


def hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 50,
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
) -> dict:
    """Run randomized hyperparameter search for XGBoost.

    Parameters
    ----------
    X : DataFrame
        Training features (raw — preprocessing is part of the pipeline).
    y : Series
        Binary labels.
    n_iter : int
        Number of random parameter combinations to try.
    cv : int
        Number of stratified cross-validation folds.
    n_jobs : int
        Parallelism for the search (-1 = all cores).
    verbose : int
        Verbosity level for sklearn.

    Returns
    -------
    dict
        Keys: best_params, best_score (PR-AUC), cv_results_summary, n_iter.
    """
    from sklearn.pipeline import Pipeline

    pos = max(int(y.sum()), 1)
    neg = len(y) - pos

    pipe = Pipeline([
        ("pre", _preprocessor()),
        ("clf", XGBClassifier(
            scale_pos_weight=neg / pos,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=config.RANDOM_SEED,
            use_label_encoder=False,
        )),
    ])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="average_precision",  # PR-AUC
        cv=skf,
        random_state=config.RANDOM_SEED,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )

    search.fit(X, y)

    # Extract clean params (strip "clf__" prefix for readability)
    best_params_clean = {
        k.replace("clf__", ""): v for k, v in search.best_params_.items()
    }

    # Top-5 results summary
    cv_df = pd.DataFrame(search.cv_results_)
    top5 = (
        cv_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        .sort_values("rank_test_score")
        .head(5)
    )

    return {
        "best_params": best_params_clean,
        "best_params_pipeline": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
        "top5": top5.to_dict(orient="records"),
        "n_iter": n_iter,
        "cv_folds": cv,
    }


def main() -> None:
    """Run hyperparameter search and persist results."""
    import joblib

    from . import ingest
    from .features import FeatureBuilder

    print("Loading data...")
    df = ingest.load_transactions()
    train_df, _ = ingest.temporal_split(df)

    print("Building features...")
    fb = FeatureBuilder().fit(train_df)
    X_train, y_train, _ = fb.fit_transform(train_df)

    # Use fewer iterations on small datasets
    n_iter = 50 if len(X_train) > 5000 else 10
    cv_folds = 3 if len(X_train) > 1000 else 2

    print(f"Starting hyperparameter search (n_iter={n_iter}, cv={cv_folds})...")
    result = hyperparameter_search(
        X_train, y_train, n_iter=n_iter, cv=cv_folds
    )

    print(f"\nBest PR-AUC (CV): {result['best_score']:.4f}")
    print(f"Best parameters: {result['best_params']}")

    # Save best model
    best_model = result["best_estimator"]
    model_out = config.MODELS_DIR / "xgboost_tuned.joblib"
    joblib.dump(best_model, model_out)
    print(f"Saved tuned model to {model_out}")

    # Save search results (without the sklearn estimator)
    report = {
        "best_params": result["best_params"],
        "best_pr_auc_cv": result["best_score"],
        "n_iter": result["n_iter"],
        "cv_folds": result["cv_folds"],
        "top5_configurations": [
            {
                "rank": r["rank_test_score"],
                "mean_pr_auc": round(r["mean_test_score"], 5),
                "std_pr_auc": round(r["std_test_score"], 5),
            }
            for r in result["top5"]
        ],
    }
    report_out = config.REPORTS_DIR / "hyperparam_search.json"
    report_out.write_text(json.dumps(report, indent=2))
    print(f"Saved search report to {report_out}")


if __name__ == "__main__":
    main()
