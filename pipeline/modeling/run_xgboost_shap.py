"""
run_xgboost_shap.py
===================
XGBoost regression with SHAP explanations for attractiveness prediction.

Produces:
  - shap_feature_importance.csv   mean |SHAP| per feature
  - metrics.txt                   CV performance summary
  - predicted_vs_actual.png       scatter plot
  - shap_summary.png              SHAP beeswarm plot (top 30)
  - shap_bar.png                  SHAP global importance bar chart
  - shap_dep_<feature>.png        dependence plots for top 6 features
  - residuals.png                 residual plot

Usage:
    python run_xgboost_shap.py
    python run_xgboost_shap.py --features-csv /path/to/ml_features.csv
    python run_xgboost_shap.py --feature-set all --tune
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_predict, KFold, GridSearchCV, RandomizedSearchCV,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

import xgboost as xgb
import shap

from data_prep import load_and_prepare, add_common_args


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost + SHAP attractiveness analysis"
    )
    add_common_args(parser)
    parser.add_argument(
        "--cv-folds", type=int, default=10,
        help="Number of CV folds (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: xgboost_shap/output/)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run randomised hyperparameter search (slower but may improve)",
    )
    args = parser.parse_args()

    # ── Output directory ─────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    X, y, feature_names, _ = load_and_prepare(
        features_csv=args.features_csv,
        labels_path=args.labels,
        feature_set=args.feature_set,
        exclude_prefixes=args.exclude_prefix,
    )

    # Standardise (not required for trees but keeps SHAP values comparable
    # with the ElasticNet analysis and makes dependence plots cleaner)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    # ── Hyperparameter tuning or sensible defaults ───────────────────────
    if args.tune:
        print("\n== Randomised hyperparameter search ==")
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda": [1.0, 3.0, 5.0, 10.0],
            "min_child_weight": [3, 5, 7, 10],
        }
        base = xgb.XGBRegressor(
            random_state=42, tree_method="hist", n_jobs=-1,
        )
        search = RandomizedSearchCV(
            base, param_dist,
            n_iter=80, cv=cv, scoring="r2",
            n_jobs=-1, verbose=1, random_state=42, refit=True,
        )
        search.fit(X_scaled, y)
        model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
        print(f"Best CV R2:  {search.best_score_:.4f}")
    else:
        # Conservative defaults — good for n~2000, p~77
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=5.0,
            min_child_weight=5,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        )

    # ── Cross-validated predictions ──────────────────────────────────────
    print("\n== Cross-validated predictions ==")
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv)

    r2 = r2_score(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    pc, _ = pearsonr(y, y_pred_cv)

    print(f"\n== {args.cv_folds}-Fold CV Performance ==")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  PC:   {pc:.4f}")

    # ── Train final model on all data for SHAP analysis ──────────────────
    print("\n== Training final model for SHAP analysis ==")
    model.fit(X_scaled, y)

    # ── SHAP values ──────────────────────────────────────────────────────
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
    except (ValueError, TypeError) as e:
        print(f"  TreeExplainer failed ({e}), falling back to Explainer")
        explainer = shap.Explainer(model.predict, X_scaled,
                                   feature_names=feature_names)
        shap_obj = explainer(X_scaled)
        shap_values = shap_obj.values

    # Global importance: mean |SHAP|
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": shap_importance,
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\n== Top 25 Features (by mean |SHAP value|) ==")
    for _, row in importance_df.head(25).iterrows():
        print(f"  {row['mean_abs_shap']:.4f}  {row['feature']}")

    # ── Save CSV & metrics ───────────────────────────────────────────────
    importance_df.to_csv(out_dir / "shap_feature_importance.csv", index=False)

    params_str = ", ".join(
        f"{k}={v}" for k, v in model.get_params().items()
        if k not in ("n_jobs", "random_state", "tree_method", "verbosity")
    )
    with open(out_dir / "metrics.txt", "w") as f:
        f.write("XGBoost + SHAP Attractiveness Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Feature set: {args.feature_set}\n")
        f.write(f"Samples:     {len(y)}\n")
        f.write(f"Features:    {len(feature_names)}\n")
        f.write(f"Model:       {params_str}\n")
        f.write(f"\n{args.cv_folds}-Fold CV Performance\n")
        f.write("-" * 30 + "\n")
        f.write(f"R2:   {r2:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"PC:   {pc:.4f}\n")

    # ── Plot 1: SHAP beeswarm (summary) ──────────────────────────────────
    plt.figure(figsize=(10, max(8, min(30, len(feature_names)) * 0.25)))
    shap.summary_plot(
        shap_values, X_scaled, feature_names=feature_names,
        show=False, max_display=30,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: SHAP bar (global importance) ─────────────────────────────
    plt.figure(figsize=(10, max(6, min(30, len(feature_names)) * 0.25)))
    shap.summary_plot(
        shap_values, X_scaled, feature_names=feature_names,
        plot_type="bar", show=False, max_display=30,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Predicted vs Actual ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y, y_pred_cv, alpha=0.5, s=20, edgecolors="k", linewidth=0.3)
    lims = [
        min(y.min(), y_pred_cv.min()) - 0.2,
        max(y.max(), y_pred_cv.max()) + 0.2,
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Attractiveness", fontsize=12)
    ax.set_ylabel("Predicted Attractiveness (CV)", fontsize=12)
    ax.set_title(
        f"XGBoost: Predicted vs Actual  (R2={r2:.3f}, MAE={mae:.3f})",
        fontsize=13,
    )
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: SHAP dependence plots for top 6 features ────────────────
    top6 = importance_df.head(6)["feature"].tolist()
    for feat in top6:
        feat_idx = feature_names.index(feat)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feat_idx, shap_values, X_scaled,
            feature_names=feature_names, show=False, ax=ax,
        )
        fig.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"shap_dep_{safe_name}.png", dpi=150)
        plt.close(fig)

    # ── Plot 5: Residuals ────────────────────────────────────────────────
    residuals = y - y_pred_cv
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred_cv, residuals, alpha=0.5, s=20,
               edgecolors="k", linewidth=0.3)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Attractiveness", fontsize=12)
    ax.set_ylabel("Residual", fontsize=12)
    ax.set_title("XGBoost: Residual Plot", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n== Output saved to {out_dir} ==")
    print("  shap_feature_importance.csv")
    print("  metrics.txt")
    print("  shap_summary.png        (beeswarm)")
    print("  shap_bar.png            (global importance)")
    print("  predicted_vs_actual.png")
    print(f"  shap_dep_*.png          (top {len(top6)} features)")
    print("  residuals.png")


if __name__ == "__main__":
    main()
