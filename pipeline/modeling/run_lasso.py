"""
run_lasso.py
============
Lasso regression for explainable attractiveness prediction.

Lasso (L1 penalty only) tends to produce sparser models than ElasticNet,
driving more coefficients to exactly zero and selecting a smaller set of
the most predictive features.

Produces:
  - coefficients.csv        sorted feature coefficients
  - metrics.txt             CV performance summary
  - predicted_vs_actual.png scatter plot
  - top_coefficients.png    horizontal bar chart of non-zero coefficients
  - residuals.png           residual plot

Usage:
    python run_lasso.py
    python run_lasso.py --features-csv /path/to/ml_features.csv
    python run_lasso.py --feature-set all --cv-folds 5
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from data_prep import load_and_prepare, add_common_args


def main():
    parser = argparse.ArgumentParser(
        description="Lasso attractiveness analysis"
    )
    add_common_args(parser)
    parser.add_argument(
        "--cv-folds", type=int, default=10,
        help="Number of CV folds (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: lasso/output/)",
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

    # ── Standardise (essential for Lasso) ─────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── LassoCV — finds optimal alpha ─────────────────────────────────
    print("\n== Training LassoCV ==")
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    model = LassoCV(
        alphas=100,
        cv=cv,
        max_iter=10000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    print(f"Best alpha: {model.alpha_:.6f}")

    # ── Cross-validated predictions ──────────────────────────────────────
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

    # ── Coefficients ─────────────────────────────────────────────────────
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
    }).sort_values("abs_coefficient", ascending=False)

    n_nonzero = (coef_df["coefficient"] != 0).sum()
    n_total = len(coef_df)
    print(f"\nNon-zero coefficients: {n_nonzero} / {n_total}")

    top = coef_df[coef_df["coefficient"] != 0].head(30)
    print(f"\n== Top {len(top)} Features (by |coefficient|) ==")
    for _, row in top.iterrows():
        sign = "+" if row["coefficient"] > 0 else "-"
        print(f"  {sign} {row['abs_coefficient']:.4f}  {row['feature']}")

    # ── Save CSV & metrics ───────────────────────────────────────────────
    coef_df.to_csv(out_dir / "coefficients.csv", index=False)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write("Lasso Attractiveness Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Feature set:           {args.feature_set}\n")
        f.write(f"Samples:               {len(y)}\n")
        f.write(f"Features:              {len(feature_names)}\n")
        f.write(f"Non-zero coefficients: {n_nonzero}\n")
        f.write(f"Best alpha:            {model.alpha_:.6f}\n")
        f.write(f"\n{args.cv_folds}-Fold CV Performance\n")
        f.write("-" * 30 + "\n")
        f.write(f"R2:   {r2:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"PC:   {pc:.4f}\n")

    # ── Plot 1: Predicted vs Actual ──────────────────────────────────────
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
        f"Lasso: Predicted vs Actual  (R2={r2:.3f}, MAE={mae:.3f})",
        fontsize=13,
    )
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Top coefficients bar chart ───────────────────────────────
    top25 = coef_df[coef_df["coefficient"] != 0].head(25)
    if len(top25) > 0:
        fig, ax = plt.subplots(
            figsize=(10, max(6, len(top25) * 0.35))
        )
        colors = [
            "#2ecc71" if c > 0 else "#e74c3c"
            for c in top25["coefficient"]
        ]
        ax.barh(range(len(top25)), top25["coefficient"].values, color=colors)
        ax.set_yticks(range(len(top25)))
        ax.set_yticklabels(top25["feature"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient (standardised features)", fontsize=11)
        ax.set_title("Lasso: Top Feature Coefficients", fontsize=13)
        ax.axvline(0, color="k", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "top_coefficients.png", dpi=150)
        plt.close(fig)

    # ── Plot 3: Residuals ────────────────────────────────────────────────
    residuals = y - y_pred_cv
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred_cv, residuals, alpha=0.5, s=20,
               edgecolors="k", linewidth=0.3)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Attractiveness", fontsize=12)
    ax.set_ylabel("Residual", fontsize=12)
    ax.set_title("Lasso: Residual Plot", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n== Output saved to {out_dir} ==")
    print("  coefficients.csv")
    print("  metrics.txt")
    print("  predicted_vs_actual.png")
    print("  top_coefficients.png")
    print("  residuals.png")


if __name__ == "__main__":
    main()
