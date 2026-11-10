#!/usr/bin/env python3
"""
cohens_d.py
============
Cohen's d feature analysis on extreme attractiveness groups.

Splits the dataset into the bottom 20% and top 20% by attractiveness
score, then ranks all features by how much they differ between the two
groups (standardised mean difference) plus single-feature ROC-AUC for
separating the two groups.

Outputs:
  cohens_d_results.csv   — full ranked table
  cohens_d_chart.png     — horizontal bar chart, colour-coded by feature group

Usage:
    python cohens_d.py --features-csv /path/to/ml_features.csv \
        --labels /path/to/labels.txt --out-csv results.csv --out-chart chart.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score

from data_prep import load_labels

QUANTILE = 0.20      # bottom/top 20%
TOP_N_CHART = 40      # features shown in chart

CATEGORY_COLOURS = {
    "depth3d": "#2196F3",
    "surf3d": "#4CAF50",
    "lm2d": "#FF9800",
    "skin": "#9C27B0",
    "age": "#F44336",
    "fem": "#E91E63",
    "shape": "#607D8B",
    "avg": "#795548",
}


def category(col):
    return col.split("__")[0] if "__" in col else "other"


def load_data(features_csv, labels_path):
    df = pd.read_csv(features_csv)
    labels = load_labels(labels_path)
    df = df.merge(labels.rename(columns={"attractiveness": "score"}), on="subject_id", how="inner")
    return df


def select_numeric_features(df):
    drop = {"subject_id", "score",
            "shape__shape_name", "shape__shape_code",
            "age__age_group", "age__gender"}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


def cohens_d(a, b):
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def roc_auc(top_vals, bot_vals):
    y = np.concatenate([np.ones(len(top_vals)), np.zeros(len(bot_vals))])
    score = np.concatenate([top_vals, bot_vals])
    if score.std() == 0:
        return 0.5
    auc = roc_auc_score(y, score)
    return max(auc, 1 - auc)  # flip so AUC reflects "higher = more attractive"


def run(features_csv, labels_path, out_csv, out_chart):
    print("Loading data...")
    df = load_data(features_csv, labels_path)

    lo_cut = df["score"].quantile(QUANTILE)
    hi_cut = df["score"].quantile(1 - QUANTILE)
    bottom = df[df["score"] <= lo_cut]
    top = df[df["score"] >= hi_cut]

    print(f"Bottom {QUANTILE*100:.0f}%: n={len(bottom)}, score <= {lo_cut:.3f}")
    print(f"Top    {QUANTILE*100:.0f}%: n={len(top)}, score >= {hi_cut:.3f}")

    feature_cols = select_numeric_features(df)
    print(f"Analysing {len(feature_cols)} numeric features...")

    rows = []
    for col in feature_cols:
        top_vals = top[col].dropna().values
        bot_vals = bottom[col].dropna().values
        if len(top_vals) < 10 or len(bot_vals) < 10:
            continue
        d = cohens_d(top_vals, bot_vals)
        auc = roc_auc(top_vals, bot_vals)
        rows.append({
            "feature": col, "category": category(col), "cohens_d": d, "abs_d": abs(d),
            "roc_auc": auc, "mean_top": top_vals.mean(), "mean_bot": bot_vals.mean(),
            "direction": "higher in attractive" if d > 0 else "lower in attractive",
        })

    results = pd.DataFrame(rows).sort_values("abs_d", ascending=False).reset_index(drop=True)
    results.index += 1
    results.to_csv(out_csv, index_label="rank")
    print(f"Full results saved to {out_csv}")

    print(f"\n{'Rank':>4}  {'|d|':>5}  {'AUC':>5}  {'Direction':<22}  Feature")
    print("-" * 75)
    for rank, row in results.head(30).iterrows():
        print(f"{rank:>4}  {row['abs_d']:>5.3f}  {row['roc_auc']:>5.3f}  "
              f"{row['direction']:<22}  {row['feature']}")

    plot_df = results.head(TOP_N_CHART).iloc[::-1]
    colours = [CATEGORY_COLOURS.get(r["category"], "#999999") for _, r in plot_df.iterrows()]
    bar_vals = plot_df["cohens_d"].values

    fig, ax = plt.subplots(figsize=(11, TOP_N_CHART * 0.35 + 2))
    ax.barh(range(len(plot_df)), bar_vals, color=colours, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (positive = higher in attractive faces)", fontsize=10)
    ax.set_title(
        f"Feature Separation: Top vs Bottom {QUANTILE*100:.0f}% Attractiveness "
        f"(n={len(top)+len(bottom)} extreme faces)",
        fontsize=11, pad=12,
    )

    for i, (val, auc) in enumerate(zip(bar_vals, plot_df["roc_auc"].values)):
        x_pos = val + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, i, f"|d|={abs(val):.2f}  AUC={auc:.2f}", va="center", ha=ha,
                fontsize=6.5, color="#333333")

    for x in [-0.8, -0.5, 0.5, 0.8]:
        ax.axvline(x, color="grey", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.text(0.505, 0.01, "medium (0.5)", transform=ax.transAxes, fontsize=7, color="grey", va="bottom")
    ax.text(0.82, 0.01, "large (0.8)", transform=ax.transAxes, fontsize=7, color="grey", va="bottom")

    legend_patches = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLOURS.items()
                      if cat in plot_df["category"].values]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, title="Category", title_fontsize=8)

    plt.tight_layout()
    plt.savefig(out_chart, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out_chart}")
    return out_csv, out_chart


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-csv", default="cohens_d_results.csv")
    ap.add_argument("--out-chart", default="cohens_d_chart.png")
    args = ap.parse_args()
    run(args.features_csv, args.labels, args.out_csv, args.out_chart)


if __name__ == "__main__":
    main()
