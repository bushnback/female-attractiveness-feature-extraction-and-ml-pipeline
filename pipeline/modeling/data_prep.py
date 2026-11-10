"""
data_prep.py
============
Shared data loading and preparation for attractiveness modelling.
Loads ml_features.csv, merges with a labels file, and returns
X, y, feature_names ready for modelling.

Used by run_lasso.py, run_elasticnet.py, and run_xgboost_shap.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Feature group prefixes (matches the `aggregate.py` column prefixes) ──────

FEATURE_GROUPS = {
    # Pure 2D landmark geometry
    "2d": ["lm2d__"],
    # 3D mesh-derived geometry
    "3d": ["shape__", "depth3d__", "surf3d__"],
    # Holistic appearance features (CLIP-based and ML-estimated)
    "appearance": ["skin__", "age__", "fem_clip__", "neoteny_clip__", "adiposity__",
                   "glasses__", "tattoo__", "hair_color__", "hair_length__"],
    # Composite / derived scores
    "composite": ["fem__", "avg__"],
}

# Columns to always exclude (non-numeric identifiers, quality flags,
# or redundant encodings)
EXCLUDE_COLS = {
    "subject_id",
    "shape__shape_name",       # string label — use shape__shape_code instead
    "shape__shape_code",       # coarse integer — less useful than continuous 3D
    "age__age_group",
    "hair_color__hair_color",  # categorical label — probabilities used instead
    "tattoo__tattoo_label",    # categorical label — scores used instead
    "tattoo__piercing_label",
}


def load_labels(labels_path):
    """
    Load a labels file into a DataFrame with columns [subject_id, attractiveness].

    Accepts either:
      - a 2-column CSV with a header (any column names; first two columns used)
      - a whitespace-separated "filename score" txt file, one pair per line
    """
    labels_path = Path(labels_path)
    if labels_path.suffix.lower() == ".csv":
        labels = pd.read_csv(labels_path)
        labels.columns = ["filename", "attractiveness"] + list(labels.columns[2:])
    else:
        labels = pd.read_csv(labels_path, sep=r"\s+", header=None,
                              names=["filename", "attractiveness"])
    labels["subject_id"] = labels["filename"].astype(str).str.rsplit(".", n=1).str[0]
    return labels[["subject_id", "attractiveness"]]


def load_and_prepare(features_csv, labels_path, feature_set="all",
                     drop_missing=True, exclude_prefixes=None):
    """Load features + labels, merge, and return modelling-ready arrays.

    Parameters
    ----------
    features_csv : str or Path — path to ml_features.csv (from aggregate.py)
    labels_path : str or Path — path to a labels file (see load_labels)
    feature_set : str
        "all"                  — every numeric feature (minus EXCLUDE_COLS)
        "2d"                   — 2D landmark features only
        "3d"                   — 3D mesh-derived features only
        "appearance"           — CLIP/vision-model features only
        "geometric"            — 2d + 3d
        "2d+composite"         — 2D + femininity/averageness composites
        "geometric+appearance" — 2d + 3d + appearance
    drop_missing : bool — if True, drop rows with any NaN in selected columns

    Returns
    -------
    X, y, feature_names, df_merged
    """
    features_csv = Path(features_csv)
    df = pd.read_csv(features_csv)
    print(f"Loaded features: {df.shape[0]} subjects x {df.shape[1]} columns")

    labels = load_labels(labels_path)
    df_merged = df.merge(labels, on="subject_id", how="inner")
    print(f"After merging with labels: {df_merged.shape[0]} subjects")

    if feature_set == "all":
        feature_cols = [
            c for c in df_merged.columns
            if c not in ("subject_id", "attractiveness") and c not in EXCLUDE_COLS
        ]
    else:
        groups = {
            "2d": FEATURE_GROUPS["2d"],
            "3d": FEATURE_GROUPS["3d"],
            "appearance": FEATURE_GROUPS["appearance"],
            "geometric": FEATURE_GROUPS["2d"] + FEATURE_GROUPS["3d"],
            "2d+composite": FEATURE_GROUPS["2d"] + FEATURE_GROUPS["composite"],
            "geometric+appearance": (FEATURE_GROUPS["2d"] + FEATURE_GROUPS["3d"]
                                      + FEATURE_GROUPS["appearance"]),
        }
        if feature_set not in groups:
            raise ValueError(
                f"Unknown feature_set: '{feature_set}'. Valid options: all, "
                f"{', '.join(groups.keys())}"
            )
        prefixes = groups[feature_set]
        feature_cols = [
            c for c in df_merged.columns
            if any(c.startswith(p) for p in prefixes) and c not in EXCLUDE_COLS
        ]

    if exclude_prefixes:
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols
                        if not any(c.startswith(p) for p in exclude_prefixes)]
        print(f"Excluded {before - len(feature_cols)} columns matching: {exclude_prefixes}")

    numeric_cols = df_merged[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    dropped = set(feature_cols) - set(numeric_cols)
    if dropped:
        print(f"Dropped non-numeric columns: {sorted(dropped)}")
    feature_cols = numeric_cols

    print(f"Selected {len(feature_cols)} features ({feature_set})")

    keep_cols = feature_cols + ["attractiveness", "subject_id"]
    subset = df_merged[keep_cols].copy()
    n_before = len(subset)
    if drop_missing:
        subset = subset.dropna(subset=feature_cols)
    n_after = len(subset)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} rows with missing feature values")

    X = subset[feature_cols].values.astype(np.float64)
    y = subset["attractiveness"].values.astype(np.float64)
    feature_names = feature_cols

    print(f"Final dataset: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Attractiveness range: {y.min():.2f} - {y.max():.2f} "
          f"(mean {y.mean():.2f}, std {y.std():.2f})")

    return X, y, feature_names, df_merged


def add_common_args(parser):
    """Add shared CLI arguments to an argparse parser."""
    parser.add_argument("--features-csv", type=str, required=True,
                        help="Path to ml_features.csv (from aggregate.py)")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to a labels file (CSV or whitespace-separated txt)")
    parser.add_argument("--feature-set", type=str, default="all",
                        choices=["all", "2d", "3d", "geometric", "appearance",
                                 "2d+composite", "geometric+appearance"],
                        help="Which feature groups to include (default: all)")
    parser.add_argument("--exclude-prefix", type=str, nargs="+", default=None,
                        metavar="PREFIX",
                        help="Drop columns whose name starts with these prefixes")
    return parser
