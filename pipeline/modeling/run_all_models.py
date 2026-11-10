#!/usr/bin/env python3
"""
run_all_models.py
=================
Runs Lasso, ElasticNet, XGBoost+SHAP, and the Cohen's d extremes analysis
in sequence against an ml_features.csv produced by aggregate.py.

Usage:
    python run_all_models.py --features-csv /path/to/ml_features.csv \
        --labels /path/to/labels.txt --output-dir /path/to/analysis
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MODELS = [
    {"name": "Lasso", "script": SCRIPT_DIR / "run_lasso.py"},
    {"name": "ElasticNet", "script": SCRIPT_DIR / "run_elasticnet.py"},
    {"name": "XGBoost+SHAP", "script": SCRIPT_DIR / "run_xgboost_shap.py"},
]


def run_model(model, features_csv, labels, feature_set, output_base, exclude_prefixes=None):
    name = model["name"]
    script = model["script"]
    outdir = output_base / name.lower().replace("+", "_")

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable, str(script),
        "--features-csv", str(features_csv),
        "--labels", str(labels),
        "--feature-set", feature_set,
        "--output-dir", str(outdir),
    ]
    if exclude_prefixes:
        cmd += ["--exclude-prefix"] + exclude_prefixes

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(script.parent))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {name} finished in {elapsed:.0f}s -> {outdir}/")
    return result.returncode == 0


def run_cohens_d(features_csv, labels, output_base):
    print(f"\n{'=' * 70}")
    print("  Cohen's d Extremes")
    print(f"{'=' * 70}")
    outdir = output_base / "cohens_d"
    outdir.mkdir(parents=True, exist_ok=True)
    script = SCRIPT_DIR / "cohens_d.py"
    cmd = [
        sys.executable, str(script),
        "--features-csv", str(features_csv),
        "--labels", str(labels),
        "--out-csv", str(outdir / "cohens_d_results.csv"),
        "--out-chart", str(outdir / "cohens_d_chart.png"),
    ]
    result = subprocess.run(cmd, cwd=str(script.parent))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all ML models on ml_features.csv")
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-set", default="all",
                        choices=["all", "2d", "3d", "geometric", "appearance",
                                 "2d+composite", "geometric+appearance"])
    parser.add_argument("--output-dir", required=True, help="Base directory for all model outputs")
    parser.add_argument("--exclude-prefix", type=str, nargs="+", default=None, metavar="PREFIX")
    parser.add_argument("--skip-cohens-d", action="store_true")
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print("  Run All Models")
    print(f"{'=' * 70}")
    print(f"  Features CSV: {args.features_csv}")
    print(f"  Labels:       {args.labels}")
    print(f"  Feature set:  {args.feature_set}")
    print(f"  Output:       {output_base}")

    t0 = time.time()
    results = {}
    for model in MODELS:
        results[model["name"]] = run_model(
            model, args.features_csv, args.labels, args.feature_set,
            output_base, exclude_prefixes=args.exclude_prefix,
        )

    if not args.skip_cohens_d:
        results["Cohen's d"] = run_cohens_d(args.features_csv, args.labels, output_base)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Summary  ({elapsed:.0f}s total)")
    print(f"{'=' * 70}")
    for name, ok in results.items():
        print(f"  {'OK    ' if ok else 'FAILED'} {name}")


if __name__ == "__main__":
    main()
