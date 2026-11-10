#!/usr/bin/env python3
"""
composite_femininity.py
========================
Compute femininity and neoteny composite scores from raw 2D/3D geometric
features (geometry_2d + geometry_3d stage outputs).

Sexual dimorphism is one of the strongest predictors of perceived
attractiveness in women (Perrett et al. 1998; Rhodes 2006). Faces with
more feminine features — larger eyes, fuller lips, smaller jaw, higher
cheekbones, less prominent brow ridge — are rated as more attractive.

This measures how strongly each face exhibits traits associated with
feminine attractiveness, relative to the rest of the sample (not against
a male population, since the pipeline targets female faces only):
  - Large eyes relative to face size      - High cheekbones (3D)
  - Full lips                             - Less prominent brow ridge (3D)
  - Narrow/tapered jaw                    - Smaller nose relative to face
  - Rounder face shape

A separate neoteny (babyfacedness) score captures youthfulness cues that
overlap with but are distinct from femininity: large eyes, small nose,
round face, full lips, larger brow-to-nasion proportion.

METHODOLOGY
───────────
Each component feature is min-max normalised to 0-1 within the sample
(1.0 = most feminine/neotenous direction), inverting features where a
LOWER raw value is more feminine (e.g. nose size, jaw width). The
composite score is the mean of all available component scores.

This produces a ranking RELATIVE TO THE SAMPLE, not population norms —
with a small sample the scores show the method, not absolute femininity.

DATA SOURCES (per subject, one subfolder per subject)
───────────────────────────────────────────────────────
  2d_features.json       — from features_2d_68.py / features_2d_106.py
  3d_features.json       — from depth_curvature.py
  surface_features.json  — from surface_geodesic.py
(any subset may be present; scoring uses whatever is available)

Usage:
    python composite_femininity.py --input /path/to/subjects --output /path/to/out
"""

import argparse
import json
import csv
import math
import numpy as np
from pathlib import Path

# Each component: (feature_name, source_file_prefix, direction, description)
#   direction = "higher" -> higher value = more feminine/neotenous
#   direction = "lower"  -> lower value = more feminine/neotenous

FEMININITY_COMPONENTS = [
    ("mean_eye_area", "2d", "higher", "Eye size relative to face — larger eyes are a strong femininity cue"),
    ("mean_eye_aspect_ratio", "2d", "higher", "Rounder/taller eyes — more feminine eye shape"),
    ("total_lip_height", "2d", "higher", "Lip fullness — fuller lips are a femininity marker"),
    ("lip_ratio_upper_lower", "2d", "higher", "Upper-to-lower lip balance — fuller upper lip is more feminine"),
    ("nose_width", "2d", "lower", "Nose width relative to IOD — narrower nose is more feminine"),
    ("nose_area", "2d", "lower", "Nose size — smaller nose relative to face is more feminine"),
    ("jaw_taper_ratio", "2d", "lower", "Jaw width relative to cheekbone width — more tapered = more feminine"),
    ("mean_jaw_angle_deg", "2d", "higher", "Jaw angle — wider (less angular) jaw = softer = more feminine"),
    ("face_width_height_ratio", "2d", "higher", "Face width relative to height — rounder face = more feminine"),
    ("mean_brow_eye_dist", "2d", "higher", "Brow-to-eye distance — higher brows = more feminine"),
    ("eye_spacing_ratio", "2d", "higher", "Eye spacing — wider-set eyes are a femininity cue"),
    ("cheekbone_prominence_mm", "3d", "higher", "Cheekbone forward projection — more prominent = more feminine"),
    ("brow_ridge_prominence_mm", "3d", "lower", "Brow ridge projection — less prominent = more feminine"),
    ("chin_projection_mm", "3d", "lower", "Chin forward projection — less projecting = more feminine"),
    ("nose_area_ratio", "surface", "lower", "Nose surface area relative to face — smaller = more feminine"),
    ("eyes_area_ratio", "surface", "higher", "Eye region area relative to face — larger = more feminine"),
    ("lips_area_ratio", "surface", "higher", "Lip area relative to face — fuller = more feminine"),
    ("jaw_contour_angularity_deg", "surface", "lower", "Jaw edge sharpness — softer = more feminine"),
]

NEOTENY_COMPONENTS = [
    ("mean_eye_area", "2d", "higher", "Large eyes — strongest neoteny signal"),
    ("mean_eye_aspect_ratio", "2d", "higher", "Round/open eyes — youthful appearance"),
    ("nose_width", "2d", "lower", "Small nose — strong neoteny cue"),
    ("nose_area", "2d", "lower", "Small nose area — neotenous"),
    ("face_width_height_ratio", "2d", "higher", "Round face — baby-like proportions"),
    ("total_lip_height", "2d", "higher", "Full lips — youthful"),
    ("brow_nasion_ratio", "2d", "higher", "Larger brow-to-nasion ratio — neotenous (higher brow-nasion segment)"),
    ("subnasale_chin_ratio", "2d", "lower", "Shorter subnasale-chin ratio — more baby-like proportions"),
    ("jaw_taper_ratio", "2d", "lower", "Tapered jaw — youthful, not yet developed"),
    ("nose_area_ratio", "surface", "lower", "Small nose surface area — neotenous"),
    ("eyes_area_ratio", "surface", "higher", "Large eye region — neotenous"),
]


def load_subject_features(subj_dir):
    """Load all available feature JSONs for a subject."""
    features = {}
    for fname, prefix in [("2d_features.json", "2d"),
                           ("3d_features.json", "3d"),
                           ("surface_features.json", "surface")]:
        fpath = Path(subj_dir) / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
                feat = data.get("features", data)
                for k, v in feat.items():
                    features[(prefix, k)] = v
    return features


def compute_composite(subjects, components):
    """
    1. For each component feature, gather values across all subjects.
    2. Normalise to 0-1 (min-max within sample).
    3. Invert if direction = "lower" (so 1.0 = most feminine/neotenous).
    4. Composite = mean of all available component scores.
    """
    component_values = {}
    for feat_name, source, direction, description in components:
        key = (source, feat_name)
        vals = []
        for subj_name, features in subjects.items():
            v = features.get(key)
            if v is not None and isinstance(v, (int, float)) and not math.isnan(v):
                vals.append((subj_name, v))
        if len(vals) >= 2:
            component_values[(feat_name, source)] = {
                "values": dict(vals), "direction": direction, "description": description,
                "min": min(v for _, v in vals), "max": max(v for _, v in vals),
            }

    results = {}
    for subj_name in subjects:
        scores, details = [], []
        for (feat_name, source), info in component_values.items():
            if subj_name not in info["values"]:
                continue
            v = info["values"][subj_name]
            vmin, vmax = info["min"], info["max"]
            rng = vmax - vmin
            normalised = (v - vmin) / rng if rng > 0 else 0.5
            if info["direction"] == "lower":
                normalised = 1.0 - normalised
            scores.append(normalised)
            details.append({
                "feature": feat_name, "source": source, "raw_value": round(v, 4),
                "score_0_1": round(normalised, 4), "direction": info["direction"],
                "description": info["description"],
            })
        if scores:
            results[subj_name] = {
                "composite_score": round(float(np.mean(scores)), 4),
                "n_components": len(scores),
                "component_scores": details,
            }
    return results


def run(input_dir, output_dir):
    """input_dir: one subfolder per subject with 2d_features.json /
    3d_features.json / surface_features.json (any subset)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subject_dirs:
        print(f"[ERROR] No subjects in {input_dir}")
        return None

    subjects = {}
    for subj_dir in subject_dirs:
        features = load_subject_features(subj_dir)
        if features:
            subjects[subj_dir.name] = features
    print(f"  {len(subjects)} subjects loaded")

    output_dir.mkdir(parents=True, exist_ok=True)

    fem_results = compute_composite(subjects, FEMININITY_COMPONENTS)
    neo_results = compute_composite(subjects, NEOTENY_COMPONENTS)

    all_scores = []
    for subj_name in subjects:
        row = {"subject": subj_name}
        if subj_name in fem_results:
            row["femininity_score"] = fem_results[subj_name]["composite_score"]
            row["femininity_n_components"] = fem_results[subj_name]["n_components"]
        if subj_name in neo_results:
            row["neoteny_score"] = neo_results[subj_name]["composite_score"]
            row["neoteny_n_components"] = neo_results[subj_name]["n_components"]
        all_scores.append(row)

        subj_out = output_dir / subj_name
        subj_out.mkdir(parents=True, exist_ok=True)
        result = {"subject": subj_name}
        if subj_name in fem_results:
            result["femininity"] = fem_results[subj_name]
        if subj_name in neo_results:
            result["neoteny"] = neo_results[subj_name]
        with open(subj_out / "composite_scores.json", "w") as f:
            json.dump(result, f, indent=2)

    csv_path = output_dir / "composite_scores_summary.csv"
    keys = ["subject", "femininity_score", "femininity_n_components",
            "neoteny_score", "neoteny_n_components"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_scores)

    both = [(fem_results[s]["composite_score"], neo_results[s]["composite_score"])
            for s in subjects if s in fem_results and s in neo_results]
    if len(both) >= 3:
        corr = np.corrcoef([x[0] for x in both], [x[1] for x in both])[0, 1]
        print(f"  Correlation between femininity and neoteny: r = {corr:.3f}")

    print(f"  Summary CSV: {csv_path.resolve()}")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Folder with one subfolder per subject (2D/3D/surface feature JSONs)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
