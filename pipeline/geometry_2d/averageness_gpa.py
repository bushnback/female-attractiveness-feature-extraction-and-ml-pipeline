#!/usr/bin/env python3
"""
averageness_gpa.py
===================
Compute face-space averageness (distinctiveness) scores using Generalised
Procrustes Analysis (GPA) on raw 2D landmarks (68pt or 106pt), with
explicit head-pose estimation (yaw/pitch/roll) to flag and exclude
pose-contaminated faces from the mean-shape computation.

WHY RAW 2D LANDMARKS, NOT 3D MESHES
─────────────────────────────────────
A learned 3D face model (e.g. MICA) projects faces into a shape space
trained on scans of real people — unusual faces are pulled toward the
centre of that training distribution, i.e. the model is already doing its
own internal averaging. Raw 2D landmarks capture unmediated variation
directly from the image.

AVERAGENESS METHOD — GENERALISED PROCRUSTES ANALYSIS (GPA)
──────────────────────────────────────────────────────────
  1. Center each landmark configuration (remove position).
  2. Scale to unit centroid size (remove size).
  3. Iteratively estimate the mean shape:
       a. Rotate each config to minimise distance to current mean (SVD).
       b. Average aligned configs -> new mean; re-scale to unit size.
       c. Repeat until convergence.
  4. Procrustes distance = ||aligned shape - mean shape||_F.
     Lower = more average face.

POSE ESTIMATION — cv2.solvePnP
────────────────────────────────
GPA cannot correct for out-of-plane head rotation (pitch/yaw). Pitched or
yawed faces inflate the Procrustes distance, making them appear more
distinctive when they are just poorly posed. Yaw/pitch/roll are estimated
with a standard 6-point 3D reference face model; faces exceeding the
threshold are excluded from the mean-shape computation (but still scored
against it).

OUTPUT COLUMNS
──────────────
  procrustes_dist          — Procrustes distance from the mean shape
                             (computed on pose-clean faces only)
  distinctiveness_rank_pct — percentile rank among all faces
  yaw_deg / pitch_deg / roll_deg — estimated head pose in degrees
  pose_flag                — 1 if pose exceeds threshold

REFERENCES
──────────
  Dryden & Mardia (1998) Statistical Shape Analysis. Wiley.
  Langlois & Roggman (1990) Attractive faces are only average.
    Psychological Science, 1(2), 115-121.
  Valentine (1991) A unified account of the effects of distinctiveness,
    inversion, and race in face recognition. QJEP, 43(2), 161-204.

Usage:
    python averageness_gpa.py --landmarks /path/to/landmarks_68pt \
        --output /path/to/out --scheme 68 [--photos /path/to/photos]
"""

import argparse
import csv
import cv2
import numpy as np
from pathlib import Path

GPA_TOL = 1e-10
GPA_MAX_ITER = 500
MIN_SUBJECTS = 20

YAW_THRESHOLD = 15.0
PITCH_THRESHOLD = 10.0
ROLL_THRESHOLD = 30.0

# Standard 3D reference face model (mm, origin at nose tip) — 6-point
# subset standard in OpenCV head-pose literature. Same anatomical points
# are used regardless of landmark scheme; only the 2D index lookup differs.
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0),  # left eye outer corner
    (225.0, 170.0, -135.0),   # right eye outer corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0),   # right mouth corner
], dtype=np.float64)

# Per-scheme 2D landmark indices matching the 6 _MODEL_3D points above:
# (nose_tip, chin, left_eye_outer, right_eye_outer, left_mouth, right_mouth)
POSE_INDEX_MAP = {
    68:  [30, 8, 45, 36, 54, 48],
    106: [86, 0, 93, 35, 61, 52],
}

N_LANDMARKS = {68: 68, 106: 106}


def _build_camera_matrix(w, h):
    focal = float(w)
    return np.array([
        [focal, 0.0, w / 2.0],
        [0.0, focal, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


def estimate_pose(landmarks, scheme, image_w, image_h):
    """
    Estimate yaw, pitch, roll (degrees) from 2D landmarks.
    Returns (yaw, pitch, roll) or (None, None, None) on failure.
    """
    idx = POSE_INDEX_MAP[scheme]
    image_pts = landmarks[idx].astype(np.float64)
    camera_matrix = _build_camera_matrix(image_w, image_h)

    success, rvec, _ = cv2.solvePnP(
        _MODEL_3D, image_pts, camera_matrix, _DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    # Cancel the y-up / y-down coordinate mismatch (baseline Rx(-pi) flip)
    R_COMP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    rmat = R_COMP @ rmat

    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = 0.0

    return yaw, pitch, roll


def _center(X):
    return X - X.mean(axis=0)


def _centroid_size(X):
    Xc = _center(X)
    return float(np.sqrt(np.sum(Xc ** 2)))


def _scale_to_unit(X):
    Xc = _center(X)
    cs = _centroid_size(Xc)
    return Xc / cs if cs > 1e-12 else Xc


def _optimal_rotation(X, target):
    """SVD Procrustes rotation: minimise ||X @ R - target||_F."""
    U, _, Vt = np.linalg.svd(X.T @ target)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.eye(X.shape[1])
    D[-1, -1] = d
    return U @ D @ Vt


def gpa(configs):
    """Generalised Procrustes Analysis on a list of (N, 2) arrays."""
    aligned = [_scale_to_unit(X) for X in configs]
    mean = aligned[0].copy()

    for i in range(GPA_MAX_ITER):
        aligned = [X @ _optimal_rotation(X, mean) for X in aligned]
        new_mean = np.mean(aligned, axis=0)
        cs = _centroid_size(new_mean)
        if cs > 1e-12:
            new_mean /= cs
        change = float(np.sqrt(np.sum((new_mean - mean) ** 2)))
        mean = new_mean
        if change < GPA_TOL:
            print(f"    GPA converged in {i + 1} iterations (delta={change:.2e})")
            break
    else:
        print(f"    GPA reached {GPA_MAX_ITER} iterations (delta={change:.2e})")

    return aligned, mean


def load_landmarks(lm_dir, n_points):
    configs = {}
    for f in sorted(Path(lm_dir).glob("*_landmarks.npy")):
        subj = f.stem.replace("_landmarks", "")
        arr = np.load(str(f))
        if arr.ndim == 2 and arr.shape[0] >= n_points and arr.shape[1] >= 2:
            configs[subj] = arr[:n_points, :2].astype(np.float64)
        else:
            print(f"    [WARN] {subj}: shape {arr.shape} unexpected — skipped")
    return configs


def run(landmarks_dir, output_dir, scheme=68, photos_dir=None,
        image_w=None, image_h=None):
    """
    scheme: 68 or 106 (landmark point count / index convention)
    image_w/image_h: required (per-photo if available) for solvePnP; if
        photos_dir is given these are read per-image automatically.
    """
    assert scheme in N_LANDMARKS, f"Unsupported scheme: {scheme}"
    landmarks_dir = Path(landmarks_dir)
    output_dir = Path(output_dir)
    photos_dir = Path(photos_dir) if photos_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {scheme}pt landmarks from: {landmarks_dir}")
    all_configs = load_landmarks(landmarks_dir, N_LANDMARKS[scheme])
    print(f"  Loaded {len(all_configs)} subjects")

    pose_results = {}
    if photos_dir and photos_dir.exists():
        print("  Estimating head pose (solvePnP)...")
        n_fail = 0
        for subj, lm in all_configs.items():
            w, h = image_w, image_h
            if w is None or h is None:
                photo = photos_dir / f"{subj}.jpg"
                img = cv2.imread(str(photo)) if photo.exists() else None
                if img is not None:
                    h, w = img.shape[:2]
                else:
                    w, h = 350, 350  # fallback
            yaw, pitch, roll = estimate_pose(lm, scheme, w, h)
            if yaw is None:
                n_fail += 1
                yaw, pitch, roll = 0.0, 0.0, 0.0
                flag = 1
            else:
                flag = int(abs(yaw) > YAW_THRESHOLD or
                           abs(pitch) > PITCH_THRESHOLD or
                           abs(roll) > ROLL_THRESHOLD)
            pose_results[subj] = {
                "yaw_deg": round(yaw, 2), "pitch_deg": round(pitch, 2),
                "roll_deg": round(roll, 2), "pose_flag": flag,
            }
        n_flagged = sum(r["pose_flag"] for r in pose_results.values())
        print(f"  Pose flags: {n_flagged}/{len(all_configs)} subjects exceed thresholds")
        if n_fail:
            print(f"  solvePnP failed on {n_fail} subjects (flagged automatically)")
    else:
        print("  [SKIP] No photos_dir — pose estimation skipped, no faces excluded from GPA")
        for subj in all_configs:
            pose_results[subj] = {"yaw_deg": "", "pitch_deg": "", "roll_deg": "", "pose_flag": 0}

    clean_subjects = [s for s in sorted(all_configs) if pose_results[s]["pose_flag"] == 0]
    all_subjects = sorted(all_configs.keys())

    print(f"  Computing mean shape from {len(clean_subjects)} pose-clean subjects...")
    if len(clean_subjects) < MIN_SUBJECTS:
        print(f"  [ERROR] Only {len(clean_subjects)} clean subjects — need >= {MIN_SUBJECTS}.")
        return None

    clean_configs = [all_configs[s] for s in clean_subjects]
    _, mean_shape = gpa(clean_configs)

    all_scores = {}
    for subj in all_subjects:
        X = _scale_to_unit(all_configs[subj])
        R = _optimal_rotation(X, mean_shape)
        aligned = X @ R
        dist = float(np.sqrt(np.sum((aligned - mean_shape) ** 2)))
        all_scores[subj] = dist

    distances = np.array([all_scores[s] for s in all_subjects])
    ranks = distances.argsort().argsort()
    rank_pct = ranks / (len(all_subjects) - 1) * 100.0

    scores = {
        subj: {
            "procrustes_dist": round(float(distances[i]), 6),
            "distinctiveness_rank_pct": round(float(rank_pct[i]), 2),
        }
        for i, subj in enumerate(all_subjects)
    }

    csv_path = output_dir / "averageness_summary.csv"
    fieldnames = ["subject", "procrustes_dist", "distinctiveness_rank_pct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subj in all_subjects:
            row = {"subject": subj}
            row.update(scores[subj])
            writer.writerow(row)

    print(f"  Output: {csv_path} ({len(all_subjects)} subjects)")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", required=True, help="Folder of *_landmarks.npy files")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--scheme", type=int, choices=[68, 106], default=68)
    ap.add_argument("--photos", default=None, help="Optional photo folder for solvePnP pose estimation")
    args = ap.parse_args()
    run(args.landmarks, args.output, scheme=args.scheme, photos_dir=args.photos)


if __name__ == "__main__":
    main()
