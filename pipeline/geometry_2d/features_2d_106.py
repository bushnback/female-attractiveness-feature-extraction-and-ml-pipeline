#!/usr/bin/env python3
"""
features_2d_106.py
===================
2D geometric feature extraction from InsightFace 106-point landmarks
(landmarks_106.py). Mirrors the same five feature groups (A-E) as the
68-point version (features_2d_68.py) so results are comparable.

LANDMARK INDEX MAP (verified empirically)
───────────────────────────────────────────────────────────────────────
   0–32   Face contour (33 pts).  0 = chin (menton).  1 / 17 = temple
          endpoints (first/last of the contour array, ~brow height).
          2 / 18 = gonion proxy (corner where the upper face-side arc
          meets the lower jaw arc).
  33–42   Right eye (10 pts).   Outer corner = 35.  Inner corner = 39.
  43–51   Right eyebrow (9 pts).
  52–71   Mouth (20 pts).       Outer corner = 52 (left) / 61 (right).
          Outer lip top/bottom centre = 71 / 53.  Inner lip top/bottom
          centre (seam) = 62 / 60.
  72–86   Nose (15 pts).        Bridge top (nasion-ish) = 72.  Tip
          (pronasale) = 86.  Subnasale (base) = 80.  Ala / nostril
          width = 77 (right) / 83 (left).
  87–96   Left eye (10 pts).    Inner corner = 89.  Outer corner = 93.
  97–105  Left eyebrow (9 pts).

Several 68pt features depended on a single hand-picked vertex (e.g. a
specific upper-eyelid point for "eye height"). The 106pt scheme samples
the eye/brow/mouth/nose contours much more densely without a clean 1:1
anatomical match for every such point, so where the 68pt script used one
named vertex, this script uses a robust aggregate over the relevant
landmark group instead (bounding-box height, convex-hull area, or
extremal-point line).

Mouth landmarks are NOT included in the symmetry mirror-pair list: the
20 mouth points don't have an unambiguous outer/inner split in this
scheme from visual inspection alone. Contour, eyebrow, eye (corners
only), and nose pairs are used instead (32 pairs total).

Usage:
    python features_2d_106.py --landmarks /path/to/landmarks_106pt \
        --output /path/to/out [--photos /path/to/photos]
"""

import argparse
import csv
import json
import math
import numpy as np
import cv2
from pathlib import Path

PHI = (1 + math.sqrt(5)) / 2  # 1.61803...

# ── Landmark index groups (verified empirically) ─────────────────────────────

CONTOUR = list(range(0, 33))
REYE    = list(range(33, 43))
REBROW  = list(range(43, 52))
MOUTH   = list(range(52, 72))
NOSE    = list(range(72, 87))
LEYE    = list(range(87, 97))
LEBROW  = list(range(97, 106))

CHIN      = 0
TEMPLE_R  = 1
TEMPLE_L  = 17
GONION_R  = 2
GONION_L  = 18

REYE_OUTER, REYE_INNER = 35, 39
LEYE_INNER, LEYE_OUTER = 89, 93

NOSE_BRIDGE_TOP = 72
NOSE_TIP        = 86
NOSE_SUBNASALE  = 80
NOSE_ALA_R      = 77
NOSE_ALA_L      = 83

MOUTH_L_CORNER, MOUTH_R_CORNER = 52, 61
UPPER_LIP_OUTER, LOWER_LIP_OUTER = 71, 53
UPPER_LIP_INNER, LOWER_LIP_INNER = 62, 60

REBROW_PEAK, LEBROW_PEAK = 49, 104     # topmost (smallest-y) point of each brow
INTERBROW_R, INTERBROW_L = 50, 102     # innermost (largest/smallest-x) brow points

# Full contour in actual geometric (perimeter) order, temple-to-temple via
# chin. Needed for the jawline curvature feature, which sums angles at
# successive vertices and therefore requires true walking order.
ORDERED_CONTOUR = [
    17, 25, 26, 27, 28, 29, 30, 31, 32, 18, 19, 20, 21, 22, 23, 24,
    0,
    8, 7, 6, 5, 4, 3, 2, 16, 15, 14, 13, 12, 11, 10, 9, 1,
]

# Mirror pairs (right_idx, left_idx) for symmetry features.
MIRROR_PAIRS_CONTOUR = [
    (1, 17), (9, 25), (10, 26), (11, 27), (12, 28), (13, 29), (14, 30),
    (15, 31), (16, 32), (2, 18), (3, 19), (4, 20), (5, 21), (6, 22),
    (7, 23), (8, 24),
]
MIRROR_PAIRS_BROW = [
    (43, 101), (48, 105), (44, 100), (45, 99), (49, 104),
    (47, 98), (51, 103), (46, 97), (50, 102),
]
MIRROR_PAIRS_EYE = [(REYE_OUTER, LEYE_OUTER), (REYE_INNER, LEYE_INNER)]
MIRROR_PAIRS_NOSE = [(75, 81), (76, 82), (77, 83), (78, 84), (79, 85)]

MIRROR_PAIRS = MIRROR_PAIRS_CONTOUR + MIRROR_PAIRS_BROW + MIRROR_PAIRS_EYE + MIRROR_PAIRS_NOSE


# ── Geometry helpers ──────────────────────────────────────────────────────────

def dist(lm, a, b):
    return float(np.linalg.norm(lm[a] - lm[b]))


def midpoint(lm, a, b):
    return (lm[a] + lm[b]) / 2.0


def angle_at_vertex(p_prev, p_vertex, p_next):
    v1 = p_prev - p_vertex
    v2 = p_next - p_vertex
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def line_angle_deg(lm, a, b):
    dx = lm[b][0] - lm[a][0]
    dy = lm[b][1] - lm[a][1]
    return math.degrees(math.atan2(dy, dx))


def bbox_dims(lm, indices):
    """Width/height of the axis-aligned bounding box of a landmark group."""
    pts = lm[indices]
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    return w, h


def hull_area(lm, indices):
    """Convex-hull area of a landmark group — robust to point ordering."""
    pts = lm[indices].astype(np.float32)
    hull = cv2.convexHull(pts)
    return float(cv2.contourArea(hull))


def max_perp_dist_from_line(pts, p_start, p_end):
    """Max perpendicular distance of pts from the line p_start->p_end."""
    line_vec = p_end - p_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0.0
    line_unit = line_vec / line_len
    best = 0.0
    for pt in pts:
        proj = np.dot(pt - p_start, line_unit)
        closest = p_start + proj * line_unit
        best = max(best, float(np.linalg.norm(pt - closest)))
    return best


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_2d_features(lm):
    """
    Extract 2D geometric features from a (106, 2) landmark array.
    All distances normalised by IOD unless noted; angles in degrees.
    """
    f = {}

    iod = dist(lm, REYE_OUTER, LEYE_OUTER)
    f['iod_px'] = round(iod, 1)

    def n(val):
        return round(val / iod, 4) if iod > 0 else float('nan')

    # GROUP A — PROPORTIONS & RATIOS
    face_width_jaw = dist(lm, TEMPLE_R, TEMPLE_L)
    f['face_width_jaw'] = n(face_width_jaw)

    cheekbone_w, _ = bbox_dims(lm, CONTOUR)
    f['face_width_cheekbone'] = n(cheekbone_w)

    brow_mid = midpoint(lm, REBROW_PEAK, LEBROW_PEAK)
    chin = lm[CHIN]
    face_height = float(np.linalg.norm(chin - brow_mid))
    f['face_height_brow_chin'] = n(face_height)

    f['face_width_height_ratio'] = round(face_width_jaw / face_height, 4) if face_height > 0 else float('nan')

    brow_to_nasion = float(np.linalg.norm(lm[NOSE_BRIDGE_TOP] - brow_mid))
    nasion_to_subnasale = dist(lm, NOSE_BRIDGE_TOP, NOSE_SUBNASALE)
    subnasale_to_chin = dist(lm, NOSE_SUBNASALE, CHIN)
    vert_total = brow_to_nasion + nasion_to_subnasale + subnasale_to_chin

    f['brow_nasion_ratio'] = round(brow_to_nasion / vert_total, 4)
    f['nasion_subnasale_ratio'] = round(nasion_to_subnasale / vert_total, 4)
    f['subnasale_chin_ratio'] = round(subnasale_to_chin / vert_total, 4)
    f['vertical_balance'] = round(
        np.std([brow_to_nasion, nasion_to_subnasale, subnasale_to_chin]) / vert_total, 4
    )

    hw_ratio = face_height / face_width_jaw if face_width_jaw > 0 else 0
    f['golden_ratio_face_hw'] = round(abs(hw_ratio - PHI) / PHI, 4)

    mouth_width = dist(lm, MOUTH_L_CORNER, MOUTH_R_CORNER)
    nose_width = dist(lm, NOSE_ALA_R, NOSE_ALA_L)
    mn_ratio = mouth_width / nose_width if nose_width > 0 else 0
    f['golden_ratio_mouth_nose'] = round(abs(mn_ratio - PHI) / PHI, 4)

    f['nose_width'] = n(nose_width)
    f['nose_width_to_face'] = round(nose_width / face_width_jaw, 4) if face_width_jaw > 0 else float('nan')

    f['mouth_width'] = n(mouth_width)
    f['mouth_to_face_ratio'] = round(mouth_width / face_width_jaw, 4) if face_width_jaw > 0 else float('nan')
    f['mouth_to_nose_ratio'] = round(mouth_width / nose_width, 4) if nose_width > 0 else float('nan')

    nose_length = dist(lm, NOSE_BRIDGE_TOP, NOSE_TIP)
    f['nose_length'] = n(nose_length)
    f['nose_length_to_face_height'] = round(nose_length / face_height, 4) if face_height > 0 else float('nan')

    upper_lip_height = dist(lm, UPPER_LIP_OUTER, UPPER_LIP_INNER)
    lower_lip_height = dist(lm, LOWER_LIP_OUTER, LOWER_LIP_INNER)
    total_lip_height = upper_lip_height + lower_lip_height
    f['upper_lip_height'] = n(upper_lip_height)
    f['lower_lip_height'] = n(lower_lip_height)
    f['total_lip_height'] = n(total_lip_height)
    f['lip_ratio_upper_lower'] = round(upper_lip_height / lower_lip_height, 4) if lower_lip_height > 0 else float('nan')
    f['lip_fullness'] = n(total_lip_height)

    # GROUP B — EYE FEATURES
    r_eye_width = dist(lm, REYE_OUTER, REYE_INNER)
    l_eye_width = dist(lm, LEYE_INNER, LEYE_OUTER)
    _, r_eye_height = bbox_dims(lm, REYE)
    _, l_eye_height = bbox_dims(lm, LEYE)
    f['r_eye_width'] = n(r_eye_width)
    f['l_eye_width'] = n(l_eye_width)
    f['r_eye_height'] = n(r_eye_height)
    f['l_eye_height'] = n(l_eye_height)

    f['r_eye_aspect_ratio'] = round(r_eye_height / r_eye_width, 4) if r_eye_width > 0 else float('nan')
    f['l_eye_aspect_ratio'] = round(l_eye_height / l_eye_width, 4) if l_eye_width > 0 else float('nan')
    f['mean_eye_aspect_ratio'] = round(
        ((r_eye_height / r_eye_width) + (l_eye_height / l_eye_width)) / 2, 4
    )

    inner_canthal = dist(lm, REYE_INNER, LEYE_INNER)
    f['inner_canthal_dist'] = n(inner_canthal)
    f['eye_spacing_ratio'] = round(inner_canthal / iod, 4)

    r_eye_tilt = line_angle_deg(lm, REYE_OUTER, REYE_INNER)
    l_eye_tilt = line_angle_deg(lm, LEYE_INNER, LEYE_OUTER)
    f['r_eye_tilt_deg'] = round(r_eye_tilt, 2)
    f['l_eye_tilt_deg'] = round(l_eye_tilt, 2)
    f['mean_eye_tilt_deg'] = round((r_eye_tilt - l_eye_tilt) / 2, 2)

    r_brow_center = lm[REBROW].mean(axis=0)
    l_brow_center = lm[LEBROW].mean(axis=0)
    r_eye_center = lm[REYE].mean(axis=0)
    l_eye_center = lm[LEYE].mean(axis=0)
    r_brow_eye_dist = float(np.linalg.norm(r_brow_center - r_eye_center))
    l_brow_eye_dist = float(np.linalg.norm(l_brow_center - l_eye_center))
    f['r_brow_eye_dist'] = n(r_brow_eye_dist)
    f['l_brow_eye_dist'] = n(l_brow_eye_dist)
    f['mean_brow_eye_dist'] = n((r_brow_eye_dist + l_brow_eye_dist) / 2)

    for indices, prefix in [(REBROW, 'r'), (LEBROW, 'l')]:
        brow_pts = lm[indices]
        x_order = np.argsort(brow_pts[:, 0])
        line_start = brow_pts[x_order[0]]
        line_end = brow_pts[x_order[-1]]
        arch = max_perp_dist_from_line(brow_pts, line_start, line_end)
        f[f'{prefix}_brow_arch'] = n(arch)

    # GROUP C — SYMMETRY
    midline_top = lm[NOSE_BRIDGE_TOP]
    midline_bot = lm[CHIN]
    midline_vec = midline_bot - midline_top
    midline_len = np.linalg.norm(midline_vec)
    midline_unit = midline_vec / midline_len if midline_len > 0 else np.array([0, 1])
    midline_perp = np.array([-midline_unit[1], midline_unit[0]])

    asymmetries = []
    for ri, li in MIRROR_PAIRS:
        r_dist = np.dot(lm[ri] - midline_top, midline_perp)
        l_dist = np.dot(lm[li] - midline_top, midline_perp)
        asymmetries.append(abs(abs(r_dist) - abs(l_dist)))

    mean_asym = float(np.mean(asymmetries))
    f['symmetry_mean_deviation'] = n(mean_asym)
    f['symmetry_max_deviation'] = n(float(np.max(asymmetries)))

    n_contour = len(MIRROR_PAIRS_CONTOUR)
    n_brow = len(MIRROR_PAIRS_BROW)
    n_eye = len(MIRROR_PAIRS_EYE)
    jaw_asym = asymmetries[0:n_contour]
    brow_asym = asymmetries[n_contour:n_contour + n_brow]
    eye_asym = asymmetries[n_contour + n_brow:n_contour + n_brow + n_eye]

    f['jaw_symmetry'] = n(float(np.mean(jaw_asym)))
    f['brow_symmetry'] = n(float(np.mean(brow_asym)))
    f['eye_symmetry'] = n(float(np.mean(eye_asym)))

    nose_tip = lm[NOSE_TIP]
    nose_proj = np.dot(nose_tip - midline_top, midline_perp)
    f['nose_midline_deviation'] = n(abs(float(nose_proj)))

    # GROUP D — JAW & FACE SHAPE
    r_jaw_angle = angle_at_vertex(lm[16], lm[GONION_R], lm[CHIN])
    l_jaw_angle = angle_at_vertex(lm[32], lm[GONION_L], lm[CHIN])
    f['r_jaw_angle_deg'] = round(r_jaw_angle, 2)
    f['l_jaw_angle_deg'] = round(l_jaw_angle, 2)
    f['mean_jaw_angle_deg'] = round((r_jaw_angle + l_jaw_angle) / 2, 2)

    chin_angle = angle_at_vertex(lm[7], lm[CHIN], lm[23])
    f['chin_angle_deg'] = round(chin_angle, 2)

    jaw_width_gonion = dist(lm, GONION_R, GONION_L)
    f['jaw_width_gonion'] = n(jaw_width_gonion)

    f['jaw_taper_ratio'] = round(jaw_width_gonion / cheekbone_w, 4) if cheekbone_w > 0 else float('nan')

    jaw_curvature_sum = 0.0
    for i in range(1, len(ORDERED_CONTOUR) - 1):
        a = angle_at_vertex(lm[ORDERED_CONTOUR[i - 1]], lm[ORDERED_CONTOUR[i]], lm[ORDERED_CONTOUR[i + 1]])
        jaw_curvature_sum += abs(180 - a)
    f['jawline_curvature_sum_deg'] = round(jaw_curvature_sum, 2)

    # GROUP E — SPATIAL RELATIONSHIPS
    mean_eye_center = (r_eye_center + l_eye_center) / 2
    eye_to_nose = float(np.linalg.norm(mean_eye_center - lm[NOSE_TIP]))
    f['eye_to_nose_tip'] = n(eye_to_nose)

    philtrum = dist(lm, NOSE_SUBNASALE, UPPER_LIP_OUTER)
    f['philtrum_length'] = n(philtrum)

    nose_to_mouth = dist(lm, NOSE_SUBNASALE, UPPER_LIP_INNER)
    f['nose_to_mouth'] = n(nose_to_mouth)

    mouth_to_chin = dist(lm, LOWER_LIP_OUTER, CHIN)
    f['mouth_to_chin'] = n(mouth_to_chin)

    f['lower_face_ratio'] = round(mouth_to_chin / subnasale_to_chin, 4) if subnasale_to_chin > 0 else float('nan')

    interbrow = dist(lm, INTERBROW_R, INTERBROW_L)
    f['interbrow_distance'] = n(interbrow)

    r_eye_area = hull_area(lm, REYE)
    l_eye_area = hull_area(lm, LEYE)
    f['r_eye_area'] = round(r_eye_area / (iod * iod), 4)
    f['l_eye_area'] = round(l_eye_area / (iod * iod), 4)
    f['mean_eye_area'] = round((r_eye_area + l_eye_area) / 2 / (iod * iod), 4)

    mouth_area = hull_area(lm, MOUTH)
    f['mouth_area'] = round(mouth_area / (iod * iod), 4)

    nose_area = hull_area(lm, NOSE)
    f['nose_area'] = round(nose_area / (iod * iod), 4)

    return f


def draw_feature_overlay(image, lm, features, subject):
    out = image.copy()

    for i, (x, y) in enumerate(lm.astype(int)):
        if i in CONTOUR: color = (180, 180, 180)
        elif i in REBROW or i in LEBROW: color = (0, 200, 255)
        elif i in NOSE: color = (255, 200, 0)
        elif i in REYE or i in LEYE: color = (255, 100, 100)
        else: color = (0, 0, 255)
        cv2.circle(out, (x, y), 2, color, -1, cv2.LINE_AA)

    def draw_line(a, b, color, thickness=1):
        p1 = tuple(lm[a].astype(int))
        p2 = tuple(lm[b].astype(int))
        cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

    draw_line(REYE_OUTER, LEYE_OUTER, (255, 255, 0), 2)
    draw_line(TEMPLE_R, TEMPLE_L, (200, 200, 200), 1)
    draw_line(NOSE_ALA_R, NOSE_ALA_L, (255, 200, 0), 1)
    draw_line(MOUTH_L_CORNER, MOUTH_R_CORNER, (0, 0, 255), 1)
    cv2.line(out, tuple(lm[NOSE_BRIDGE_TOP].astype(int)), tuple(lm[CHIN].astype(int)),
             (255, 255, 255), 1, cv2.LINE_AA)

    lines = [
        f"Subject: {subject}",
        f"IOD: {features['iod_px']:.0f} px",
        f"Face W/H ratio: {features['face_width_height_ratio']:.3f}",
        f"Golden ratio (H/W): {features['golden_ratio_face_hw']:.3f}",
        f"Eye aspect ratio: {features['mean_eye_aspect_ratio']:.3f}",
        f"Lip fullness: {features['lip_fullness']:.3f} IOD",
        f"Jaw angle: {features['mean_jaw_angle_deg']:.1f} deg",
        f"Symmetry: {features['symmetry_mean_deviation']:.4f} IOD",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    lh = 20
    padding = 15
    panel_w = 400
    h_img, w_img = out.shape[:2]
    panel_h = max(h_img, padding * 2 + lh * len(lines) + 10)

    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    ty = padding + 18
    for line in lines:
        cv2.putText(panel, line, (padding, ty), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        ty += lh

    if h_img < panel_h:
        pad = np.zeros((panel_h - h_img, w_img, 3), dtype=np.uint8)
        out = np.vstack([out, pad])
    elif panel_h < h_img:
        pad = np.zeros((h_img - panel_h, panel_w, 3), dtype=np.uint8)
        pad[:] = (40, 40, 40)
        panel = np.vstack([panel, pad])

    return np.hstack([out, panel])


def run(landmarks_dir, output_dir, photos_dir=None, max_overlays=10):
    landmarks_dir = Path(landmarks_dir)
    output_dir = Path(output_dir)
    photos_dir = Path(photos_dir) if photos_dir else None

    lm_files = sorted(landmarks_dir.glob("*_landmarks.npy"))
    if not lm_files:
        print(f"[ERROR] No landmark files in {landmarks_dir}")
        return None

    print(f"  {len(lm_files)} subjects found")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_overlay = 0

    for lm_path in lm_files:
        subject = lm_path.stem.replace("_landmarks", "")

        lm = np.load(str(lm_path))
        assert lm.shape == (106, 2), f"Unexpected shape: {lm.shape} for {subject}"

        features = extract_2d_features(lm)
        features["subject"] = subject

        subj_out = output_dir / subject
        subj_out.mkdir(parents=True, exist_ok=True)
        features_json = {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in features.items()}
        with open(subj_out / "2d_features.json", "w") as f_out:
            json.dump({"subject": subject, "features": features_json}, f_out, indent=2)

        if photos_dir and n_overlay < max_overlays:
            photo_path = photos_dir / f"{subject}.jpg"
            if photo_path.exists():
                image = cv2.imread(str(photo_path))
                if image is not None:
                    overlay = draw_feature_overlay(image, lm, features, subject)
                    cv2.imwrite(str(subj_out / "overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n_overlay += 1

        all_results.append(features)

    if not all_results:
        print("[ERROR] No subjects processed.")
        return None

    csv_path = output_dir / "2d_features_summary.csv"
    keys = ["subject"] + [k for k in all_results[0].keys() if k != "subject"]
    with open(csv_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"  Total features extracted: {len(keys) - 2}")  # minus subject, iod_px
    print(f"  Summary CSV: {csv_path.resolve()}")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", required=True, help="Folder of *_landmarks.npy (106,2) files")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--photos", default=None, help="Optional photo folder for overlay visualisation")
    args = ap.parse_args()
    run(args.landmarks, args.output, photos_dir=args.photos)


if __name__ == "__main__":
    main()
