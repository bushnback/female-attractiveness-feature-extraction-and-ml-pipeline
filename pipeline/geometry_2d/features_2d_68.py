#!/usr/bin/env python3
"""
features_2d_68.py
==================
2D geometric feature extraction from 68-point iBUG/FAN landmarks
(landmarks_68.py). All measurements are normalised by inter-ocular
distance (IOD) so they are scale-invariant.

LANDMARK INDEX CONVENTION (Multi-PIE / iBUG 68-point)
──────────────────────────────────────────────────────
   0–16   Jawline contour
  17–21   Right eyebrow
  22–26   Left eyebrow
  27–30   Nose bridge
  31–35   Nose tip / nostrils
  36–41   Right eye (36=outer, 39=inner corner)
  42–47   Left eye (42=inner, 45=outer corner)
  48–59   Outer lip contour
  60–67   Inner lip contour

FEATURES EXTRACTED (~50 features in 5 groups)
─────────────────────────────────────────────
  GROUP A — PROPORTIONS & RATIOS
  GROUP B — EYE FEATURES
  GROUP C — SYMMETRY
  GROUP D — JAW & FACE SHAPE
  GROUP E — SPATIAL RELATIONSHIPS

All features normalised by IOD except angles (degrees) and pure ratios.

Usage:
    python features_2d_68.py --landmarks /path/to/landmarks_68pt \
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


def dist(lm, a, b):
    return float(np.linalg.norm(lm[a] - lm[b]))


def midpoint(lm, a, b):
    return (lm[a] + lm[b]) / 2.0


def polygon_area(pts):
    """Area of a polygon defined by ordered 2D points (shoelace formula)."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


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


def extract_2d_features(lm):
    """Extract all 2D geometric features from a (68, 2) landmark array."""
    f = {}

    iod = dist(lm, 36, 45)
    f['iod_px'] = round(iod, 1)

    def n(val):
        return round(val / iod, 4) if iod > 0 else float('nan')

    # GROUP A — PROPORTIONS & RATIOS
    face_width_jaw = dist(lm, 0, 16)
    f['face_width_jaw'] = n(face_width_jaw)

    jaw_pts = lm[0:17]
    face_width_cheekbone = float(jaw_pts[:, 0].max() - jaw_pts[:, 0].min())
    f['face_width_cheekbone'] = n(face_width_cheekbone)

    brow_mid = midpoint(lm, 19, 24)
    chin = lm[8]
    face_height = float(np.linalg.norm(chin - brow_mid))
    f['face_height_brow_chin'] = n(face_height)

    f['face_width_height_ratio'] = round(face_width_jaw / face_height, 4) if face_height > 0 else float('nan')

    # NOTE: no hairline point in this landmark model, so these are NOT
    # classical facial thirds (trichion-based) — they measure the three
    # vertical segments between brow midpoint and chin.
    brow_to_nasion = float(np.linalg.norm(lm[27] - brow_mid))
    nasion_to_subnasale = dist(lm, 27, 33)
    subnasale_to_chin = dist(lm, 33, 8)
    vert_total = brow_to_nasion + nasion_to_subnasale + subnasale_to_chin

    f['brow_nasion_ratio'] = round(brow_to_nasion / vert_total, 4)
    f['nasion_subnasale_ratio'] = round(nasion_to_subnasale / vert_total, 4)
    f['subnasale_chin_ratio'] = round(subnasale_to_chin / vert_total, 4)
    f['vertical_balance'] = round(
        np.std([brow_to_nasion, nasion_to_subnasale, subnasale_to_chin]) / vert_total, 4
    )

    hw_ratio = face_height / face_width_jaw if face_width_jaw > 0 else 0
    f['golden_ratio_face_hw'] = round(abs(hw_ratio - PHI) / PHI, 4)

    mouth_width = dist(lm, 48, 54)
    nose_width = dist(lm, 31, 35)
    mn_ratio = mouth_width / nose_width if nose_width > 0 else 0
    f['golden_ratio_mouth_nose'] = round(abs(mn_ratio - PHI) / PHI, 4)

    f['nose_width'] = n(nose_width)
    f['nose_width_to_face'] = round(nose_width / face_width_jaw, 4) if face_width_jaw > 0 else float('nan')

    f['mouth_width'] = n(mouth_width)
    f['mouth_to_face_ratio'] = round(mouth_width / face_width_jaw, 4) if face_width_jaw > 0 else float('nan')
    f['mouth_to_nose_ratio'] = round(mouth_width / nose_width, 4) if nose_width > 0 else float('nan')

    nose_length = dist(lm, 27, 30)
    f['nose_length'] = n(nose_length)
    f['nose_length_to_face_height'] = round(nose_length / face_height, 4) if face_height > 0 else float('nan')

    upper_lip_height = dist(lm, 51, 62)
    lower_lip_height = dist(lm, 57, 66)
    total_lip_height = upper_lip_height + lower_lip_height
    f['upper_lip_height'] = n(upper_lip_height)
    f['lower_lip_height'] = n(lower_lip_height)
    f['total_lip_height'] = n(total_lip_height)
    f['lip_ratio_upper_lower'] = round(upper_lip_height / lower_lip_height, 4) if lower_lip_height > 0 else float('nan')
    f['lip_fullness'] = n(total_lip_height)

    # GROUP B — EYE FEATURES
    r_eye_width = dist(lm, 36, 39)
    l_eye_width = dist(lm, 42, 45)
    r_eye_height = dist(lm, 37, 41)
    l_eye_height = dist(lm, 43, 47)
    f['r_eye_width'] = n(r_eye_width)
    f['l_eye_width'] = n(l_eye_width)
    f['r_eye_height'] = n(r_eye_height)
    f['l_eye_height'] = n(l_eye_height)

    f['r_eye_aspect_ratio'] = round(r_eye_height / r_eye_width, 4) if r_eye_width > 0 else float('nan')
    f['l_eye_aspect_ratio'] = round(l_eye_height / l_eye_width, 4) if l_eye_width > 0 else float('nan')
    f['mean_eye_aspect_ratio'] = round(
        ((r_eye_height / r_eye_width) + (l_eye_height / l_eye_width)) / 2, 4
    )

    inner_canthal = dist(lm, 39, 42)
    f['inner_canthal_dist'] = n(inner_canthal)
    f['eye_spacing_ratio'] = round(inner_canthal / iod, 4)

    r_eye_tilt = line_angle_deg(lm, 36, 39)
    l_eye_tilt = line_angle_deg(lm, 42, 45)
    f['r_eye_tilt_deg'] = round(r_eye_tilt, 2)
    f['l_eye_tilt_deg'] = round(l_eye_tilt, 2)
    f['mean_eye_tilt_deg'] = round((r_eye_tilt - l_eye_tilt) / 2, 2)

    r_brow_center = lm[17:22].mean(axis=0)
    l_brow_center = lm[22:27].mean(axis=0)
    r_eye_center = lm[36:42].mean(axis=0)
    l_eye_center = lm[42:48].mean(axis=0)
    r_brow_eye_dist = float(np.linalg.norm(r_brow_center - r_eye_center))
    l_brow_eye_dist = float(np.linalg.norm(l_brow_center - l_eye_center))
    f['r_brow_eye_dist'] = n(r_brow_eye_dist)
    f['l_brow_eye_dist'] = n(l_brow_eye_dist)
    f['mean_brow_eye_dist'] = n((r_brow_eye_dist + l_brow_eye_dist) / 2)

    for indices, prefix in [(range(17, 22), 'r'), (range(22, 27), 'l')]:
        brow_pts = lm[list(indices)]
        line_start = brow_pts[0]
        line_end = brow_pts[-1]
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        if line_len > 0:
            line_unit = line_vec / line_len
            perp_dists = []
            for pt in brow_pts:
                proj = np.dot(pt - line_start, line_unit)
                closest = line_start + proj * line_unit
                perp_dists.append(np.linalg.norm(pt - closest))
            f[f'{prefix}_brow_arch'] = n(max(perp_dists))
        else:
            f[f'{prefix}_brow_arch'] = float('nan')

    # GROUP C — SYMMETRY
    midline_top = lm[27]
    midline_bot = lm[8]
    midline_vec = midline_bot - midline_top
    midline_len = np.linalg.norm(midline_vec)
    midline_unit = midline_vec / midline_len if midline_len > 0 else np.array([0, 1])
    midline_perp = np.array([-midline_unit[1], midline_unit[0]])

    mirror_pairs = [
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # brows
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),  # eyes
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),  # jaw
        (48, 54), (49, 53), (50, 52), (59, 55), (58, 56),  # outer lip
        (60, 64), (61, 63), (67, 65),  # inner lip
        (31, 35), (32, 34),  # nose
    ]

    asymmetries = []
    for ri, li in mirror_pairs:
        r_dist = np.dot(lm[ri] - midline_top, midline_perp)
        l_dist = np.dot(lm[li] - midline_top, midline_perp)
        asymmetries.append(abs(abs(r_dist) - abs(l_dist)))

    mean_asym = float(np.mean(asymmetries))
    f['symmetry_mean_deviation'] = n(mean_asym)
    f['symmetry_max_deviation'] = n(float(np.max(asymmetries)))

    eye_asym = [asymmetries[i] for i in range(5, 11)]
    f['eye_symmetry'] = n(float(np.mean(eye_asym)))

    brow_asym = [asymmetries[i] for i in range(0, 5)]
    f['brow_symmetry'] = n(float(np.mean(brow_asym)))

    jaw_asym = [asymmetries[i] for i in range(11, 19)]
    f['jaw_symmetry'] = n(float(np.mean(jaw_asym)))

    nose_tip = lm[30]
    nose_proj = np.dot(nose_tip - midline_top, midline_perp)
    f['nose_midline_deviation'] = n(abs(float(nose_proj)))

    # GROUP D — JAW & FACE SHAPE
    r_jaw_angle = angle_at_vertex(lm[2], lm[4], lm[8])
    l_jaw_angle = angle_at_vertex(lm[14], lm[12], lm[8])
    f['r_jaw_angle_deg'] = round(r_jaw_angle, 2)
    f['l_jaw_angle_deg'] = round(l_jaw_angle, 2)
    f['mean_jaw_angle_deg'] = round((r_jaw_angle + l_jaw_angle) / 2, 2)

    chin_angle = angle_at_vertex(lm[6], lm[8], lm[10])
    f['chin_angle_deg'] = round(chin_angle, 2)

    jaw_width_gonion = dist(lm, 4, 12)
    f['jaw_width_gonion'] = n(jaw_width_gonion)

    f['jaw_taper_ratio'] = round(jaw_width_gonion / face_width_cheekbone, 4) if face_width_cheekbone > 0 else float('nan')

    jaw_curvature_sum = 0
    for i in range(1, 16):
        a = angle_at_vertex(lm[i - 1], lm[i], lm[i + 1])
        jaw_curvature_sum += abs(180 - a)
    f['jawline_curvature_sum_deg'] = round(jaw_curvature_sum, 2)

    # GROUP E — SPATIAL RELATIONSHIPS
    mean_eye_center = (r_eye_center + l_eye_center) / 2
    eye_to_nose = float(np.linalg.norm(mean_eye_center - lm[30]))
    f['eye_to_nose_tip'] = n(eye_to_nose)

    philtrum = dist(lm, 33, 51)
    f['philtrum_length'] = n(philtrum)

    nose_to_mouth = dist(lm, 33, 62)
    f['nose_to_mouth'] = n(nose_to_mouth)

    mouth_to_chin = dist(lm, 57, 8)
    f['mouth_to_chin'] = n(mouth_to_chin)

    f['lower_face_ratio'] = round(mouth_to_chin / subnasale_to_chin, 4) if subnasale_to_chin > 0 else float('nan')

    interbrow = dist(lm, 21, 22)
    f['interbrow_distance'] = n(interbrow)

    r_eye_area = polygon_area(lm[36:42])
    l_eye_area = polygon_area(lm[42:48])
    f['r_eye_area'] = round(r_eye_area / (iod * iod), 4)
    f['l_eye_area'] = round(l_eye_area / (iod * iod), 4)
    f['mean_eye_area'] = round((r_eye_area + l_eye_area) / 2 / (iod * iod), 4)

    mouth_area = polygon_area(lm[48:60])
    f['mouth_area'] = round(mouth_area / (iod * iod), 4)

    nose_area = polygon_area(lm[27:36])
    f['nose_area'] = round(nose_area / (iod * iod), 4)

    return f


def draw_feature_overlay(image, lm, features, subject):
    out = image.copy()

    for i, (x, y) in enumerate(lm.astype(int)):
        if i <= 16: color = (180, 180, 180)
        elif i <= 26: color = (0, 200, 255)
        elif i <= 35: color = (255, 200, 0)
        elif i <= 47: color = (255, 100, 100)
        else: color = (0, 0, 255)
        cv2.circle(out, (x, y), 3, color, -1, cv2.LINE_AA)

    def draw_line(a, b, color, thickness=1):
        p1 = tuple(lm[a].astype(int))
        p2 = tuple(lm[b].astype(int))
        cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

    draw_line(36, 45, (255, 255, 0), 2)
    draw_line(0, 16, (200, 200, 200), 1)
    draw_line(31, 35, (255, 200, 0), 1)
    draw_line(48, 54, (0, 0, 255), 1)
    cv2.line(out, tuple(lm[27].astype(int)), tuple(lm[8].astype(int)),
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
        assert lm.shape == (68, 2), f"Unexpected shape: {lm.shape}"

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

    print(f"  Total features extracted: {len(keys) - 2}")
    print(f"  Summary CSV: {csv_path.resolve()}")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", required=True, help="Folder of *_landmarks.npy (68,2) files")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--photos", default=None, help="Optional photo folder for overlay visualisation")
    args = ap.parse_args()
    run(args.landmarks, args.output, photos_dir=args.photos)


if __name__ == "__main__":
    main()
