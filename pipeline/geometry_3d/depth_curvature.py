#!/usr/bin/env python3
"""
depth_curvature.py
===================
Extract 3D depth and curvature features from DECA FLAME meshes that cannot
be captured by 2D landmark ratios alone: nose protrusion, orbital depth,
cheekbone prominence, chin projection, forehead slope, facial convexity,
lip protrusion, 3D symmetry, and surface curvature at key regions.

COORDINATE SYSTEM (FLAME)
─────────────────────────
  X — lateral:   positive = face-right,  negative = face-left
  Y — vertical:  positive = up (forehead), negative = down (chin)
  Z — depth:     positive = forward (face surface), negative = backward

DATA SOURCES (per subject)
───────────────────────────
  mesh.ply                       — FLAME mesh from deca_mesh.py
  dad3dheads_landmarks_3d.npy    — from dad_landmarks_445.py
  dad3dheads_landmark_indices.npy
  dad_3dheads_keypoints/*.npy    — region definitions (bundled)

Requires: open3d

Usage:
    python depth_curvature.py --input /path/to/subjects --output /path/to/out
    (input: one subfolder per subject, each containing mesh.ply +
     dad3dheads_landmarks_3d.npy + dad3dheads_landmark_indices.npy)
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

REGION_DIR_DEFAULT = Path(__file__).resolve().parent / "dad_3dheads_keypoints"


def load_region_indices(region_dir=REGION_DIR_DEFAULT):
    """Load DAD-3DHeads region definitions -> {region_name: vertex_indices}."""
    regions = {}
    for npy_file in sorted(Path(region_dir).glob("*.npy")):
        data = np.load(str(npy_file), allow_pickle=True).item()
        for subregion, indices in data.items():
            regions[subregion] = np.array(indices)
    return regions


def get_region_landmarks(lm_3d, lm_indices, region_vertex_indices):
    region_set = set(region_vertex_indices)
    mask = np.array([idx in region_set for idx in lm_indices])
    return lm_3d[mask]


def fit_plane(points):
    """Fit a plane to 3D points via SVD. Returns (centroid, normal)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


def point_to_plane_distance(point, plane_centroid, plane_normal):
    return np.dot(point - plane_centroid, plane_normal)


def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def extract_3d_features(mesh, lm_3d, lm_indices, regions):
    """Extract all 3D depth/curvature features for one subject."""
    vertices = np.asarray(mesh.vertices).copy()
    features = {}

    nose_bridge = get_region_landmarks(lm_3d, lm_indices, regions.get("nose_front_bridge", []))
    nose_lwing1 = get_region_landmarks(lm_3d, lm_indices, regions.get("left_wing_1", []))
    nose_rwing1 = get_region_landmarks(lm_3d, lm_indices, regions.get("right_wing_1", []))
    nose_lwing2 = get_region_landmarks(lm_3d, lm_indices, regions.get("left_wing_2", []))
    nose_rwing2 = get_region_landmarks(lm_3d, lm_indices, regions.get("right_wing_2", []))
    nose_lnostril = get_region_landmarks(lm_3d, lm_indices, regions.get("left_nostrel", []))
    nose_rnostril = get_region_landmarks(lm_3d, lm_indices, regions.get("right_nostrel", []))
    nose_all_wings = np.concatenate([nose_lwing1, nose_rwing1, nose_lwing2, nose_rwing2,
                                      nose_lnostril, nose_rnostril])

    reye_top = get_region_landmarks(lm_3d, lm_indices, regions.get("reye_top", []))
    reye_bot = get_region_landmarks(lm_3d, lm_indices, regions.get("reye_bottom", []))
    leye_top = get_region_landmarks(lm_3d, lm_indices, regions.get("leye_top", []))
    leye_bot = get_region_landmarks(lm_3d, lm_indices, regions.get("leye_bottom", []))

    all_eyes = np.concatenate([reye_top, reye_bot, leye_top, leye_bot])

    upper_rbrow = get_region_landmarks(lm_3d, lm_indices, regions.get("upper_rbrow", []))
    upper_lbrow = get_region_landmarks(lm_3d, lm_indices, regions.get("upper_lbrow", []))
    lower_rbrow = get_region_landmarks(lm_3d, lm_indices, regions.get("lower_rbrow", []))
    lower_lbrow = get_region_landmarks(lm_3d, lm_indices, regions.get("lower_lbrow", []))
    all_brows_upper = np.concatenate([upper_rbrow, upper_lbrow])
    all_brows_lower = np.concatenate([lower_rbrow, lower_lbrow])

    ulipout = get_region_landmarks(lm_3d, lm_indices, regions.get("ulipout", []))
    llipout = get_region_landmarks(lm_3d, lm_indices, regions.get("llipout", []))
    all_lips = np.concatenate([ulipout, llipout])

    forehead = get_region_landmarks(lm_3d, lm_indices, regions.get("forehead", []))
    forehead_extra = get_region_landmarks(lm_3d, lm_indices, regions.get("extra", []))
    all_forehead = np.concatenate([forehead, forehead_extra]) if len(forehead_extra) > 0 else forehead

    r_cheek_contour = get_region_landmarks(lm_3d, lm_indices, regions.get("right_cheek_contour", []))
    l_cheek_contour = get_region_landmarks(lm_3d, lm_indices, regions.get("left_cheek_contour", []))

    r_chin = get_region_landmarks(lm_3d, lm_indices, regions.get("right_chin_contour", []))
    l_chin = get_region_landmarks(lm_3d, lm_indices, regions.get("left_chin_contour", []))
    all_chin = np.concatenate([r_chin, l_chin])

    r_temple = get_region_landmarks(lm_3d, lm_indices, regions.get("right_temple", []))
    l_temple = get_region_landmarks(lm_3d, lm_indices, regions.get("left_temple", []))

    # 1. Nose protrusion — nose tip Z relative to cheek plane
    cheek_pts = np.concatenate([r_cheek_contour, l_cheek_contour])
    if len(cheek_pts) >= 3 and len(nose_bridge) >= 1:
        cheek_centroid, cheek_normal = fit_plane(cheek_pts)
        nose_tip = nose_bridge[np.argmax(nose_bridge[:, 2])]
        nose_protrusion = point_to_plane_distance(nose_tip, cheek_centroid, cheek_normal)
        features["nose_protrusion_mm"] = round(float(nose_protrusion), 2)
    else:
        features["nose_protrusion_mm"] = float('nan')

    # 2. Orbital depth — brow ridge Z minus eye surface Z
    if len(all_brows_lower) >= 2 and len(all_eyes) >= 2:
        brow_ridge_z = all_brows_lower[:, 2].mean()
        eye_surface_z = all_eyes[:, 2].mean()
        orbital_depth = brow_ridge_z - eye_surface_z
        features["orbital_depth_mm"] = round(float(orbital_depth), 2)
        face_depth = lm_3d[:, 2].max() - lm_3d[:, 2].min()
        features["orbital_depth_ratio"] = round(float(orbital_depth / face_depth), 4)
    else:
        features["orbital_depth_mm"] = float('nan')
        features["orbital_depth_ratio"] = float('nan')

    # 3. Cheekbone prominence — cheek Z vs temple Z
    if len(r_cheek_contour) >= 2 and len(l_cheek_contour) >= 2:
        r_cheek_z = r_cheek_contour[:, 2].max()
        l_cheek_z = l_cheek_contour[:, 2].max()
        cheekbone_max_z = (r_cheek_z + l_cheek_z) / 2
        temple_z = (np.concatenate([r_temple, l_temple])[:, 2].mean()
                    if len(r_temple) > 0 and len(l_temple) > 0 else float('nan'))
        if not np.isnan(temple_z):
            features["cheekbone_prominence_mm"] = round(float(cheekbone_max_z - temple_z), 2)
        else:
            features["cheekbone_prominence_mm"] = float('nan')
        all_cheek = np.concatenate([r_cheek_contour, l_cheek_contour])
        features["cheekbone_width_mm"] = round(float(all_cheek[:, 0].max() - all_cheek[:, 0].min()), 2)
    else:
        features["cheekbone_prominence_mm"] = float('nan')
        features["cheekbone_width_mm"] = float('nan')

    # 4. Chin projection — chin tip Z minus lip base Z
    if len(all_chin) >= 2 and len(all_lips) >= 2:
        chin_tip = all_chin[np.argmin(all_chin[:, 1])]
        lip_base_z = all_lips[:, 2].mean()
        chin_projection = chin_tip[2] - lip_base_z
        features["chin_projection_mm"] = round(float(chin_projection), 2)
        face_depth = lm_3d[:, 2].max() - lm_3d[:, 2].min()
        features["chin_projection_ratio"] = round(float(chin_projection / face_depth), 4)
    else:
        features["chin_projection_mm"] = float('nan')
        features["chin_projection_ratio"] = float('nan')

    # 5. Forehead slope — angle of brow->forehead-top vector vs vertical
    if len(all_forehead) >= 2 and len(all_brows_upper) >= 2:
        fh_top = all_forehead[np.argmax(all_forehead[:, 1])]
        brow_mid = all_brows_upper.mean(axis=0)
        fh_vec = fh_top - brow_mid
        vertical = np.array([0, 1, 0])
        features["forehead_slope_deg"] = round(float(angle_between_vectors(fh_vec, vertical)), 2)
    else:
        features["forehead_slope_deg"] = float('nan')

    # 6. Facial convexity — glabella -> nose tip -> chin tip profile angle
    if len(all_brows_lower) >= 2 and len(nose_bridge) >= 1 and len(all_chin) >= 2:
        glabella = all_brows_lower.mean(axis=0)
        nose_tip = nose_bridge[np.argmax(nose_bridge[:, 2])]
        chin_tip = all_chin[np.argmin(all_chin[:, 1])]
        v1 = glabella - nose_tip
        v2 = chin_tip - nose_tip
        features["facial_convexity_deg"] = round(float(angle_between_vectors(v1, v2)), 2)
    else:
        features["facial_convexity_deg"] = float('nan')

    # 7. Mid-face depth ratio
    face_depth = lm_3d[:, 2].max() - lm_3d[:, 2].min()
    features["face_depth_mm"] = round(float(face_depth), 2)
    if not np.isnan(features.get("nose_protrusion_mm", float('nan'))):
        features["midface_depth_ratio"] = round(features["nose_protrusion_mm"] / face_depth, 4)
    else:
        features["midface_depth_ratio"] = float('nan')

    # 8. Nose width ratio (3D)
    if len(nose_all_wings) >= 4 and len(all_eyes) >= 4:
        nose_width_3d = nose_all_wings[:, 0].max() - nose_all_wings[:, 0].min()
        r_eye_center = np.concatenate([reye_top, reye_bot]).mean(axis=0)
        l_eye_center = np.concatenate([leye_top, leye_bot]).mean(axis=0)
        inter_eye_3d = np.linalg.norm(r_eye_center - l_eye_center)
        features["nose_width_mm"] = round(float(nose_width_3d), 2)
        features["nose_width_to_eye_dist"] = round(float(nose_width_3d / inter_eye_3d), 4)
    else:
        features["nose_width_mm"] = float('nan')
        features["nose_width_to_eye_dist"] = float('nan')

    # 9. Brow ridge prominence
    if len(all_brows_upper) >= 2 and len(all_forehead) >= 2:
        brow_z = all_brows_upper[:, 2].mean()
        forehead_z = all_forehead[:, 2].mean()
        features["brow_ridge_prominence_mm"] = round(float(brow_z - forehead_z), 2)
    else:
        features["brow_ridge_prominence_mm"] = float('nan')

    # 10. Lip protrusion — lip center offset from nose-chin line
    if len(all_lips) >= 2 and len(nose_bridge) >= 1 and len(all_chin) >= 2:
        nose_tip = nose_bridge[np.argmax(nose_bridge[:, 2])]
        chin_tip = all_chin[np.argmin(all_chin[:, 1])]
        lip_center = all_lips.mean(axis=0)
        nc_vec = chin_tip - nose_tip
        nc_len = np.linalg.norm(nc_vec)
        if nc_len > 0:
            nc_unit = nc_vec / nc_len
            t = np.dot(lip_center - nose_tip, nc_unit)
            closest_on_line = nose_tip + t * nc_unit
            lip_offset = lip_center - closest_on_line
            lip_protrusion = lip_offset[2]
            features["lip_protrusion_mm"] = round(float(lip_protrusion), 2)
            features["lip_protrusion_ratio"] = round(float(lip_protrusion / face_depth), 4)
        else:
            features["lip_protrusion_mm"] = float('nan')
            features["lip_protrusion_ratio"] = float('nan')
    else:
        features["lip_protrusion_mm"] = float('nan')
        features["lip_protrusion_ratio"] = float('nan')

    # 11. 3D facial symmetry — mirrored-region depth asymmetry
    symmetry_pairs = [
        (r_cheek_contour, l_cheek_contour),
        (reye_top, leye_top),
        (reye_bot, leye_bot),
        (upper_rbrow, upper_lbrow),
        (r_temple, l_temple),
    ]
    z_diffs = []
    for r_pts, l_pts in symmetry_pairs:
        if len(r_pts) >= 2 and len(l_pts) >= 2:
            z_diffs.append(abs(r_pts[:, 2].mean() - l_pts[:, 2].mean()))
    if z_diffs:
        features["depth_asymmetry_mm"] = round(float(np.mean(z_diffs)), 2)
        features["depth_asymmetry_ratio"] = round(float(np.mean(z_diffs) / face_depth), 4)
    else:
        features["depth_asymmetry_mm"] = float('nan')
        features["depth_asymmetry_ratio"] = float('nan')

    # 12. Surface curvature at key regions (discrete Laplacian magnitude)
    mesh.compute_vertex_normals()
    triangles = np.asarray(mesh.triangles)
    adj = defaultdict(set)
    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[tri[i]].add(tri[j])

    curvature = np.zeros(len(vertices))
    for vi in range(len(vertices)):
        neighbors = list(adj[vi])
        if len(neighbors) == 0:
            continue
        neighbor_mean = vertices[neighbors].mean(axis=0)
        curvature[vi] = np.linalg.norm(vertices[vi] - neighbor_mean)

    def region_curvature(*region_names):
        idxs = np.concatenate([regions.get(r, np.array([], dtype=int)) for r in region_names])
        valid = idxs[idxs < len(vertices)]
        return round(float(curvature[valid].mean()), 4) if len(valid) > 0 else float('nan')

    features["cheekbone_curvature"] = region_curvature("right_cheek_contour", "left_cheek_contour")
    features["jaw_curvature"] = region_curvature("right_chin_contour", "left_chin_contour")
    nb_idx = regions.get("nose_front_bridge", np.array([], dtype=int))
    nb_valid = nb_idx[nb_idx < len(vertices)]
    features["nose_bridge_curvature"] = (round(float(curvature[nb_valid].mean()), 4)
                                          if len(nb_valid) > 0 else float('nan'))

    return features


def run(input_dir, output_dir, region_dir=REGION_DIR_DEFAULT):
    """input_dir: one subfolder per subject, each with mesh.ply +
    dad3dheads_landmarks_3d.npy + dad3dheads_landmark_indices.npy."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = load_region_indices(region_dir)
    print(f"  Loaded {len(regions)} sub-regions")

    subject_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "mesh.ply").exists()
    ])
    if not subject_dirs:
        print(f"[ERROR] No subjects found in {input_dir}")
        return None
    print(f"  Found {len(subject_dirs)} subjects")

    all_results = []
    for subj_dir in subject_dirs:
        subject = subj_dir.name
        lm_path = subj_dir / "dad3dheads_landmarks_3d.npy"
        idx_path = subj_dir / "dad3dheads_landmark_indices.npy"
        if not lm_path.exists() or not idx_path.exists():
            print(f"  [SKIP] {subject}: missing landmark files")
            continue

        mesh = o3d.io.read_triangle_mesh(str(subj_dir / "mesh.ply"))
        lm_3d = np.load(str(lm_path))
        lm_indices = np.load(str(idx_path))

        try:
            features = extract_3d_features(mesh, lm_3d, lm_indices, regions)
        except Exception as e:
            print(f"  [FAIL] {subject}: {e}")
            continue

        subj_out = output_dir / subject
        subj_out.mkdir(parents=True, exist_ok=True)
        with open(subj_out / "3d_features.json", "w") as f:
            json.dump({"subject": subject, "features": features}, f, indent=2)

        features["subject"] = subject
        all_results.append(features)
        print(f"  {subject}: OK")

    if not all_results:
        print("[ERROR] No subjects processed.")
        return None

    csv_path = output_dir / "3d_features_summary.csv"
    keys = ["subject"] + [k for k in all_results[0].keys() if k != "subject"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"  Summary CSV: {csv_path.resolve()}")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Folder with one subfolder per subject (mesh.ply + DAD landmarks)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--region-dir", default=str(REGION_DIR_DEFAULT))
    args = ap.parse_args()
    run(args.input, args.output, region_dir=args.region_dir)


if __name__ == "__main__":
    main()
