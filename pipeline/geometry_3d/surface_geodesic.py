#!/usr/bin/env python3
"""
surface_geodesic.py
====================
Extract surface area ratios, geodesic distances, and regional angularity
features from DECA FLAME meshes — complementary to depth_curvature.py.

  - Surface area ratios answer "how big is this feature relative to the
    face?" (e.g. nose area / face area)
  - Geodesic distances capture surface complexity — the ratio of
    along-the-surface distance to straight-line distance reveals how much
    curvature/relief exists along a path
  - Regional angularity measures how sharp or smooth each area is
  - Surface normal variance measures how flat vs. sculpted a region is

These are fundamentally 3D features: 2D areas are distorted by
perspective, and surface distances/angles require actual 3D geometry.

DATA SOURCES (per subject)
───────────────────────────
  mesh.ply, dad3dheads_landmarks_3d.npy, dad3dheads_landmark_indices.npy
  dad_3dheads_keypoints/*.npy (bundled region definitions)

Requires: open3d, scipy

Usage:
    python surface_geodesic.py --input /path/to/subjects --output /path/to/out
"""

import argparse
import csv
import json
import math
import heapq
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

REGION_DIR_DEFAULT = Path(__file__).resolve().parent / "dad_3dheads_keypoints"

HIGH_LEVEL_REGIONS = {
    'nose': ['nose_front_bridge', 'central_philtrum', 'nose_left_bridge', 'left_philtrum',
             'nose_right_bridge', 'right_philtrum', 'left_wing_1', 'right_wing_1',
             'left_wing_2', 'right_wing_2', 'left_wing_3', 'right_wing_3',
             'left_nostrel', 'right_nostrel'],
    'eyes': ['reye_bottom', 'reye_top', 'leye_bottom', 'leye_top',
             'around_left_eye_up', 'around_left_eye_down',
             'around_right_eye_up', 'around_right_eye_down'],
    'brows': ['upper_rbrow', 'middle_rbrow', 'lower_rbrow',
              'upper_lbrow', 'middle_lbrow', 'lower_lbrow'],
    'lips': ['llipin', 'llipout', 'ulipin', 'ulipout',
             'around_upper_lip', 'around_lower_lip'],
    'forehead': ['forehead', 'extra'],
    'cheeks': ['right_cheek_contour', 'left_cheek_contour'],
    'chin': ['right_chin_contour', 'left_chin_contour'],
    'jaw_contour': ['right_contour', 'left_contour',
                    'right_lip_contour', 'left_lip_contour'],
    'temples': ['left_temple', 'right_temple'],
}

FACE_Z_THRESHOLD = -20.0  # excludes back-of-head triangles


def load_subregion_indices(region_dir=REGION_DIR_DEFAULT):
    regions = {}
    for npy_file in sorted(Path(region_dir).glob("*.npy")):
        data = np.load(str(npy_file), allow_pickle=True).item()
        for subregion, indices in data.items():
            regions[subregion] = np.array(indices)
    return regions


def build_vertex_region_map(vertices, subregions):
    """Assign every mesh vertex to a high-level facial region via
    nearest-landmark Voronoi assignment with a distance threshold."""
    vertex_labels = {}
    for hl_name, sr_names in HIGH_LEVEL_REGIONS.items():
        for sr in sr_names:
            if sr in subregions:
                for vi in subregions[sr]:
                    vertex_labels[int(vi)] = hl_name

    labeled_indices = np.array(list(vertex_labels.keys()))
    labeled_positions = vertices[labeled_indices]

    lm_tree = cKDTree(labeled_positions)
    dd, _ = lm_tree.query(labeled_positions, k=2)
    max_dist = np.median(dd[:, 1]) * 4

    distances, nearest_idx = lm_tree.query(vertices)
    region_per_vertex = []
    for i in range(len(vertices)):
        if distances[i] > max_dist:
            region_per_vertex.append('other')
        else:
            region_per_vertex.append(vertex_labels[labeled_indices[nearest_idx[i]]])
    return region_per_vertex


def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def compute_region_areas(vertices, triangles, region_per_vertex):
    """Surface area per facial region (front-face triangles only)."""
    region_areas = defaultdict(float)
    total_face_area = 0.0
    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        if v0[2] < FACE_Z_THRESHOLD or v1[2] < FACE_Z_THRESHOLD or v2[2] < FACE_Z_THRESHOLD:
            continue
        a = triangle_area(v0, v1, v2)
        total_face_area += a
        r = Counter([region_per_vertex[tri[0]], region_per_vertex[tri[1]],
                     region_per_vertex[tri[2]]]).most_common(1)[0][0]
        if r != 'other':
            region_areas[r] += a
    return dict(region_areas), total_face_area


def build_mesh_adjacency(vertices, triangles):
    adj = defaultdict(list)
    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    vi, vj = tri[i], tri[j]
                    adj[vi].append((vj, float(np.linalg.norm(vertices[vi] - vertices[vj]))))
    return adj


def geodesic_distance(adj, start_vertex, end_vertex, num_vertices):
    """Shortest path distance along mesh edges (Dijkstra)."""
    dist = np.full(num_vertices, np.inf)
    dist[start_vertex] = 0.0
    visited = np.zeros(num_vertices, dtype=bool)
    heap = [(0.0, start_vertex)]
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if u == end_vertex:
            return d
        for v, w in adj[u]:
            if not visited[v] and d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return float('inf')


def compute_geodesic_ratios(vertices, triangles, lm_3d, lm_indices, subregions):
    """
    Geodesic/Euclidean ratios for key facial paths. Ratio > 1.0 means the
    surface path is longer than straight-line (surface relief/curvature).
    """
    adj = build_mesh_adjacency(vertices, triangles)
    n_verts = len(vertices)
    ratios = {}

    def find_landmark_vertex(target_subregions, selector='mean', restrict_to=None):
        """Get a vertex index from a set of sub-regions.

        restrict_to pins the returned vertex to one side of a mesh opening
        (eye slit, mouth slit): the combined centroid of both sides sits
        inside the opening, so without restriction the nearest vertex can
        land on either side depending on the subject's geometry, causing a
        discrete jump in geodesic-path length when it flips sides between
        subjects. Restricting the candidate pool to one side keeps the
        endpoint consistent while preserving the combined-centroid target.
        """
        all_vi = []
        for sr in target_subregions:
            if sr in subregions:
                all_vi.extend(subregions[sr].tolist())
        if not all_vi:
            return None
        pts = vertices[all_vi]
        if selector == 'mean':
            target_pos = pts.mean(axis=0)
            if restrict_to is not None:
                cand_vi = []
                for sr in restrict_to:
                    if sr in subregions:
                        cand_vi.extend(subregions[sr].tolist())
                if not cand_vi:
                    return None
            else:
                cand_vi = all_vi
            cand_pts = vertices[cand_vi]
            dists = np.linalg.norm(cand_pts - target_pos, axis=1)
            return cand_vi[np.argmin(dists)]
        elif selector == 'top':
            return all_vi[np.argmax(pts[:, 1])]
        elif selector == 'bottom':
            return all_vi[np.argmin(pts[:, 1])]
        elif selector == 'front':
            return all_vi[np.argmax(pts[:, 2])]
        return all_vi[0]

    # Path 1: nose bridge (top to tip)
    nose_top_vi = find_landmark_vertex(['nose_front_bridge'], 'top')
    nose_tip_vi = find_landmark_vertex(['nose_front_bridge'], 'front')
    if nose_top_vi is not None and nose_tip_vi is not None and nose_top_vi != nose_tip_vi:
        geo = geodesic_distance(adj, nose_top_vi, nose_tip_vi, n_verts)
        euc = np.linalg.norm(vertices[nose_top_vi] - vertices[nose_tip_vi])
        ratios['nose_bridge_geodesic_ratio'] = round(geo / euc, 4) if euc > 0 else float('nan')
        ratios['nose_bridge_geodesic_mm'] = round(geo, 2)
        ratios['nose_bridge_euclidean_mm'] = round(euc, 2)

    # Path 2: right eye center to mouth center
    r_eye_vi = find_landmark_vertex(['reye_top', 'reye_bottom'], 'mean', restrict_to=['reye_bottom'])
    mouth_vi = find_landmark_vertex(['ulipout', 'llipout'], 'mean', restrict_to=['ulipout'])
    if r_eye_vi is not None and mouth_vi is not None:
        geo = geodesic_distance(adj, r_eye_vi, mouth_vi, n_verts)
        euc = np.linalg.norm(vertices[r_eye_vi] - vertices[mouth_vi])
        ratios['eye_to_mouth_geodesic_ratio'] = round(geo / euc, 4) if euc > 0 else float('nan')

    # Path 3: across forehead (temple to temple)
    r_temple_vi = find_landmark_vertex(['right_temple'], 'mean')
    l_temple_vi = find_landmark_vertex(['left_temple'], 'mean')
    if r_temple_vi is not None and l_temple_vi is not None:
        geo = geodesic_distance(adj, r_temple_vi, l_temple_vi, n_verts)
        euc = np.linalg.norm(vertices[r_temple_vi] - vertices[l_temple_vi])
        ratios['forehead_geodesic_ratio'] = round(geo / euc, 4) if euc > 0 else float('nan')

    # Path 4: nose tip to chin (vertical mid-face path)
    chin_vi = find_landmark_vertex(['right_chin_contour', 'left_chin_contour'], 'bottom')
    if nose_tip_vi is not None and chin_vi is not None:
        geo = geodesic_distance(adj, nose_tip_vi, chin_vi, n_verts)
        euc = np.linalg.norm(vertices[nose_tip_vi] - vertices[chin_vi])
        ratios['nose_to_chin_geodesic_ratio'] = round(geo / euc, 4) if euc > 0 else float('nan')

    return ratios


def compute_angularity(vertices, triangles, region_per_vertex):
    """Mean dihedral angle deviation from 180 deg per region (sharper = more chiseled)."""
    tri_normals = {}
    for i, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm > 0:
            tri_normals[i] = n / norm

    edge_to_tris = defaultdict(list)
    for i, tri in enumerate(triangles):
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            edge = tuple(sorted([tri[a], tri[b]]))
            edge_to_tris[edge].append(i)

    region_angles = defaultdict(list)
    for edge, tri_ids in edge_to_tris.items():
        if len(tri_ids) != 2:
            continue
        if tri_ids[0] not in tri_normals or tri_ids[1] not in tri_normals:
            continue
        n1 = tri_normals[tri_ids[0]]
        n2 = tri_normals[tri_ids[1]]
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        dihedral_deviation = math.degrees(math.acos(cos_angle))
        r = Counter([region_per_vertex[edge[0]], region_per_vertex[edge[1]]]).most_common(1)[0][0]
        if r != 'other':
            region_angles[r].append(dihedral_deviation)

    return {r: round(float(np.mean(angles)), 4) for r, angles in region_angles.items()}


def compute_normal_variance(mesh, vertices, region_per_vertex):
    """Variance of surface normals per region (low = flat, high = sculpted)."""
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    region_normals = defaultdict(list)
    for i in range(len(vertices)):
        r = region_per_vertex[i]
        if r != 'other' and vertices[i][2] > FACE_Z_THRESHOLD:
            region_normals[r].append(normals[i])

    variance = {}
    for r, norms in region_normals.items():
        if len(norms) < 3:
            continue
        variance[r] = round(float(np.array(norms).var(axis=0).sum()), 6)
    return variance


def extract_surface_features(mesh, lm_3d, lm_indices, subregions):
    """Extract all surface features for one subject."""
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles)
    features = {}

    region_per_vertex = build_vertex_region_map(vertices, subregions)

    region_areas, total_face_area = compute_region_areas(vertices, triangles, region_per_vertex)
    assigned_area = sum(region_areas.values())
    features['total_face_area_mm2'] = round(total_face_area, 1)
    features['assigned_area_mm2'] = round(assigned_area, 1)

    for region_name in ['nose', 'eyes', 'lips', 'forehead', 'cheeks', 'chin', 'brows', 'temples']:
        area = region_areas.get(region_name, 0.0)
        features[f'{region_name}_area_mm2'] = round(area, 1)
        features[f'{region_name}_area_ratio'] = round(area / assigned_area, 4) if assigned_area > 0 else 0.0

    nose_a = region_areas.get('nose', 0.0)
    lip_a = region_areas.get('lips', 0.0)
    eye_a = region_areas.get('eyes', 0.0)
    features['nose_to_lip_area_ratio'] = round(nose_a / lip_a, 4) if lip_a > 0 else float('nan')
    features['eye_to_nose_area_ratio'] = round(eye_a / nose_a, 4) if nose_a > 0 else float('nan')

    geo_ratios = compute_geodesic_ratios(vertices, triangles, lm_3d, lm_indices, subregions)
    features.update(geo_ratios)

    angularity = compute_angularity(vertices, triangles, region_per_vertex)
    for region_name in ['jaw_contour', 'cheeks', 'nose', 'chin', 'brows']:
        features[f'{region_name}_angularity_deg'] = angularity.get(region_name, float('nan'))

    normal_var = compute_normal_variance(mesh, vertices, region_per_vertex)
    for region_name in ['cheeks', 'forehead', 'nose', 'chin']:
        features[f'{region_name}_normal_variance'] = normal_var.get(region_name, float('nan'))

    return features


def run(input_dir, output_dir, region_dir=REGION_DIR_DEFAULT):
    """input_dir: one subfolder per subject, each with mesh.ply +
    dad3dheads_landmarks_3d.npy + dad3dheads_landmark_indices.npy."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subregions = load_subregion_indices(region_dir)
    print(f"  {len(subregions)} sub-regions loaded")

    subject_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "mesh.ply").exists()
    ])
    if not subject_dirs:
        print(f"[ERROR] No subjects found in {input_dir}")
        return None
    print(f"  {len(subject_dirs)} subjects found")

    all_results = []
    for subj_dir in subject_dirs:
        subject = subj_dir.name
        lm_path = subj_dir / "dad3dheads_landmarks_3d.npy"
        idx_path = subj_dir / "dad3dheads_landmark_indices.npy"
        if not lm_path.exists() or not idx_path.exists():
            print(f"  [SKIP] {subject}: missing files")
            continue

        mesh = o3d.io.read_triangle_mesh(str(subj_dir / "mesh.ply"))
        lm_3d = np.load(str(lm_path))
        lm_indices = np.load(str(idx_path))

        try:
            features = extract_surface_features(mesh, lm_3d, lm_indices, subregions)
        except Exception as e:
            print(f"  [FAIL] {subject}: {e}")
            continue

        subj_out = output_dir / subject
        subj_out.mkdir(parents=True, exist_ok=True)
        with open(subj_out / "surface_features.json", "w") as f:
            json.dump({"subject": subject, "features": features}, f, indent=2)

        features["subject"] = subject
        all_results.append(features)
        print(f"  {subject}: OK")

    if not all_results:
        print("[ERROR] No subjects processed.")
        return None

    csv_path = output_dir / "surface_features_summary.csv"
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
