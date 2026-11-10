#!/usr/bin/env python3
"""
dad_landmarks_445.py
=====================
Extract DAD-3DHeads' 445 semantic landmark positions from a FLAME-topology
mesh (works with DECA output, 5023 vertices / 9976 faces) by indexing
directly into the mesh's vertex array.

The region index files (dad_3dheads_keypoints/*.npy) are DAD-3DHeads'
own published region definitions (Martyniuk et al. 2022) — small index
arrays, not model weights, bundled alongside this script.

Requires: open3d (for mesh loading)

Usage:
    python dad_landmarks_445.py --mesh /path/to/mesh.ply --output /path/to/out
"""

import argparse
from pathlib import Path
import numpy as np

KEYPOINTS_DIR = Path(__file__).resolve().parent / "dad_3dheads_keypoints"

# "cheeks" region excluded — overlaps heavily with other regions and was
# found to add noise rather than signal in earlier experiments.
EXCLUDE_REGIONS = {"cheeks"}


def load_landmark_indices(keypoints_dir=KEYPOINTS_DIR, exclude_regions=EXCLUDE_REGIONS):
    all_indices = []
    for npy_file in sorted(Path(keypoints_dir).glob("*.npy")):
        if npy_file.stem in exclude_regions:
            continue
        data = np.load(str(npy_file), allow_pickle=True).item()
        for subregion, indices in data.items():
            all_indices.extend(indices.tolist())
    return sorted(set(all_indices))


def extract_landmarks(mesh_path, indices):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)

    max_idx = max(indices)
    if max_idx >= len(vertices):
        raise ValueError(f"max index {max_idx} >= {len(vertices)} vertices in mesh")

    return vertices[indices]


def run(mesh_path, output_dir, keypoints_dir=KEYPOINTS_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = load_landmark_indices(keypoints_dir)
    print(f"  {len(indices)} unique landmark indices loaded")

    landmarks_3d = extract_landmarks(mesh_path, indices)
    np.save(str(output_dir / "dad3dheads_landmarks_3d.npy"), landmarks_3d)
    np.save(str(output_dir / "dad3dheads_landmark_indices.npy"), np.array(indices))
    print(f"  Extracted {len(landmarks_3d)} landmarks -> {output_dir}")
    return output_dir


def run_batch(subjects_with_mesh, output_dir, keypoints_dir=KEYPOINTS_DIR):
    """subjects_with_mesh: {subject_id: path_to_subject_dir_containing_mesh.ply}"""
    output_dir = Path(output_dir)
    indices = load_landmark_indices(keypoints_dir)
    print(f"  {len(indices)} unique landmark indices loaded")

    results = {}
    for subj, mesh_dir in subjects_with_mesh.items():
        subj_out = output_dir / subj
        subj_out.mkdir(parents=True, exist_ok=True)
        try:
            landmarks_3d = extract_landmarks(Path(mesh_dir) / "mesh.ply", indices)
            np.save(str(subj_out / "dad3dheads_landmarks_3d.npy"), landmarks_3d)
            np.save(str(subj_out / "dad3dheads_landmark_indices.npy"), np.array(indices))
            results[subj] = subj_out
            print(f"  {subj}: extracted {len(landmarks_3d)} landmarks")
        except Exception as e:
            print(f"  {subj}: SKIP ({e})")

    print(f"  DAD landmarks available: {len(results)}/{len(subjects_with_mesh)}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", required=True, help="Path to a single mesh.ply")
    ap.add_argument("--output", required=True, help="Output folder")
    args = ap.parse_args()
    run(args.mesh, args.output)


if __name__ == "__main__":
    main()
