#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end facial-attractiveness feature extraction pipeline. Takes a
folder of face photos and produces a single ML-ready ml_features.csv,
mirroring the architecture in Figure 3 of the accompanying thesis:

  Photos ──┬──→ 2D landmarks ──→ 2D geometric features ──┐
           │                      + GPA averageness       │
           │                                              │
           ├──→ DECA 3D mesh ──→ DAD-3DHeads 445 lm ──┬──→ face shape    │
           │                                          ├──→ 3D depth     ├─→ Feature
           │                                          └──→ 3D surface   │  Aggregation
           │                                                            │  (ml_features.csv)
           └──→ derived (vision-model) features ─────────────────────────┘
                age, skin quality, femininity, neoteny, adiposity,
                hair colour/length, glasses, tattoo/piercing,
                composite femininity/neoteny (2D+3D combined)
                                      │
                                      ▼
                    (optional) Modeling & Explainability
                    Lasso / ElasticNet / XGBoost+SHAP / Cohen's d

LANDMARK SCHEME
───────────────
Pass --landmarks 68 or --landmarks 106 to choose the 2D landmark
detector. 106pt (InsightFace) consistently outperformed 68pt (FAN) in
testing and is the recommended default; 68pt is kept for comparison /
compatibility with studies that use the classic iBUG convention.

EXTERNAL DEPENDENCIES (not bundled — see setup/README.md)
───────────────────────────────────────────────────────────
  --deca-repo            clone of github.com/yfeng95/DECA (+ FLAME data)
  --face-parsing-ckpt    BiSeNet checkpoint (68pt landmarks only)
  --mivolo-repo          clone of github.com/WildChlamydia/MiVOLO
  --mivolo-detector-weights / --mivolo-checkpoint   MiVOLO model files
All CLIP/SigLIP2-based stages download their models automatically via
open_clip on first use (no separate setup needed).

Usage:
    python run_pipeline.py --input /path/to/photos --work-dir /path/to/work \
        --landmarks 106 --deca-repo /path/to/DECA \
        --mivolo-repo /path/to/MiVOLO \
        --mivolo-detector-weights /path/to/yolov8x_person_face.pt \
        --mivolo-checkpoint /path/to/model_imdb_cross_person_4.24_99.46.pth.tar

    # Feature extraction only, skip anything needing external repos:
    python run_pipeline.py --input photos/ --work-dir work/ --skip-3d --skip-age

    # Also run the modeling stage against a labels file:
    python run_pipeline.py --input photos/ --work-dir work/ --labels labels.txt
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR / "pipeline"

for sub in ["geometry_2d", "geometry_3d", "derived", "aggregate", "modeling"]:
    sys.path.insert(0, str(PIPELINE_DIR / sub))

# ── Stage modules ─────────────────────────────────────────────────────────────
import landmarks_68
import landmarks_106
import features_2d_68
import features_2d_106
import averageness_gpa

import deca_mesh
import dad_landmarks_445
import face_shape
import depth_curvature
import surface_geodesic

import age_mivolo
import skin_quality_clip
import femininity_clip
import neoteny_clip
import adiposity_clip
import glasses_clip
import tattoo_piercing_clip
import hair_color
import hair_length
import composite_femininity

from aggregate import aggregate, source


def banner(text):
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


def run_2d(photos_dir, work_dir, scheme, face_parsing_ckpt=None):
    banner(f"2D Geometry ({scheme}pt landmarks)")
    lm_dir = work_dir / f"landmarks_{scheme}pt"
    feat_dir = work_dir / f"features_2d_{scheme}pt"
    avg_dir = work_dir / "averageness"

    if scheme == 106:
        landmarks_106.run(photos_dir, lm_dir)
        features_2d_106.run(lm_dir, feat_dir, photos_dir=photos_dir)
    else:
        if not face_parsing_ckpt:
            print("  [ERROR] --face-parsing-ckpt is required for 68pt landmarks")
            return None, None
        landmarks_68.run(photos_dir, lm_dir, Path(face_parsing_ckpt))
        features_2d_68.run(lm_dir, feat_dir, photos_dir=photos_dir)

    averageness_gpa.run(lm_dir, avg_dir, scheme=scheme, photos_dir=photos_dir)
    return feat_dir, avg_dir


def run_3d(photos_dir, work_dir, deca_repo, device="cpu"):
    banner("3D Geometry (DECA)")
    mesh_dir = work_dir / "deca_meshes"
    dad_dir = work_dir / "dad_landmarks"
    shape_dir = work_dir / "face_shape"
    depth_dir = work_dir / "depth_curvature"
    surface_dir = work_dir / "surface_geodesic"

    mesh_results = deca_mesh.run(photos_dir, mesh_dir, deca_repo, device=device)
    if not mesh_results:
        print("  [WARN] No DECA meshes produced — skipping all 3D stages")
        return None, None, None

    dad_results = dad_landmarks_445.run_batch(mesh_results, dad_dir)

    # depth_curvature.py and surface_geodesic.py expect mesh.ply alongside
    # the DAD landmark files in the same per-subject folder.
    import shutil
    for subj, subj_dad_dir in dad_results.items():
        mesh_src = Path(mesh_results[subj]) / "mesh.ply"
        if mesh_src.exists():
            shutil.copy2(str(mesh_src), str(Path(subj_dad_dir) / "mesh.ply"))

    shape_input = work_dir / "_shape_input"
    for subj, subj_dad_dir in dad_results.items():
        subj_in = shape_input / subj
        subj_in.mkdir(parents=True, exist_ok=True)
        src = Path(subj_dad_dir) / "dad3dheads_landmarks_3d.npy"
        if src.exists():
            import shutil
            shutil.copy2(str(src), str(subj_in / "dad3dheads_landmarks_3d.npy"))
    face_shape.run_batch(str(shape_input), str(shape_dir))

    depth_curvature.run(dad_dir, depth_dir)
    surface_geodesic.run(dad_dir, surface_dir)

    return shape_dir, depth_dir, surface_dir


def run_derived(photos_dir, work_dir, mivolo_repo=None, mivolo_detector_weights=None,
                mivolo_checkpoint=None, device=None):
    banner("Derived (vision-model) features")
    results = {}

    if mivolo_repo and mivolo_detector_weights and mivolo_checkpoint:
        results["age"] = age_mivolo.run(
            photos_dir, work_dir / "age", mivolo_repo,
            mivolo_detector_weights, mivolo_checkpoint, device=device,
        )
    else:
        print("  [SKIP] Perceived age — MiVOLO repo/weights not provided")

    skin_quality_clip.analyze_folder(str(photos_dir), str(work_dir / "skin_quality"))
    results["skin"] = work_dir / "skin_quality" / "all_scores.csv"

    femininity_clip.analyze_folder(
        str(photos_dir), str(work_dir / "femininity_clip"),
        model_name="ViT-SO400M-14-SigLIP2", pretrained="webli",
    )
    results["fem_clip"] = work_dir / "femininity_clip" / "all_scores.csv"

    neoteny_clip.analyze_folder(str(photos_dir), str(work_dir / "neoteny_clip"))
    results["neoteny_clip"] = work_dir / "neoteny_clip" / "all_scores.csv"

    adiposity_clip.process_folder(str(photos_dir), str(work_dir / "adiposity"))
    results["adiposity"] = work_dir / "adiposity" / "all_scores.csv"

    # Glasses + tattoo/piercing are validated with SigLIP2 — override the
    # module's default (openai CLIP) before calling.
    glasses_clip._MODEL_NAME = "ViT-SO400M-14-SigLIP2"
    glasses_clip._PRETRAINED = "webli"
    glasses_clip.process_folder(str(photos_dir), str(work_dir / "glasses"))
    results["glasses"] = work_dir / "glasses" / "all_scores.csv"

    tattoo_piercing_clip._MODEL_NAME = "ViT-SO400M-14-SigLIP2"
    tattoo_piercing_clip._PRETRAINED = "webli"
    tattoo_piercing_clip.process_folder(str(photos_dir), str(work_dir / "tattoo_piercing"))
    results["tattoo"] = work_dir / "tattoo_piercing" / "all_scores.csv"

    hair_color.process_folder(str(photos_dir), str(work_dir / "hair_color"))
    results["hair_color"] = work_dir / "hair_color" / "all_scores.csv"

    hair_length.run(photos_dir, work_dir / "hair_length")
    results["hair_length"] = work_dir / "hair_length" / "all_scores.csv"

    return results


def run_composite(feat_2d_dir, depth_dir, surface_dir, work_dir):
    """Merge 2D + 3D feature JSONs per subject into one folder, then score."""
    banner("Composite femininity/neoteny (2D + 3D)")
    merged_dir = work_dir / "_composite_input"
    for src_dir, fname in [(feat_2d_dir, "2d_features.json"),
                           (depth_dir, "3d_features.json"),
                           (surface_dir, "surface_features.json")]:
        if not src_dir or not Path(src_dir).exists():
            continue
        for subj_dir in Path(src_dir).iterdir():
            if not subj_dir.is_dir():
                continue
            src = subj_dir / fname
            if src.exists():
                dst_dir = merged_dir / subj_dir.name
                dst_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(str(src), str(dst_dir / fname))

    composite_dir = work_dir / "composite"
    composite_femininity.run(merged_dir, composite_dir)
    return composite_dir


def run_aggregate(scheme, feat_2d_dir, avg_dir, shape_dir, depth_dir, surface_dir,
                  composite_dir, derived, output_csv):
    banner("Feature Aggregation")
    sources = [
        source("lm2d", feat_2d_dir / "2d_features_summary.csv", id_col="subject"),
        source("avg", avg_dir / "averageness_summary.csv", id_col="subject"),
    ]
    if shape_dir:
        sources.append(source("shape", shape_dir / "all_faces_summary.csv",
                              id_col="person_id", drop_cols=["conf_2nd_shape"]))
    if depth_dir:
        sources.append(source("depth3d", depth_dir / "3d_features_summary.csv", id_col="subject"))
    if surface_dir:
        sources.append(source("surf3d", surface_dir / "surface_features_summary.csv", id_col="subject"))
    if composite_dir:
        sources.append(source("fem", composite_dir / "composite_scores_summary.csv", id_col="subject",
                              drop_cols=["femininity_n_components", "neoteny_n_components"]))

    if derived.get("age"):
        sources.append(source("age", derived["age"], id_col="subject"))
    if derived.get("skin"):
        sources.append(source("skin", derived["skin"], id_col="image_id"))
    if derived.get("fem_clip"):
        sources.append(source("fem_clip", derived["fem_clip"], id_col="image_id"))
    if derived.get("neoteny_clip"):
        sources.append(source("neoteny_clip", derived["neoteny_clip"], id_col="image_id"))
    if derived.get("adiposity"):
        sources.append(source("adiposity", derived["adiposity"], id_col="image_id"))
    if derived.get("glasses"):
        sources.append(source("glasses", derived["glasses"], id_col="image_id"))
    if derived.get("tattoo"):
        sources.append(source("tattoo", derived["tattoo"], id_col="image_id"))
    if derived.get("hair_color"):
        sources.append(source("hair_color", derived["hair_color"], id_col="image_id"))
    if derived.get("hair_length"):
        sources.append(source("hair_length", derived["hair_length"], id_col="image_id"))

    return aggregate(sources, output_csv)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(SCRIPT_DIR / "input"),
                    help="Folder of face photos (.jpg) — default: ./input/")
    ap.add_argument("--work-dir", default=str(SCRIPT_DIR / "output"),
                    help="Where all intermediate + final output goes — default: ./output/")
    ap.add_argument("--landmarks", type=int, choices=[68, 106], default=106,
                    help="2D landmark scheme (default: 106, recommended)")
    ap.add_argument("--face-parsing-ckpt", default=None,
                    help="BiSeNet checkpoint path (required only for --landmarks 68)")
    ap.add_argument("--deca-repo", default=None, help="Path to a clone of github.com/yfeng95/DECA")
    ap.add_argument("--mivolo-repo", default=None, help="Path to a clone of github.com/WildChlamydia/MiVOLO")
    ap.add_argument("--mivolo-detector-weights", default=None)
    ap.add_argument("--mivolo-checkpoint", default=None)
    ap.add_argument("--device", default=None, help="'cpu' or 'cuda' (default: auto-detect per stage)")
    ap.add_argument("--skip-3d", action="store_true", help="Skip DECA/DAD/depth/surface/composite stages")
    ap.add_argument("--skip-derived", action="store_true", help="Skip all vision-model (CLIP/MiVOLO) stages")
    ap.add_argument("--labels", default=None,
                    help="Optional labels file (CSV or whitespace txt) — if given, runs the modeling stage")
    ap.add_argument("--feature-set", default="all",
                    choices=["all", "2d", "3d", "geometric", "appearance",
                             "2d+composite", "geometric+appearance"])
    args = ap.parse_args()

    photos_dir = Path(args.input)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    photos_dir.mkdir(parents=True, exist_ok=True)
    n_photos = len(list(photos_dir.glob("*.jpg"))) + len(list(photos_dir.glob("*.jpeg")))
    if n_photos == 0:
        print(f"[FATAL] No .jpg/.jpeg photos found in {photos_dir}")
        print(f"        Drop your photos in there (or pass --input /other/folder) and rerun.")
        return
    print(f"  Input: {photos_dir}  ({n_photos} photos)")
    print(f"  Output: {work_dir}")

    feat_2d_dir, avg_dir = run_2d(photos_dir, work_dir, args.landmarks, args.face_parsing_ckpt)
    if feat_2d_dir is None:
        print("[FATAL] 2D stage failed, aborting.")
        return

    shape_dir = depth_dir = surface_dir = composite_dir = None
    if not args.skip_3d:
        if not args.deca_repo:
            print("  [SKIP] 3D stage — pass --deca-repo to enable (or use --skip-3d to silence this)")
        else:
            shape_dir, depth_dir, surface_dir = run_3d(photos_dir, work_dir, args.deca_repo,
                                                        device=args.device or "cpu")
            if depth_dir:
                composite_dir = run_composite(feat_2d_dir, depth_dir, surface_dir, work_dir)

    derived = {}
    if not args.skip_derived:
        derived = run_derived(photos_dir, work_dir / "derived", args.mivolo_repo,
                              args.mivolo_detector_weights, args.mivolo_checkpoint,
                              device=args.device)

    output_csv = work_dir / "ml_features.csv"
    ml_features_csv = run_aggregate(args.landmarks, feat_2d_dir, avg_dir, shape_dir, depth_dir,
                                    surface_dir, composite_dir, derived, output_csv)

    if args.labels and ml_features_csv:
        banner("Modeling & Explainability")
        import subprocess
        modeling_dir = PIPELINE_DIR / "modeling"
        subprocess.run([
            sys.executable, str(modeling_dir / "run_all_models.py"),
            "--features-csv", str(ml_features_csv),
            "--labels", args.labels,
            "--feature-set", args.feature_set,
            "--output-dir", str(work_dir / "analysis"),
        ])

    banner("Done")
    print(f"  ml_features.csv: {output_csv}")


if __name__ == "__main__":
    main()
