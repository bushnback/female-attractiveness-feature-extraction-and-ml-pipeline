#!/usr/bin/env python3
"""
age_mivolo.py
=============
Estimate perceived (apparent) age from face photos using MiVOLO
(Kuprashevich & Tolstykh, 2023, arXiv 2307.04616) — a vision-transformer
age/gender estimator using face + body context, with a YOLOv8 face/body
detector for crop extraction.

Perceived youthfulness is a robust attractiveness predictor independent
of actual age (Rhodes 2006; Fink et al. 2006; Ebner 2008).

REQUIRES (not bundled — see setup/README.md):
  - A clone of the MiVOLO repo (https://github.com/WildChlamydia/MiVOLO),
    importable as `mivolo` — pass its path via --mivolo-repo.
  - Two pretrained checkpoints, downloaded by setup/download_weights.py:
      yolov8x_person_face.pt
      model_imdb_cross_person_4.24_99.46.pth.tar

Usage:
    python age_mivolo.py --input /path/to/photos --output /path/to/out \
        --mivolo-repo /path/to/MiVOLO \
        --detector-weights /path/to/yolov8x_person_face.pt \
        --checkpoint /path/to/model_imdb_cross_person_4.24_99.46.pth.tar
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from argparse import Namespace

import cv2
import numpy as np
import torch

# PyTorch 2.11+ defaults weights_only=True in torch.load, which breaks
# the older ultralytics (8.1.0) used by MiVOLO. Monkey-patch to restore
# the old behaviour before importing any model code.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def age_to_group(age):
    if age < 18:
        return "under-18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    else:
        return "55+"


def get_predictor(mivolo_repo, detector_weights, checkpoint, device):
    sys.path.insert(0, str(Path(mivolo_repo).resolve()))
    from mivolo.predictor import Predictor

    config = Namespace(
        detector_weights=str(detector_weights),
        checkpoint=str(checkpoint),
        device=device,
        with_persons=True,
        disable_faces=False,
        draw=False,
    )
    return Predictor(config, verbose=True)


def estimate_age(image_path, predictor):
    """Run MiVOLO on a single image. Returns (result_dict, None) or (None, None)."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [ERROR] Cannot read image: {image_path.name}")
        return None

    detected_objects, _ = predictor.recognize(img)
    if detected_objects.n_faces == 0:
        print(f"  [ERROR] No face detected: {image_path.name}")
        return None

    face_inds = detected_objects.get_bboxes_inds("face")
    best_face_ind, best_area = None, 0
    for ind in face_inds:
        bbox = detected_objects.yolo_results.boxes[ind].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area, best_face_ind = area, ind

    if best_face_ind is None:
        print(f"  [ERROR] Could not identify best face: {image_path.name}")
        return None

    age = detected_objects.ages[best_face_ind]
    gender = detected_objects.genders[best_face_ind]
    gender_score = detected_objects.gender_scores[best_face_ind]
    if age is None:
        print(f"  [ERROR] No age estimate for face: {image_path.name}")
        return None

    return {
        "perceived_age": round(float(age), 1),
        "age_group": age_to_group(age),
        "gender": gender,
        "gender_score": round(float(gender_score), 4) if gender_score is not None else None,
    }


def run(input_dir, output_dir, mivolo_repo, detector_weights, checkpoint, device=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for p, name in [(detector_weights, "YOLOv8 detector"), (checkpoint, "MiVOLO checkpoint")]:
        if not Path(p).exists():
            print(f"[ERROR] {name} not found: {p}")
            return None

    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}")
        return None

    images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")))
    if not images:
        print(f"[ERROR] No images in {input_dir}")
        return None

    print(f"  Input: {input_dir.resolve()}  ({len(images)} images)")
    print(f"  Device: {device}")
    print("  Loading MiVOLO model...")
    predictor = get_predictor(mivolo_repo, detector_weights, checkpoint, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for image_path in images:
        subject_name = image_path.stem
        result = estimate_age(image_path, predictor)
        if result is None:
            continue
        print(f"  {subject_name}: age={result['perceived_age']:.1f} "
              f"group={result['age_group']} gender={result.get('gender')}")

        subject_out = {"subject": subject_name, "features": result}
        all_results.append(subject_out)

        subj_dir = output_dir / subject_name
        subj_dir.mkdir(parents=True, exist_ok=True)
        with open(subj_dir / "perceived_age.json", "w") as f:
            json.dump(subject_out, f, indent=2)

    if not all_results:
        print("[ERROR] No subjects processed successfully.")
        return None

    csv_path = output_dir / "perceived_age_summary.csv"
    fieldnames = ["subject", "perceived_age", "age_group", "gender"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "subject": r["subject"],
                "perceived_age": r["features"]["perceived_age"],
                "age_group": r["features"]["age_group"],
                "gender": r["features"].get("gender", ""),
            })

    ages = [r["features"]["perceived_age"] for r in all_results]
    print(f"\n  Subjects: {len(all_results)}  Mean age: {np.mean(ages):.1f}  "
          f"Range: {min(ages):.1f}-{max(ages):.1f}")
    print(f"  CSV: {csv_path.resolve()}")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mivolo-repo", required=True, help="Path to a clone of github.com/WildChlamydia/MiVOLO")
    ap.add_argument("--detector-weights", required=True, help="Path to yolov8x_person_face.pt")
    ap.add_argument("--checkpoint", required=True, help="Path to the MiVOLO age/gender checkpoint")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run(args.input, args.output, args.mivolo_repo, args.detector_weights, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
