"""
Main CLIP pipeline: load image -> CLIP score + quality flags -> JSON.

Single-pass design (unlike classical): CLIP softmax probabilities are
already in [0, 1] so no batch normalization is needed. Each image is
scored independently.

Two entry points:
  - analyze_folder(folder, output_dir): batch process
  - analyze_image(path, scorer, checker): single-image API
"""

from __future__ import annotations
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from skin_schema import SkinAnalysisResult, SkinFeatures, QualityFlags
from skin_scorer import CLIPSkinScorer, DEFAULT_MODEL, DEFAULT_PRETRAINED
from skin_quality_check import FaceQualityChecker
from skin_prompts import ATTRIBUTE_ORDER

# ── Model variants ────────────────────────────────────────────────────────────
# Add new rows here to expose additional models in the interactive prompt.

MODEL_VARIANTS = {
    "1": {
        "label":      "ViT-L-14  |  OpenCLIP  |  laion2b_s32b_b82k  (current default)",
        "model":      "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "output_tag": "clip",
    },
    "2": {
        "label":      "ViT-SO400M-14-SigLIP2  |  Google SigLIP 2  |  webli  (experimental)",
        "model":      "ViT-SO400M-14-SigLIP2",
        "pretrained": "webli",
        "output_tag": "siglip2",
    },
}


def _prompt_model_selection() -> dict:
    print("\nSelect model variant:")
    for key, v in MODEL_VARIANTS.items():
        print(f"  {key}.  {v['label']}")
    print()
    while True:
        choice = input("Enter number: ").strip()
        if choice in MODEL_VARIANTS:
            selected = MODEL_VARIANTS[choice]
            print(f"\n  \u2192 {selected['label']}\n")
            return selected
        print(f"  Invalid choice — enter one of: {', '.join(MODEL_VARIANTS.keys())}")


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def analyze_image(image_path: str,
                  scorer: CLIPSkinScorer,
                  checker: FaceQualityChecker) -> dict:
    """
    Score a single image. Returns a dict matching the shared schema.
    """
    image_id = Path(image_path).stem

    # Load image (BGR for OpenCV/MediaPipe, also build PIL for CLIP)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {
            "image_id": image_id,
            "approach": "clip",
            "skin_features": asdict(SkinFeatures()),
            "quality_flags": asdict(QualityFlags(face_occluded=True, lighting_ok=False)),
            "raw_features": {},
            "notes": "Could not read image file.",
        }

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Quality flags via MediaPipe
    quality = checker.assess(image_bgr)
    flags = QualityFlags(
        lighting_ok=quality.lighting_ok,
        face_occluded=quality.face_occluded,
        makeup_possible=True,  # heuristic placeholder, same as classical
        skin_area_pixels=quality.skin_area_pixels,
    )

    notes = "" if quality.face_detected else "MediaPipe did not detect a face; CLIP scored the full image regardless."

    # CLIP scoring
    score_results = scorer.score_image(pil_image)

    # Build SkinFeatures from results
    skin_features = SkinFeatures(
        texture_smoothness=score_results["texture_smoothness"].score,
        wrinkle_score=score_results["wrinkle_score"].score,
        tone_evenness=score_results["tone_evenness"].score,
        redness_score=score_results["redness_score"].score,
        pigmentation_score=score_results["pigmentation_score"].score,
        blemish_score=score_results["blemish_score"].score,
        under_eye_score=score_results["under_eye_score"].score,
        shine_score=score_results["shine_score"].score,
        overall_skin_quality=score_results["overall_skin_quality"].score,
    )

    # Raw features: per-attribute similarity scores and probabilities
    raw_features = {
        "attribute_details": {
            attr: {
                "score": score_results[attr].score,
                "p_grade1": score_results[attr].p_grade1,
                "p_grade2": score_results[attr].p_grade2,
                "p_grade3": score_results[attr].p_grade3,
                "sim_grade1": score_results[attr].sim_grade1,
                "sim_grade2": score_results[attr].sim_grade2,
                "sim_grade3": score_results[attr].sim_grade3,
                "confidence": score_results[attr].confidence,
            }
            for attr in ATTRIBUTE_ORDER
        },
        "model": {
            "name": scorer.model.__class__.__name__,
            "device": scorer.device,
        },
    }

    return {
        "image_id": image_id,
        "approach": "clip",
        "skin_features": asdict(skin_features),
        "quality_flags": asdict(flags),
        "raw_features": raw_features,
        "notes": notes,
    }


def analyze_folder(folder: str, output_dir: str,
                   model_name: str = DEFAULT_MODEL,
                   pretrained: str = DEFAULT_PRETRAINED,
                   device: Optional[str] = None) -> pd.DataFrame:
    """
    Batch-process all images in `folder`. Single-pass design.
    Writes per-image JSON files and a combined CSV.

    Returns the combined DataFrame.
    """
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_paths:
        print(f"No images found in {folder}")
        return pd.DataFrame()

    print(f"[init] Loading CLIP model ({model_name} / {pretrained})...")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: running on CPU. Expect ~3-5s/image.")
    t0 = time.time()
    scorer = CLIPSkinScorer(model_name=model_name, pretrained=pretrained, device=device)
    print(f"[init] CLIP loaded on {scorer.device} ({time.time() - t0:.1f}s)")

    print("[init] Loading MediaPipe (may download model on first run)...")
    t1 = time.time()
    checker = FaceQualityChecker()
    print(f"[init] MediaPipe ready ({time.time() - t1:.1f}s)")

    total = len(image_paths)
    print(f"[processing] {total} images from {folder}")
    t_start = time.time()

    rows = []
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i}/{total}] {path.name}", end=" ", flush=True)
        t_img = time.time()
        try:
            result = analyze_image(str(path), scorer, checker)
        except Exception as e:
            print(f"-> error: {e}")
            continue

        with open(output_dir / f"{result['image_id']}.json", "w") as f:
            json.dump(result, f, indent=2)

        # Flatten for CSV
        row = {"image_id": result["image_id"]}
        row.update(result["skin_features"])
        rows.append(row)
        print(f"-> overall={result['skin_features']['overall_skin_quality']:.3f}  ({time.time() - t_img:.2f}s)")

    if not rows:
        print("No images successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("image_id")
    csv_path = output_dir / "all_scores.csv"
    df.to_csv(csv_path)
    print(f"\nSaved {len(df)} results to {output_dir}")
    print(f"Combined CSV: {csv_path}")

    panel_script = Path(__file__).parent / "make_panel.py"
    if panel_script.exists():
        spec = importlib.util.spec_from_file_location("make_panel", panel_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.make_all_panels(folder, csv_path, output_dir)
    else:
        print("(make_panel.py not found — skipping validation panel)")

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CLIP zero-shot skin quality analysis pipeline."
    )
    parser.add_argument("input_folder", nargs="?", default="images",
                        help="Folder of face images (default: ./images).")
    parser.add_argument("output_folder", nargs="?", default="output",
                        help="Where to write JSON + CSV (default: ./output).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenCLIP model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                        help=f"OpenCLIP pretrained tag (default: {DEFAULT_PRETRAINED}).")
    parser.add_argument("--device", default=None,
                        help="Device (cuda or cpu). Auto-detected if not set.")
    args = parser.parse_args()

    if not Path(args.input_folder).exists():
        print(f"Error: input folder '{args.input_folder}' does not exist.")
        print(f"Place face images in a folder called 'images' next to this script,")
        print(f"or pass a different folder as the first argument.")
        return

    variant    = _prompt_model_selection()
    output_dir = Path(args.output_folder) / variant["output_tag"]

    analyze_folder(args.input_folder, str(output_dir),
                   model_name=variant["model"],
                   pretrained=variant["pretrained"],
                   device=args.device)


if __name__ == "__main__":
    main()
