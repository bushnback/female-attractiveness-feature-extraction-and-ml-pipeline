"""
Neoteny / cuteness CLIP pipeline.

Encodes each image once then scores all 5 neoteny attributes via pre-computed
prompt centroids. Writes all_scores.csv and prints a progress bar keyed to
overall_cuteness. Results are saved to a subdirectory named after the model
variant so runs never overwrite each other.

Usage:
    conda activate skin_clip
    python pipeline.py [images_folder] [output_folder]

Default images_folder : images/
Default output_folder : output/
"""

from __future__ import annotations
import csv
import importlib.util
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from neoteny_scorer import CLIPNeotenySscorer, DEFAULT_MODEL, DEFAULT_PRETRAINED
from neoteny_prompts import ATTRIBUTE_ORDER


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_VARIANTS = {
    "1": {
        "label":      "ViT-L-14  |  OpenCLIP  |  laion2b_s32b_b82k  (default)",
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
            print(f"\n  → {selected['label']}\n")
            return selected
        print(f"  Invalid choice — enter one of: {', '.join(MODEL_VARIANTS.keys())}")


def analyze_folder(
    folder:     str = "images",
    output_dir: str = "output",
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    device:     Optional[str] = None,
) -> Path:
    folder     = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {folder}")
        return output_dir / "all_scores.csv"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[init] Loading CLIP ({model_name} / {pretrained}) on {device} ...")
    t0     = time.time()
    scorer = CLIPNeotenySscorer(model_name=model_name, pretrained=pretrained, device=device)
    print(f"[init] Ready ({time.time() - t0:.1f}s)")

    total = len(image_paths)
    print(f"[processing] {total} images from {folder}")

    rows = []
    for i, path in enumerate(image_paths, 1):
        print(f"  [{i}/{total}] {path.name}", end=" ", flush=True)
        t_img = time.time()
        try:
            pil    = Image.open(path).convert("RGB")
            scores = scorer.score_image(pil)
        except Exception as e:
            print(f"-> error: {e}")
            continue

        overall = scores.get("overall_babyface", 0.0)
        bar = "█" * int(overall * 20) + "░" * (20 - int(overall * 20))
        print(f"  [{bar}]  babyface={overall*100:5.1f}%  ({time.time() - t_img:.2f}s)")

        row = {"image_id": path.stem}
        row.update({attr: f"{scores[attr]:.6f}" for attr in ATTRIBUTE_ORDER})
        rows.append(row)

    if not rows:
        print("No images successfully processed.")
        return output_dir / "all_scores.csv"

    csv_path   = output_dir / "all_scores.csv"
    fieldnames = ["image_id"] + ATTRIBUTE_ORDER
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows -> {csv_path}")

    panel_script = Path(__file__).parent / "make_panel.py"
    if panel_script.exists():
        spec = importlib.util.spec_from_file_location("make_panel", panel_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.make_panel(
            images_dir  = folder,
            scores_csv  = csv_path,
            output_dir  = output_dir,
            score_col   = "overall_babyface",
            labels      = ("MOST BABYFACED", "MIDDLE", "LEAST BABYFACED"),
            output_name = "validation_panel.jpg",
        )
    else:
        print("(make_panel.py not found — skipping panel)")

    return csv_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLIP neoteny / cuteness scorer")
    parser.add_argument("input_folder",  nargs="?", default="images")
    parser.add_argument("output_folder", nargs="?", default="output")
    parser.add_argument("--device",  default=None)
    parser.add_argument("--variant", default=None, choices=list(MODEL_VARIANTS.keys()),
                        help="Model variant number (skips interactive prompt)")
    parser.add_argument("--attempt", default=None,
                        help="Attempt label, e.g. attempt1 (creates a subdirectory)")
    args = parser.parse_args()

    if not Path(args.input_folder).exists():
        print(f"Error: '{args.input_folder}' does not exist.")
        return

    variant    = MODEL_VARIANTS[args.variant] if args.variant else _prompt_model_selection()
    output_dir = Path(args.output_folder) / variant["output_tag"]
    if args.attempt:
        output_dir = output_dir / args.attempt

    analyze_folder(
        args.input_folder,
        str(output_dir),
        model_name=variant["model"],
        pretrained=variant["pretrained"],
        device=args.device,
    )


if __name__ == "__main__":
    main()
