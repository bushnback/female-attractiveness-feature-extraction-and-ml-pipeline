"""
Hair colour classifier using CLIP zero-shot classification.

Pipeline:
  1. Detect face with OpenCV Haar cascade to find a head crop
  2. Crop a generous head region around the face (includes hair above/beside)
  3. Run CLIP on the crop with prompt ensembling per colour category
  4. Output predicted label + softmax probabilities for all 4 categories

Usage:
    conda activate skin_clip
    python hair_color.py [images_folder] [output_folder]
"""

from __future__ import annotations
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
import open_clip


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CATEGORIES = ["black", "brown", "blonde", "red"]

# ── Model globals (lazy-loaded) ───────────────────────────────────────────────────

_MODEL_NAME = "ViT-L-14"
_PRETRAINED  = "openai"

_device        = None
_model         = None
_preprocess    = None
_tokenizer     = None
_text_features = None
_face_cascade  = None

# ── Model variants ────────────────────────────────────────────────────────────

MODEL_VARIANTS = {
    "1": {
        "label":      "ViT-L-14  |  OpenCLIP  |  openai  (current default)",
        "model":      "ViT-L-14",
        "pretrained": "openai",
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

# Multiple prompts per category - CLIP works better with prompt ensembling.
# Each prompt should describe the same concept in different natural phrasings.
PROMPTS = {
    "black": [
        "a photo of a person with black hair",
        "a portrait of someone with dark black hair",
        "a person with jet black hair",
    ],
    "brown": [
        "a photo of a person with brown hair",
        "a portrait of someone with brunette hair",
        "a person with dark brown hair",
        "a person with light brown hair",
    ],
    "blonde": [
        "a photo of a person with blonde hair",
        "a portrait of someone with blond hair",
        "a person with golden blonde hair",
        "a person with platinum blonde hair",
        "a person with light blonde hair",
    ],
    "red": [
        "a photo of a person with red hair",
        "a portrait of someone with ginger hair",
        "a person with auburn hair",
        "a redhead",
    ],
}


def make_head_crop(img_bgr, face_cascade):
    """Detect face and return a generous crop covering the head and hair.

    Falls back to the whole image if no face is found.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    if len(faces) == 0:
        return img_bgr

    # Take the largest face
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    # Expand generously: hair extends well above the face and often below the chin
    pad_top = int(fh * 1.3)     # lots of room above for hair on top of head
    pad_side = int(fw * 1.0)    # room beside for long hair
    pad_bottom = int(fh * 1.5)  # room below for shoulder-length hair

    x1 = max(0, x - pad_side)
    y1 = max(0, y - pad_top)
    x2 = min(w, x + fw + pad_side)
    y2 = min(h, y + fh + pad_bottom)

    return img_bgr[y1:y2, x1:x2]


def _get_model():
    global _device, _model, _preprocess, _tokenizer, _text_features, _face_cascade
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {_MODEL_NAME} ({_PRETRAINED}) on {_device} ...")
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED, device=_device
    )
    _model.eval()
    _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)

    print("Encoding text prompts ...")
    category_embeddings = {}
    with torch.inference_mode():
        for category, prompts in PROMPTS.items():
            tokens = tokenizer_encode(prompts)
            embs   = _model.encode_text(tokens)
            embs   = embs / embs.norm(dim=-1, keepdim=True)
            avg    = embs.mean(dim=0)
            category_embeddings[category] = avg / avg.norm()
    _text_features = torch.stack([category_embeddings[c] for c in CATEGORIES])

    _face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def tokenizer_encode(prompts):
    return _tokenizer(prompts).to(_device)


def _classify_image(pil_img: Image.Image) -> np.ndarray:
    image_input = _preprocess(pil_img).unsqueeze(0).to(_device)
    with torch.inference_mode():
        image_features = _model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ _text_features.T).squeeze(0)
        probs = logits.softmax(dim=-1).cpu().numpy()
    return probs


def process_folder(
    folder:     Union[str, Path] = "images",
    output_dir: Union[str, Path] = "output",
) -> Path:
    folder     = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    if not images:
        print(f"No images found in {folder}")
        return output_dir / "all_scores.csv"

    _get_model()

    rows = []
    for i, path in enumerate(images, 1):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"  [{i}/{len(images)}] {path.name}  [unreadable]")
            continue

        crop    = make_head_crop(img_bgr, _face_cascade)
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        probs   = _classify_image(Image.fromarray(img_rgb))

        best_idx    = int(np.argmax(probs))
        label       = CATEGORIES[best_idx]
        confidence  = float(probs[best_idx])
        score_str   = "  ".join(f"{c}={p:.2f}" for c, p in zip(CATEGORIES, probs))
        print(f"  [{i}/{len(images)}] {path.name}  ->  {label} ({confidence:.2f})   [{score_str}]")

        row = {"image_id": path.stem, "hair_color": label, "confidence": f"{confidence:.4f}"}
        for c, p in zip(CATEGORIES, probs):
            row[c] = f"{p:.4f}"
        rows.append(row)

    fieldnames = ["image_id", "hair_color", "confidence"] + CATEGORIES
    csv_path   = output_dir / "all_scores.csv"
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
        mod.make_panel(folder, csv_path, output_dir, label_col="hair_color", categories=CATEGORIES)
    else:
        print("(make_panel.py not found — skipping validation panel)")

    return csv_path


if __name__ == "__main__":
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "images"
    base_dir   = sys.argv[2] if len(sys.argv) > 2 else "output"

    variant    = _prompt_model_selection()
    _MODEL_NAME = variant["model"]
    _PRETRAINED = variant["pretrained"]
    output_dir  = str(Path(base_dir) / variant["output_tag"])

    process_folder(images_dir, output_dir)
