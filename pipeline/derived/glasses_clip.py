"""
Glasses detector using CLIP zero-shot contrastive prompting.

Algorithm:
  1. At module load, encode GLASSES_PROMPTS and NO_GLASSES_PROMPTS into two
     centroid embeddings (average of per-prompt L2-normalised embeddings, then
     re-normalised). Cached for the lifetime of the process.
  2. Per image: encode once → L2-normalise → cosine similarity against both
     centroids → softmax([sim_glasses, sim_no_glasses] * 100.0) → P(glasses).

Score convention:
  0.0 = no glasses
  1.0 = wearing glasses

What CLIP can see in a face-crop image:
  Detectable: prescription glasses, sunglasses, reading glasses, thick-framed
              glasses, thin-framed glasses, tinted lenses
  NOT detectable: contact lenses, glasses resting on head or hanging from collar

Dependencies: open_clip_torch, torch, Pillow, numpy
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
import open_clip


# ── Model ─────────────────────────────────────────────────────────────────────

_MODEL_NAME = "ViT-L-14"
_PRETRAINED  = "openai"

# ── Model variants ────────────────────────────────────────────────────────────
# Add new rows here to expose additional models in the interactive prompt.

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

_device: torch.device | None = None
_model       = None
_preprocess  = None
_tokenizer   = None
_glasses_centroid:    torch.Tensor | None = None
_no_glasses_centroid: torch.Tensor | None = None

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_SCORE_COL    = "glasses_score"


# ── Prompts ───────────────────────────────────────────────────────────────────

GLASSES_PROMPTS = [
    "a photo of a woman wearing glasses",
    "a portrait of a woman with eyeglasses on her face",
    "a woman wearing prescription spectacles",
    "a photo of a woman with thick-framed glasses",
    "a portrait of a woman wearing clear-lens glasses",
    "a woman with glasses sitting on her nose",
    "a photo of a woman wearing rimless or thin-framed glasses",
]

NO_GLASSES_PROMPTS = [
    "a photo of a woman not wearing glasses",
    "a portrait of a woman with no eyewear on her face",
    "a woman with bare eyes and no glasses",
    "a photo of a woman with no spectacles or eyewear",
    "a portrait of a woman whose eyes are unobstructed by glasses",
    "a woman without glasses showing her natural eyes clearly",
    "a photo of a woman with no frames or lenses in front of her eyes",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_model():
    global _device, _model, _preprocess, _tokenizer
    global _glasses_centroid, _no_glasses_centroid

    if _model is not None:
        return

    _device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED, device=_device
    )
    _model.eval()
    _tokenizer  = open_clip.get_tokenizer(_MODEL_NAME)

    def _encode_prompts(prompts: list[str]) -> torch.Tensor:
        tokens = _tokenizer(prompts).to(_device)
        with torch.inference_mode():
            embs = _model.encode_text(tokens)
        embs   = embs / embs.norm(dim=-1, keepdim=True)
        centroid = embs.mean(dim=0)
        return centroid / centroid.norm()

    _glasses_centroid    = _encode_prompts(GLASSES_PROMPTS)
    _no_glasses_centroid = _encode_prompts(NO_GLASSES_PROMPTS)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_glasses(pil_image: Image.Image) -> float:
    """Return P(glasses) in [0, 1]. 1.0 = wearing glasses."""
    _get_model()
    tensor = _preprocess(pil_image).unsqueeze(0).to(_device)
    with torch.inference_mode():
        emb = _model.encode_image(tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb[0]

    sim_glasses    = float((emb @ _glasses_centroid).item())
    sim_no_glasses = float((emb @ _no_glasses_centroid).item())

    logits = torch.tensor([sim_glasses, sim_no_glasses]) * 100.0
    probs  = logits.softmax(dim=0)
    return float(probs[0].item())


def batch_score_glasses(pil_images: list[Image.Image]) -> list[float]:
    _get_model()
    tensors = torch.stack([_preprocess(img) for img in pil_images]).to(_device)
    with torch.inference_mode():
        embs = _model.encode_image(tensors)
    embs = embs / embs.norm(dim=-1, keepdim=True)

    sim_glasses    = (embs @ _glasses_centroid).unsqueeze(1)
    sim_no_glasses = (embs @ _no_glasses_centroid).unsqueeze(1)
    logits = torch.cat([sim_glasses, sim_no_glasses], dim=1) * 100.0
    probs  = logits.softmax(dim=1)
    return probs[:, 0].tolist()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_folder(
    folder:     Union[str, Path] = "images",
    output_dir: Union[str, Path] = "output",
    batch_size: int = 16,
) -> Path:
    folder     = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {folder}")
        return output_dir / "all_scores.csv"

    print(f"Found {len(image_paths)} images in {folder}")
    print(f"Loading {_MODEL_NAME} ({_PRETRAINED}) ...")
    _get_model()

    rows = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        pil_images, valid_paths = [], []
        for p in batch_paths:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as exc:
                print(f"  [skip] {p.name}: {exc}")

        if not pil_images:
            continue

        scores = batch_score_glasses(pil_images)

        for p, score in zip(valid_paths, scores):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            j   = i + valid_paths.index(p) + 1
            print(f"  {j:>4}/{len(image_paths)}  {p.name:<40}  [{bar}]  {score:.4f}")
            rows.append({"image_id": p.stem, CSV_SCORE_COL: f"{score:.6f}"})

    out_path = output_dir / "all_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", CSV_SCORE_COL])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} scores -> {out_path}")

    panel_script = Path(__file__).parent / "make_panel.py"
    if panel_script.exists():
        spec = importlib.util.spec_from_file_location("make_panel", panel_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.make_panel(folder, out_path, output_dir)
    else:
        print("(make_panel.py not found — skipping validation panel)")

    return out_path


if __name__ == "__main__":
    import sys
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "images"
    base_dir   = sys.argv[2] if len(sys.argv) > 2 else "output"

    variant = _prompt_model_selection()
    _MODEL_NAME = variant["model"]
    _PRETRAINED = variant["pretrained"]
    output_dir  = str(Path(base_dir) / variant["output_tag"])

    process_folder(images_dir, output_dir)
