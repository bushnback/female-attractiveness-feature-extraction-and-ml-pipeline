"""
Facial fullness / adiposity scorer using CLIP zero-shot contrastive prompting.

Algorithm:
  1. At module load, encode FULL_PROMPTS and SLIM_PROMPTS into two centroid
     embeddings (average of per-prompt L2-normalized embeddings, then
     re-normalized).  These are cached for the lifetime of the process.
  2. Per image: encode once -> L2-normalize -> cosine similarity against both
     centroids -> softmax([sim_full, sim_slim] * 100.0) -> return P(full).

Score convention: 0.0 = very slim/narrow face, 1.0 = very full/round face.

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
_PRETRAINED = "openai"

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
            print(f"\n  \u2192 {selected['label']}\n")
            return selected
        print(f"  Invalid choice — enter one of: {', '.join(MODEL_VARIANTS.keys())}")

_device: torch.device | None = None
_model = None
_preprocess = None
_tokenizer = None
_full_centroid: torch.Tensor | None = None
_slim_centroid: torch.Tensor | None = None


# ── Prompts ───────────────────────────────────────────────────────────────────

FULL_PROMPTS = [
    "a photo of a chubby woman with a fat, round face",
    "a portrait of an overweight woman with chubby cheeks and a double chin",
    "a photo of a heavy-set woman with a puffy, fleshy face",
    "a portrait of a fat woman with a wide, bloated face and full cheeks",
    "a photo of an obese woman with very full, rounded facial features",
    "a woman with a chubby face, fat cheeks, and a soft undefined jawline",
    "a photo of a heavyset woman with a round, padded face and thick neck",
]

SLIM_PROMPTS = [
    "a photo of a slim woman with a lean, oval-shaped face",
    "a portrait of a thin woman with a narrow face and prominent cheekbones",
    "a photo of a slender woman with a delicate, fine-featured face",
    "a portrait of a skinny woman with hollow cheeks and a refined oval face",
    "a photo of a fit woman with a slim face, visible cheekbones, and a defined chin",
    "a woman with a slender face, high cheekbones, and a graceful jawline",
    "a photo of an underweight woman with a gaunt, thin face and sunken cheeks",
]


# ── Initialisation ────────────────────────────────────────────────────────────

def _get_model():
    """Load model once and cache module-level globals."""
    global _device, _model, _preprocess, _tokenizer
    global _full_centroid, _slim_centroid

    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED, device=_device
    )
    _model.eval()
    _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)

    _full_centroid = _encode_centroid(FULL_PROMPTS)
    _slim_centroid = _encode_centroid(SLIM_PROMPTS)


@torch.inference_mode()
def _encode_centroid(prompts: list[str]) -> torch.Tensor:
    """Encode a list of prompts -> average L2-normalized embeddings -> renormalize."""
    tokens = _tokenizer(prompts).to(_device)
    embs = _model.encode_text(tokens)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    centroid = embs.mean(dim=0)
    centroid = centroid / centroid.norm()
    return centroid


# ── Core scoring ──────────────────────────────────────────────────────────────

def _load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    """Accept a file path or PIL Image; always return a PIL Image."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


@torch.inference_mode()
def _encode_images(pil_images: list[Image.Image]) -> torch.Tensor:
    """Encode a batch of PIL images -> (N, D) L2-normalized tensor."""
    tensors = torch.stack([_preprocess(img) for img in pil_images]).to(_device)
    embs = _model.encode_image(tensors)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs


# ── Public API ────────────────────────────────────────────────────────────────

def score_facial_fullness(image: Union[str, Path, Image.Image]) -> float:
    """Return facial fullness score in [0, 1].

    0.0 = very slim / narrow face.
    1.0 = very full / round face.

    Parameters
    ----------
    image:
        A file path (str or Path) or a PIL Image.  Assumed to be already
        cropped to the face — no face detection is performed.

    Returns
    -------
    float
        Probability that the face is full/round rather than slim/narrow.
    """
    _get_model()
    pil = _load_image(image)
    img_emb = _encode_images([pil])[0]

    sim_full = float(img_emb @ _full_centroid)
    sim_slim = float(img_emb @ _slim_centroid)

    logits = torch.tensor([sim_full, sim_slim]) * 100.0
    probs = logits.softmax(dim=0)
    return float(probs[0].item())


def batch_score_facial_fullness(
    images: list[Union[str, Path, Image.Image]],
) -> list[float]:
    """Score a batch of images efficiently with a single CLIP image-encoding pass.

    Parameters
    ----------
    images:
        List of file paths (str / Path) or PIL Images.

    Returns
    -------
    list[float]
        Facial fullness scores in [0, 1], one per input image.
    """
    _get_model()
    pil_images = [_load_image(img) for img in images]
    img_embs = _encode_images(pil_images)

    sim_full = img_embs @ _full_centroid
    sim_slim = img_embs @ _slim_centroid

    logits = torch.stack([sim_full, sim_slim], dim=1) * 100.0
    probs = logits.softmax(dim=1)
    return probs[:, 0].cpu().tolist()


# ── Folder processing ────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def process_folder(
    folder: Union[str, Path] = "images",
    output_dir: Union[str, Path] = "output",
    batch_size: int = 16,
) -> Path:
    """Score all images in `folder` and write results to `output_dir/all_scores.csv`.

    Parameters
    ----------
    folder:
        Directory containing input images.  Defaults to ``images/``.
    output_dir:
        Directory where ``all_scores.csv`` is written.  Created if absent.
    batch_size:
        Number of images to encode in one CLIP forward pass.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    folder = Path(folder)
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
        pil_images = []
        valid_paths = []
        for p in batch_paths:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as exc:
                print(f"  [skip] {p.name}: {exc}")

        if not pil_images:
            continue

        scores = batch_score_facial_fullness(pil_images)

        for p, score in zip(valid_paths, scores):
            image_id = p.stem
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {i + valid_paths.index(p) + 1:>4}/{len(image_paths)}  {p.name:<40}  [{bar}]  {score:.4f}")
            rows.append({"image_id": image_id, "facial_fullness_score": f"{score:.6f}"})

    out_path = output_dir / "all_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "facial_fullness_score"])
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


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    images_dir = sys.argv[1] if len(sys.argv) > 1 else "images"
    base_dir   = sys.argv[2] if len(sys.argv) > 2 else "output"

    variant = _prompt_model_selection()
    _MODEL_NAME = variant["model"]
    _PRETRAINED = variant["pretrained"]
    output_dir  = str(Path(base_dir) / variant["output_tag"])

    process_folder(images_dir, output_dir)
