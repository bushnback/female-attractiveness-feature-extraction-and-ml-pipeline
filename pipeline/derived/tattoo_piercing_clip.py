"""
Tattoo and piercing detector using CLIP zero-shot contrastive prompting.

Produces TWO independent float scores per image:
  tattoo_score   — probability of visible tattoo ink (0 = none, 1 = heavy tattoos)
  piercing_score — probability of notable facial piercings (0 = none/plain ear studs,
                   1 = prominent/multiple facial piercings)

Algorithm:
  1. At module load, encode six prompt sets (3 grades x 2 dimensions) into centroids.
     Each centroid = mean of per-prompt L2-normalised embeddings, re-normalised.
     Centroids are cached for the lifetime of the process.
  2. Per image: encode once -> L2-normalise ->
       tattoo_score   : 3-way softmax -> p_grade2*0.5 + p_grade3*1.0
       piercing_score : 3-way softmax -> p_grade2*0.5 + p_grade3*1.0

Both dimensions use 3-grade. Earlier attempts at 3-grade piercings failed because
grade2 described piercings as "subtle" and "barely noticeable" — CLIP placed those
prompts at grade1 level, raising the floor for clean faces to ~0.19. The fix is to
describe grade2 as a clearly visible nose ring or lip ring (something CLIP can see),
not an invisible stud. Grade1 and grade2 also explicitly disambiguate metal piercings
from moles, freckles, and dimples.

What CLIP can see in a face-crop image:
  Visible: face/neck/chest tattoos, nose rings, lip rings, eyebrow piercings,
           gauges, septum piercings, cheek piercings
  NOT visible: body/arm tattoos outside the crop, plain small ear studs

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
            print(f"\n  \u2192 {selected['label']}\n")
            return selected
        print(f"  Invalid choice — enter one of: {', '.join(MODEL_VARIANTS.keys())}")

_device: torch.device | None = None
_model = None
_preprocess = None
_tokenizer = None
_tattoo_g1:   torch.Tensor | None = None
_tattoo_g2:   torch.Tensor | None = None
_tattoo_g3:   torch.Tensor | None = None
_piercing_g1: torch.Tensor | None = None
_piercing_g2: torch.Tensor | None = None
_piercing_g3: torch.Tensor | None = None


# ── Prompts ───────────────────────────────────────────────────────────────────

# ── Tattoo grades ─────────────────────────────────────────────────────────────

TATTOO_GRADE1 = [
    "a photo of a woman with no visible tattoos",
    "a portrait of a woman with clear natural unmarked skin and no tattoo ink",
    "a woman with completely clean skin — no tattoos visible anywhere",
]

TATTOO_GRADE2 = [
    "a photo of a woman with one small tattoo visible on her neck or chest",
    "a portrait of a woman with a small subtle tattoo — modest and easy to miss",
    "a woman with a single minor tattoo that is small and unobtrusive",
]

TATTOO_GRADE3 = [
    "a photo of a woman with heavy tattoos covering her neck and chest",
    "a portrait of a heavily tattooed woman with bold extensive tattoo artwork across her neck and chest",
    "a woman with extensive tattoo coverage on her visible skin — bold designs on neck, chest, and face",
]

# ── Piercing grades ────────────────────────────────────────────────────────────

PIERCING_GRADE1 = [
    "a photo of a woman with no rings or metal piercings on her face — earrings and necklaces are fine",
    "a portrait of a woman with no ring through her nose or lip — any dark spots are moles or freckles not metal",
    "a woman with no piercings on her nose, lips, eyebrows, or cheeks — no metal through her skin",
]

PIERCING_GRADE2 = [
    "a portrait of a woman with a single visible nose ring, not a mole or freckle",
    "a photo of a woman with one nose ring or lip ring clearly visible on her face",
    "a woman with a single nose ring or lip ring — one metal ring, nothing else pierced on her face",
]

PIERCING_GRADE3 = [
    "a photo of a woman with multiple piercings all over her face — rings through her nose and lip and a bar through her eyebrow",
    "a woman with big and obvious piercings on her face",
    "a portrait of a woman with big stretched holes in her earlobes and multiple rings pierced through her face",
    "a woman with rings or studs pierced through her nose, lip, eyebrow, and cheeks all at once",
]


# ── Initialisation ────────────────────────────────────────────────────────────

def _get_model():
    """Load model once and cache module-level globals."""
    global _device, _model, _preprocess, _tokenizer

    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        _MODEL_NAME, pretrained=_PRETRAINED, device=_device
    )
    _model.eval()
    _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)

    global _tattoo_g1, _tattoo_g2, _tattoo_g3
    global _piercing_g1, _piercing_g2, _piercing_g3

    _tattoo_g1   = _encode_centroid(TATTOO_GRADE1)
    _tattoo_g2   = _encode_centroid(TATTOO_GRADE2)
    _tattoo_g3   = _encode_centroid(TATTOO_GRADE3)
    _piercing_g1 = _encode_centroid(PIERCING_GRADE1)
    _piercing_g2 = _encode_centroid(PIERCING_GRADE2)
    _piercing_g3 = _encode_centroid(PIERCING_GRADE3)


@torch.inference_mode()
def _encode_centroid(prompts: list[str]) -> torch.Tensor:
    """Encode a list of prompts -> average L2-normalized embeddings -> renormalize."""
    tokens = _tokenizer(prompts).to(_device)
    embs   = _model.encode_text(tokens)
    embs   = embs / embs.norm(dim=-1, keepdim=True)
    centroid = embs.mean(dim=0)
    centroid = centroid / centroid.norm()
    return centroid


# ── Labelling ───────────────────────────────────────────────────────────────

def _label(score: float) -> str:
    """Convert continuous score to categorical label."""
    if score < 0.30:
        return "none"
    if score < 0.60:
        return "uncertain"
    return "present"


# ── Core scoring ─────────────────────────────────────────────────────────────

def _grade_score(img_emb: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor, g3: torch.Tensor) -> float:
    """3-grade softmax score. Returns p_grade2*0.5 + p_grade3*1.0."""
    s1, s2, s3 = float(img_emb @ g1), float(img_emb @ g2), float(img_emb @ g3)
    probs = torch.tensor([s1, s2, s3]).mul(100.0).softmax(dim=0)
    return float(probs[1] * 0.5 + probs[2] * 1.0)


def _load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


@torch.inference_mode()
def _encode_images(pil_images: list[Image.Image]) -> torch.Tensor:
    """Encode a batch of PIL images -> (N, D) L2-normalized tensor."""
    tensors = torch.stack([_preprocess(img) for img in pil_images]).to(_device)
    embs    = _model.encode_image(tensors)
    embs    = embs / embs.norm(dim=-1, keepdim=True)
    return embs


# ── Public API ────────────────────────────────────────────────────────────────

def score_image(image: Union[str, Path, Image.Image]) -> dict:
    """Return {"tattoo_score": float, "piercing_score": float}, each in [0, 1].

    tattoo_score   — 0.0 = clean skin, ~0.5 = one small tattoo, 1.0 = heavy tattoos.
    piercing_score — 0.0 = no facial piercings, ~0.5 = one subtle nose stud,
                     1.0 = multiple prominent facial piercings.
    combined_body_art_score — max(tattoo_score, piercing_score).
    """
    _get_model()
    pil     = _load_image(image)
    img_emb = _encode_images([pil])[0]

    tattoo_score   = _grade_score(img_emb, _tattoo_g1,   _tattoo_g2,   _tattoo_g3)
    piercing_score = _grade_score(img_emb, _piercing_g1, _piercing_g2, _piercing_g3)
    return {
        "tattoo_score":            tattoo_score,
        "piercing_score":          piercing_score,
        "combined_body_art_score": max(tattoo_score, piercing_score),
    }


def batch_score_images(images: list[Union[str, Path, Image.Image]]) -> list[dict]:
    """Score a batch of images in one CLIP forward pass.

    Returns list of {"tattoo_score": float, "piercing_score": float}.
    Score = p_grade2*0.5 + p_grade3*1.0 per dimension.
    """
    _get_model()
    pil_images = [_load_image(img) for img in images]
    img_embs   = _encode_images(pil_images)

    probs_t = torch.stack(
        [img_embs @ _tattoo_g1, img_embs @ _tattoo_g2, img_embs @ _tattoo_g3], dim=1
    ).mul(100.0).softmax(dim=1)
    probs_p = torch.stack(
        [img_embs @ _piercing_g1, img_embs @ _piercing_g2, img_embs @ _piercing_g3], dim=1
    ).mul(100.0).softmax(dim=1)

    tattoo_scores   = (probs_t[:, 1] * 0.5 + probs_t[:, 2] * 1.0).cpu().tolist()
    piercing_scores = (probs_p[:, 1] * 0.5 + probs_p[:, 2] * 1.0).cpu().tolist()
    return [
        {
            "tattoo_score":            t,
            "piercing_score":          p,
            "combined_body_art_score": max(t, p),
        }
        for t, p in zip(tattoo_scores, piercing_scores)
    ]


# ── Folder processing ─────────────────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def process_folder(
    folder:     Union[str, Path] = "images",
    output_dir: Union[str, Path] = "output",
    batch_size: int = 16,
) -> Path:
    """Score all images in `folder` and write results to `output_dir/all_scores.csv`."""
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

        score_dicts = batch_score_images(pil_images)

        for p, sd in zip(valid_paths, score_dicts):
            image_id = p.stem
            t = sd["tattoo_score"]
            ps = sd["piercing_score"]
            bar_t = "█" * int(t  * 20) + "░" * (20 - int(t  * 20))
            bar_p = "█" * int(ps * 20) + "░" * (20 - int(ps * 20))
            idx = valid_paths.index(p)
            print(
                f"  {i + idx + 1:>4}/{len(image_paths)}  {p.name:<36}  "
                f"T:[{bar_t}] {t:.4f}  P:[{bar_p}] {ps:.4f}"
            )
            rows.append({
                "image_id":               image_id,
                "tattoo_score":           f"{t:.6f}",
                "piercing_score":         f"{ps:.6f}",
                "combined_body_art_score": f"{max(t, ps):.6f}",
                "tattoo_label":           _label(t),
                "piercing_label":         _label(ps),
            })

    out_path = output_dir / "all_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "tattoo_score", "piercing_score", "combined_body_art_score", "tattoo_label", "piercing_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} scores -> {out_path}")

    panel_script = Path(__file__).parent / "make_panel.py"
    if panel_script.exists():
        spec = importlib.util.spec_from_file_location("make_panel", panel_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.make_panel(
            folder, out_path, output_dir,
            labels=("MOST TATTOOED", "SCORE 0.45–0.55", "LEAST TATTOOED"),
            score_col="tattoo_score",
            output_name="validation_panel_tattoo.jpg",
            middle_range=(0.45, 0.55),
        )
        mod.make_panel(
            folder, out_path, output_dir,
            labels=("MOST PIERCED", "SCORE 0.45–0.55", "LEAST PIERCED"),
            score_col="piercing_score",
            output_name="validation_panel_piercing.jpg",
            middle_range=(0.45, 0.55),
        )
        mod.make_panel(
            folder, out_path, output_dir,
            labels=("MOST BODY ART", "SCORE 0.45–0.55", "LEAST BODY ART"),
            score_col="combined_body_art_score",
            output_name="validation_panel_combined.jpg",
            middle_range=(0.45, 0.55),
        )
    else:
        print("(make_panel.py not found — skipping validation panels)")

    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "images"
    base_dir   = sys.argv[2] if len(sys.argv) > 2 else "output"

    variant = _prompt_model_selection()
    _MODEL_NAME = variant["model"]
    _PRETRAINED = variant["pretrained"]
    output_dir  = str(Path(base_dir) / variant["output_tag"])

    process_folder(images_dir, output_dir)
