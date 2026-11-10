#!/usr/bin/env python3
"""
hair_length.py
===============
Hair length classifier using CLIP zero-shot classification.
Categories: short / long / pulled-back (pulled-back is reported as "long"
in the output — pulled-back hair is long hair, just styled up).

DESIGN NOTES
────────────
- No cropping: the full image is passed to CLIP so it can see whether hair
  reaches the shoulders, rather than cropping tightly around the head
  (which often cuts off the evidence needed to judge length).
- Pulled-back hair (ponytail, bun, updo) is its own internal class with
  separate prompts, because forcing it into a binary short/long choice
  misclassified it as short (face fully exposed looks visually similar to
  a short haircut). It maps to "long" in the output label, but the
  internal class and score are also saved so borderline cases can be
  inspected.

Usage:
    python hair_length.py --input /path/to/photos --output /path/to/out
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import open_clip

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Three internal classes — pulled_back maps to "long" in the output
CATEGORIES = ["short", "long", "pulled_back"]

PROMPTS = {
    "short": [
        "a person with short hair that does not cover the neck",
        "a person with hair above the jawline",
        "a person with a short bob or pixie cut",
        "a person with short hair showing the full neck and ears",
        "a person with a neat short haircut framing the face",
        "a person with short hair where both ears are clearly visible",
        "a person with hair that does not reach the cheeks",
        "a person with short hair leaving the sides of the face fully exposed",
    ],
    "long": [
        "a person with hair covering the sides of the neck",
        "a person with long hair framing the face and falling past the chin",
        "a person with long hair visible beside the face",
        "a person with hair falling loosely around the shoulders",
        "a person with long flowing hair framing both sides of the face",
        "a person with hair falling on both sides of the face covering the cheeks",
        "a person with a centre part and hair framing both sides of the face",
        "a person with hair covering the sides of the face beside the ears",
    ],
    "pulled_back": [
        "a person with all their hair swept back from the face",
        "a person with hair pulled back into a ponytail",
        "a person with hair pulled back exposing the ears and neck completely",
        "a person with a sleek pulled-back hairstyle showing a clear forehead",
        "a person with hair tied back so the face is fully exposed",
        "a person with a bun or updo with no hair falling forward",
        "a person with long hair put up on top of the head",
        "a person with their hair pinned up leaving the neck and sides bare",
        "a person with an updo hairstyle with hair gathered on top",
        "a person with hair up showing a completely clear neck and shoulders",
    ],
}

OUTPUT_LABEL = {
    "short": "short",
    "long": "long",
    "pulled_back": "long",
}


def encode_prompts(model, tokenizer, device):
    category_embeddings = {}
    with torch.no_grad():
        for category, prompts in PROMPTS.items():
            tokens = tokenizer(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            avg = text_features.mean(dim=0)
            avg = avg / avg.norm()
            category_embeddings[category] = avg
    return torch.stack([category_embeddings[c] for c in CATEGORIES])


def classify(img_bgr, model, preprocess, text_features, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    image_input = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).squeeze(0)
        probs = logits.softmax(dim=-1).cpu().numpy()

    return probs


def run(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS)
    if not images:
        print(f"No images found in {input_dir}")
        return output_dir / "all_scores.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device} ...")

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    print("Encoding text prompts ...")
    text_features = encode_prompts(model, tokenizer, device)

    rows = []
    for i, path in enumerate(images, 1):
        img = cv2.imread(str(path))
        if img is None:
            print(f"{i}/{len(images)}: {path.name}  [unreadable]")
            continue

        probs = classify(img, model, preprocess, text_features, device)
        best_internal = CATEGORIES[int(np.argmax(probs))]
        output_label = OUTPUT_LABEL[best_internal]

        score_str = "  ".join(f"{c}={p:.2f}" for c, p in zip(CATEGORIES, probs))
        print(f"{i}/{len(images)}: {path.name}  ->  {output_label} ({best_internal})   [{score_str}]")

        rows.append({
            "image_id": path.stem,
            "hair_length": output_label,
            "internal": best_internal,
            "short": f"{probs[0]:.4f}",
            "long": f"{probs[1]:.4f}",
            "pulled_back": f"{probs[2]:.4f}",
        })

    out_path = output_dir / "all_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "hair_length", "internal", "short", "long", "pulled_back"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Folder of photos")
    ap.add_argument("--output", required=True, help="Output folder (writes all_scores.csv)")
    args = ap.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
