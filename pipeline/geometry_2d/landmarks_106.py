#!/usr/bin/env python3
"""
landmarks_106.py
================
Detect InsightFace 106-point 2D landmarks for every photo in a folder.

IMPORTANT — landmark scope (verified empirically): the InsightFace 106-point
scheme stops at the brow/temple line, same as a 68-point FAN model. It does
NOT reach the forehead or hairline.

INDEX MAP (InsightFace landmark_2d_106):
    0–32    Face contour (33 pts): 0 = chin (menton), 1/17 = temple endpoints
    33–42   Right eye (10 pts):    outer corner = 35, inner corner = 39
    43–51   Right eyebrow (9 pts)
    52–71   Mouth (20 pts):        outer + inner lip contour
    72–86   Nose (15 pts):         bridge, ala, tip
    87–96   Left eye (10 pts):     inner corner = 89, outer corner = 93
    97–105  Left eyebrow (9 pts)

Usage:
    python landmarks_106.py --input /path/to/photos --output /path/to/out
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis


def get_app(det_size=(320, 320)):
    """
    NOTE: a det_size much larger than the source image silently drops faces
    on InsightFace's SCRFD detector (e.g. (640,640) dropped ~50% of a
    350x350-photo dataset). Auto-size det_size to the dataset's actual photo
    dimensions via `auto_det_size()` rather than assuming a fixed value.
    """
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def auto_det_size(input_dir, sample_n=5, min_dim=320, max_dim=640):
    """Pick a det_size matching the dataset's actual image dimensions."""
    sample_imgs = sorted(Path(input_dir).glob("*.jpg"))[:sample_n] + \
                  sorted(Path(input_dir).glob("*.jpeg"))[:sample_n]
    largest = min_dim
    for p in sample_imgs:
        img = cv2.imread(str(p))
        if img is not None:
            largest = max(largest, img.shape[0], img.shape[1])
    size = min(int(np.ceil(largest / 32.0) * 32), max_dim)
    return (size, size)


def pick_largest_face(faces):
    def area(face):
        x1, y1, x2, y2 = face.bbox.astype(float)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)


def draw_overlay(image_bgr, landmarks_106):
    output = image_bgr.copy()
    for x, y in landmarks_106.astype(int):
        cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
    return output


def process_image(image_path, app, output_dir, save_overlay=False):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return "FAIL_READ"

    faces = app.get(image_bgr)
    if not faces:
        return "NO_FACE"

    face = pick_largest_face(faces)
    landmarks_106 = getattr(face, "landmark_2d_106", None)
    if landmarks_106 is None:
        return "NO_LANDMARKS"

    output_points = output_dir / f"{image_path.stem}_landmarks.npy"
    np.save(str(output_points), landmarks_106.astype(np.float32))

    if save_overlay:
        annotated = draw_overlay(image_bgr, landmarks_106)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_landmarks.jpg"), annotated)

    return "OK"


def run(input_dir, output_dir, save_overlays=False, det_size=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")))
    if not images:
        print(f"No JPEG images found in: {input_dir}")
        return {}

    if det_size is None:
        det_size = auto_det_size(input_dir)
    print(f"Found {len(images)} images")
    print(f"Input : {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"det_size: {det_size}")

    app = get_app(det_size)

    counts = {"OK": 0, "NO_FACE": 0, "NO_LANDMARKS": 0, "FAIL_READ": 0}
    results = {}
    for i, image_path in enumerate(images, 1):
        dst = output_dir / f"{image_path.stem}_landmarks.npy"
        if dst.exists():
            counts["OK"] += 1
            results[image_path.stem] = dst
        else:
            status = process_image(image_path, app, output_dir, save_overlays)
            counts[status] += 1
            if status == "OK":
                results[image_path.stem] = dst
            else:
                print(f"  {image_path.name}: {status}")
        if i % 200 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] OK={counts['OK']} "
                  f"NO_FACE={counts['NO_FACE']} NO_LM={counts['NO_LANDMARKS']} "
                  f"FAIL={counts['FAIL_READ']}")

    print("Done.", counts)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Folder of .jpg/.jpeg photos")
    ap.add_argument("--output", required=True, help="Output folder for *_landmarks.npy")
    ap.add_argument("--overlays", action="store_true", help="Also save numbered overlay JPEGs")
    args = ap.parse_args()
    run(args.input, args.output, save_overlays=args.overlays)


if __name__ == "__main__":
    main()
