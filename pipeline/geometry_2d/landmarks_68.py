#!/usr/bin/env python3
"""
landmarks_68.py
================
Detect 68-point 2D landmarks (FAN, via the `face_alignment` library),
refining the jawline (points 0-16) against a BiSeNet face-parsing mask so it
follows the visible skin boundary rather than FAN's raw (often slightly
off) prediction.

Requires a BiSeNet face-parsing checkpoint (see setup/download_weights.py)
and the `face_parsing` CLI tool it provides on PATH.

Usage:
    python landmarks_68.py --input /path/to/photos --output /path/to/out \
        --face-parsing-ckpt /path/to/79999_iter.pth
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
import face_alignment
from insightface.app import FaceAnalysis


def get_detector():
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_landmark_model():
    return face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device="cpu",
    )


def pick_largest_face(faces):
    def area(face):
        x1, y1, x2, y2 = face.bbox.astype(float)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)


def expand_bbox(bbox, image_shape, scale=0.15):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * scale
    pad_y = bh * scale
    nx1 = max(0, int(x1 - pad_x))
    ny1 = max(0, int(y1 - pad_y))
    nx2 = min(w, int(x2 + pad_x))
    ny2 = min(h, int(y2 + pad_y))
    return nx1, ny1, nx2, ny2


def run_face_parsing(crop_image_path, result_dir, work_dir, ckpt):
    input_dir = work_dir / "tmp_face_parsing_input"

    if input_dir.exists():
        shutil.rmtree(input_dir)
    if result_dir.exists():
        shutil.rmtree(result_dir)

    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    input_image = input_dir / "tmp_crop.jpg"

    img = cv2.imread(str(crop_image_path))
    if img is None:
        raise RuntimeError(f"Could not read crop image: {crop_image_path}")

    cv2.imwrite(str(input_image), img)

    cmd = [
        "face_parsing",
        "--img_path", str(input_dir),
        "--res_path", str(result_dir),
        "--ckpt", str(ckpt),
        "--color_style", "face-parsing-style",
    ]
    subprocess.run(cmd, check=True)

    candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        candidates.extend(result_dir.glob(pattern))

    if not candidates:
        raise RuntimeError(f"No mask image produced in: {result_dir}")

    preferred = [p for p in candidates if p.stem.startswith("tmp_crop")]
    return preferred[0] if preferred else candidates[0]


def refine_jawline(landmarks, mask):
    """Snap jawline points (0-16) to the nearest in-mask pixel if they land
    outside the BiSeNet face-region mask."""
    refined = landmarks.copy()
    h, w = mask.shape[:2]

    for i in range(17):
        x, y = refined[i]
        xi = int(round(x))
        yi = int(round(y))
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)

        if mask[yi, xi] > 0:
            continue

        radius = 15
        best = None
        best_d = None
        for yy in range(max(0, yi - radius), min(h, yi + radius + 1)):
            for xx in range(max(0, xi - radius), min(w, xi + radius + 1)):
                if mask[yy, xx] > 0:
                    d = (xx - xi) ** 2 + (yy - yi) ** 2
                    if best_d is None or d < best_d:
                        best_d = d
                        best = (xx, yy)
        if best is not None:
            refined[i] = best

    return refined


def draw_overlay(image, detector_pts, landmarks):
    out = image.copy()
    if detector_pts is not None:
        for x, y in detector_pts.astype(int):
            cv2.circle(out, (x, y), 3, (0, 255, 255), -1)
    for x, y in landmarks.astype(int):
        cv2.circle(out, (x, y), 2, (0, 255, 0), -1)
    return out


def process_image(path, detector, landmark_model, output_dir, face_parsing_ckpt, save_overlay=True):
    img = cv2.imread(str(path))
    if img is None:
        print("FAILED TO READ:", path.name)
        return False

    faces = detector.get(img)
    if not faces:
        print("NO FACE:", path.name)
        return False

    face = pick_largest_face(faces)
    crop_box = expand_bbox(face.bbox, img.shape)
    x1, y1, x2, y2 = crop_box
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        print("EMPTY CROP:", path.name)
        return False

    crop_path = output_dir / "tmp_crop.jpg"
    mask_dir = output_dir / "tmp_face_parsing_output"
    cv2.imwrite(str(crop_path), crop)

    try:
        mask_path = run_face_parsing(crop_path, mask_dir, output_dir, face_parsing_ckpt)
    except Exception as e:
        print(f"FACE PARSING FAILED: {path.name} ({e})")
        return False

    mask_img = cv2.imread(str(mask_path))
    if mask_img is None:
        print("NO MASK:", path.name)
        return False

    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) if mask_img.ndim == 3 else mask_img
    _, mask = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    preds = landmark_model.get_landmarks(crop_rgb)
    if preds is None or len(preds) == 0:
        print("NO LANDMARKS:", path.name)
        return False

    landmarks = preds[0].astype(np.float32)
    landmarks = refine_jawline(landmarks, mask)
    landmarks[:, 0] += x1
    landmarks[:, 1] += y1

    out_npy = output_dir / f"{path.stem}_landmarks.npy"
    np.save(str(out_npy), landmarks)

    if save_overlay:
        detector_pts = getattr(face, "kps", None)
        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        resized_mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        full_mask[y1:y2, x1:x2] = resized_mask
        overlay = draw_overlay(img, detector_pts, landmarks)
        cv2.imwrite(str(output_dir / f"{path.stem}_landmarks.jpg"), overlay)

    print("OK:", path.name)
    return True


def run(input_dir, output_dir, face_parsing_ckpt, save_overlays=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print("Input folder not found:", input_dir)
        return {}

    detector = get_detector()
    landmark_model = get_landmark_model()

    images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")))
    print("Found", len(images), "images")

    results = {}
    for img in images:
        dst = output_dir / f"{img.stem}_landmarks.npy"
        if dst.exists():
            results[img.stem] = dst
            continue
        if process_image(img, detector, landmark_model, output_dir, face_parsing_ckpt, save_overlays):
            results[img.stem] = dst

    print("Done")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Folder of .jpg/.jpeg photos")
    ap.add_argument("--output", required=True, help="Output folder for *_landmarks.npy")
    ap.add_argument("--face-parsing-ckpt", required=True,
                    help="Path to BiSeNet face-parsing checkpoint (79999_iter.pth)")
    ap.add_argument("--overlays", action="store_true", help="Also save overlay JPEGs")
    args = ap.parse_args()
    run(args.input, args.output, Path(args.face_parsing_ckpt), args.overlays)


if __name__ == "__main__":
    main()
