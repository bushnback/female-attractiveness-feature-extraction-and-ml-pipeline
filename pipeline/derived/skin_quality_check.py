"""
Quality flag assessment for CLIP pipeline.

CLIP itself doesn't need region masks (it scores the whole image), but we
use MediaPipe to populate the same `quality_flags` fields as the classical
pipeline so output is comparable.

If MediaPipe fails to detect a face, we still produce a valid result with
face_occluded=True and lighting_ok=False.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pathlib
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_CACHE = pathlib.Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"


def _ensure_model() -> str:
    if not _MODEL_CACHE.exists():
        _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe face landmarker model to {_MODEL_CACHE} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
    return str(_MODEL_CACHE)


# Face oval landmark indices (same as classical pipeline)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


@dataclass
class FaceQualityResult:
    lighting_ok: bool
    face_occluded: bool
    skin_area_pixels: int
    face_detected: bool


class FaceQualityChecker:
    """Lightweight MediaPipe wrapper just for quality flag assessment."""

    def __init__(self):
        model_path = _ensure_model()
        base_options = _mp_python.BaseOptions(model_asset_path=model_path)
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        self._detector = _mp_vision.FaceLandmarker.create_from_options(options)

    def __del__(self):
        try:
            self._detector.close()
        except Exception:
            pass

    def assess(self, image_bgr: np.ndarray) -> FaceQualityResult:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return FaceQualityResult(
                lighting_ok=False,
                face_occluded=True,
                skin_area_pixels=0,
                face_detected=False,
            )

        face = result.face_landmarks[0]
        landmarks = np.array(
            [[lm.x * w, lm.y * h] for lm in face],
            dtype=np.float32,
        )

        # Build face oval mask
        face_oval = self._polygon_mask(landmarks, FACE_OVAL, (h, w))

        # Build skin mask: face oval minus eyes, brows, lips
        skin = face_oval.copy()
        for idx_list in (LEFT_EYE_OUTLINE, RIGHT_EYE_OUTLINE,
                         LEFT_BROW, RIGHT_BROW, OUTER_LIP):
            exclude = self._polygon_mask(landmarks, idx_list, (h, w))
            exclude = cv2.dilate(exclude, np.ones((5, 5), np.uint8), iterations=1)
            skin = cv2.bitwise_and(skin, cv2.bitwise_not(exclude))

        skin_pixels = int((skin > 0).sum())
        face_pixels = int((face_oval > 0).sum())

        # Lighting check: mean L of skin
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        L = lab[..., 0][skin > 0]
        if len(L) > 0:
            mean_L = float(np.mean(L))
            lighting_ok = 70 <= mean_L <= 220
        else:
            lighting_ok = False

        # Occlusion: skin/face ratio
        face_occluded = (skin_pixels / face_pixels) < 0.4 if face_pixels > 0 else True

        return FaceQualityResult(
            lighting_ok=lighting_ok,
            face_occluded=face_occluded,
            skin_area_pixels=skin_pixels,
            face_detected=True,
        )

    @staticmethod
    def _polygon_mask(landmarks: np.ndarray, indices: list, shape: tuple) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        pts = landmarks[indices].astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask
