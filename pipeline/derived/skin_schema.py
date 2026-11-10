"""
Output schema for skin quality analysis.

Identical to the classical and VLM pipelines. All three approaches produce
the same JSON structure so cross-experiment comparison is uniform.

Score convention: 0.0 = worst skin, 1.0 = best skin. Higher is always better.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class SkinFeatures:
    """Per-attribute skin quality scores. All in [0.0, 1.0], higher is better."""
    texture_smoothness: float = 0.0
    wrinkle_score: float = 0.0           # high = few wrinkles
    tone_evenness: float = 0.0
    redness_score: float = 0.0           # high = little redness
    pigmentation_score: float = 0.0      # high = no dark spots
    blemish_score: float = 0.0           # high = no blemishes
    under_eye_score: float = 0.0         # high = bright under-eye
    shine_score: float = 0.0             # high = balanced (not oily)
    overall_skin_quality: float = 0.0


@dataclass
class QualityFlags:
    lighting_ok: bool = True
    face_occluded: bool = False
    makeup_possible: bool = False
    skin_area_pixels: int = 0


@dataclass
class SkinAnalysisResult:
    image_id: str
    approach: str  # "classical" | "clip" | "vlm"
    skin_features: SkinFeatures = field(default_factory=SkinFeatures)
    quality_flags: QualityFlags = field(default_factory=QualityFlags)
    raw_features: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
