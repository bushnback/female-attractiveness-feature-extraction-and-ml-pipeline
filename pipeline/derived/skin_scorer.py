"""
CLIP-based skin attribute scorer.

Uses OpenCLIP (ViT-L/14 trained on LAION-2B) for zero-shot scoring.
For each attribute:
  1. Encode all "good" prompts and average their text embeddings -> good_emb
  2. Encode all "bad" prompts and average their text embeddings -> bad_emb
  3. Encode the image -> image_emb
  4. Compute cosine similarity image_emb vs good_emb and image_emb vs bad_emb
  5. Apply softmax with CLIP's logit_scale -> probability of "good"
  6. Score = P(good) in [0, 1]

Text embeddings are computed once at construction time and cached.
Each image then needs only one image encoding pass + 9 attribute scores
(simple matrix ops).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import open_clip

from skin_prompts import GRADE_PROMPTS, ATTRIBUTE_ORDER, get_all_prompts, get_prompt_indices


# Default model. ViT-L/14 LAION-2B is a strong fine-grained CLIP variant.
DEFAULT_MODEL = "ViT-L-14"
DEFAULT_PRETRAINED = "laion2b_s32b_b82k"
# Higher-res alternative: --model ViT-L-14-336 --pretrained openai (336x336 vs 224x224)


@dataclass
class CLIPScoreResult:
    """Per-attribute scoring output with debug info (3-grade ordinal scoring)."""
    score: float          # weighted grade position: p1*0 + p2*0.5 + p3*1.0
    p_grade1: float       # probability of worst grade
    p_grade2: float       # probability of middle grade
    p_grade3: float       # probability of best grade
    sim_grade1: float     # cosine similarity to grade1 embedding
    sim_grade2: float     # cosine similarity to grade2 embedding
    sim_grade3: float     # cosine similarity to grade3 embedding
    confidence: float     # p_grade3 - p_grade1 (positive = leans good)
    p_good: float         # alias for p_grade3 (backward compat)
    p_bad: float          # alias for p_grade1 (backward compat)
    sim_good: float       # alias for sim_grade3 (backward compat)
    sim_bad: float        # alias for sim_grade1 (backward compat)


class CLIPSkinScorer:
    """
    Loads OpenCLIP, pre-computes prompt embeddings, scores images against
    each attribute's good/bad embedding pair.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: Optional[str] = None,
    ):
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model + preprocessing transform
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # CLIP's learned temperature (logit_scale). Using model.logit_scale
        # gives proper softmax temperatures that match CLIP's training.
        self.logit_scale = float(self.model.logit_scale.exp().detach().cpu().item())

        # Pre-compute text embeddings for all prompts (one-time cost)
        self._prompt_embeddings = self._encode_all_prompts()
        self._attribute_embeddings = self._build_attribute_embeddings()

    @torch.inference_mode()
    def _encode_all_prompts(self) -> torch.Tensor:
        """Encode every prompt across all attributes. Returns (N, D) tensor."""
        prompts = get_all_prompts()
        tokens = self.tokenizer(prompts).to(self.device)
        embeddings = self.model.encode_text(tokens)
        # L2 normalize so cosine = dot product
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def _build_attribute_embeddings(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        For each attribute, average the 3 grade prompt embeddings.
        Returns {attribute: (grade1_emb, grade2_emb, grade3_emb)} each (D,) L2-normalized.
        """
        indices = get_prompt_indices()
        out = {}
        for attr in ATTRIBUTE_ORDER:
            embs = []
            for grade in ("grade1", "grade2", "grade3"):
                s = indices[attr][grade]
                emb = self._prompt_embeddings[s].mean(dim=0)
                emb = emb / emb.norm()
                embs.append(emb)
            out[attr] = tuple(embs)
        return out

    @torch.inference_mode()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to a normalized embedding (D,)."""
        # preprocess returns (C, H, W); add batch dim
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0]  # back to (D,)

    def score_image(self, pil_image: Image.Image) -> Dict[str, CLIPScoreResult]:
        """Score a PIL image across all attributes using 3-grade ordinal scoring."""
        image_emb = self.encode_image(pil_image)

        results = {}
        for attr in ATTRIBUTE_ORDER:
            g1_emb, g2_emb, g3_emb = self._attribute_embeddings[attr]
            sim1 = float((image_emb @ g1_emb).item())
            sim2 = float((image_emb @ g2_emb).item())
            sim3 = float((image_emb @ g3_emb).item())

            # Numerically stable 3-class softmax with CLIP's logit_scale
            l1, l2, l3 = sim1 * self.logit_scale, sim2 * self.logit_scale, sim3 * self.logit_scale
            max_l = max(l1, l2, l3)
            e1, e2, e3 = np.exp(l1 - max_l), np.exp(l2 - max_l), np.exp(l3 - max_l)
            denom = e1 + e2 + e3
            p1, p2, p3 = float(e1 / denom), float(e2 / denom), float(e3 / denom)

            # Weighted position: grade1=1.0, grade2=0.5, grade3=0.0  (high = worse skin)
            score = p1 * 1.0 + p2 * 0.5

            results[attr] = CLIPScoreResult(
                score=score,
                p_grade1=p1, p_grade2=p2, p_grade3=p3,
                sim_grade1=sim1, sim_grade2=sim2, sim_grade3=sim3,
                confidence=p3 - p1,
                p_good=p3, p_bad=p1,
                sim_good=sim3, sim_bad=sim1,
            )

        return results
