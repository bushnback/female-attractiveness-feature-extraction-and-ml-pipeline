"""
CLIP-based neoteny / cuteness attribute scorer.

Image is encoded once; all 5 attributes are scored via matrix ops against
pre-computed prompt embeddings. Score convention: 0.0 = least neotenous
(mature-looking), 1.0 = most neotenous (cute/childlike).
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import open_clip

from neoteny_prompts import GRADE_PROMPTS, ATTRIBUTE_ORDER, get_all_prompts, get_prompt_indices


DEFAULT_MODEL      = "ViT-L-14"
DEFAULT_PRETRAINED = "laion2b_s32b_b82k"


class CLIPNeotenySscorer:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer   = open_clip.get_tokenizer(model_name)
        self.logit_scale = float(self.model.logit_scale.exp().detach().cpu().item())

        self._prompt_embeddings    = self._encode_all_prompts()
        self._attribute_embeddings = self._build_attribute_embeddings()

    @torch.inference_mode()
    def _encode_all_prompts(self) -> torch.Tensor:
        prompts = get_all_prompts()
        tokens  = self.tokenizer(prompts).to(self.device)
        embs    = self.model.encode_text(tokens)
        return embs / embs.norm(dim=-1, keepdim=True)

    def _build_attribute_embeddings(self) -> Dict[str, Tuple[torch.Tensor, ...]]:
        indices = get_prompt_indices()
        out = {}
        for attr in ATTRIBUTE_ORDER:
            embs = []
            for grade in ("grade1", "grade2", "grade3"):
                s   = indices[attr][grade]
                emb = self._prompt_embeddings[s].mean(dim=0)
                emb = emb / emb.norm()
                embs.append(emb)
            out[attr] = tuple(embs)
        return out

    @torch.inference_mode()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        tensor    = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0]

    def score_image(self, pil_image: Image.Image) -> Dict[str, float]:
        """Return {attribute: score} for all attributes. Score in [0, 1], higher = more neotenous."""
        image_emb = self.encode_image(pil_image)
        results   = {}
        for attr in ATTRIBUTE_ORDER:
            g1, g2, g3 = self._attribute_embeddings[attr]
            s1 = float((image_emb @ g1).item())
            s2 = float((image_emb @ g2).item())
            s3 = float((image_emb @ g3).item())

            l1, l2, l3 = s1 * self.logit_scale, s2 * self.logit_scale, s3 * self.logit_scale
            max_l = max(l1, l2, l3)
            e1, e2, e3 = np.exp(l1 - max_l), np.exp(l2 - max_l), np.exp(l3 - max_l)
            denom       = e1 + e2 + e3
            p1, p2, p3  = e1 / denom, e2 / denom, e3 / denom

            # grade3 = most neotenous = 1.0
            results[attr] = float(p2 * 0.5 + p3 * 1.0)
        return results
