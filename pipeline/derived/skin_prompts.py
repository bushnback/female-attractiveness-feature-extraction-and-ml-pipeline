"""
Prompt definitions for CLIP-based skin attribute scoring.

3-grade ordinal scoring: grade1 (worst) / grade2 (average) / grade3 (best).
Each grade has 3 paraphrases for prompt ensembling — embeddings are averaged
within each grade before scoring.

Score = p_grade1 * 1.0 + p_grade2 * 0.5 + p_grade3 * 0.0
where p_gradeN = softmax probability over the 3 competing grade embeddings.

The middle anchor (grade2) is the key addition over binary scoring — it
pulls probability mass away from the extremes for average-quality faces,
spreading the score distribution across [0, 1] rather than compressing
everything toward 1.0.

Score convention: 1.0 = worst skin, 0.0 = best skin. Higher = more problematic.
Consistent with adiposity and tattoo pipelines where high score = more of the
measured attribute (wrinkles, blemishes, redness, etc.).
"""

from typing import Dict, List


GRADE_PROMPTS: Dict[str, Dict[str, List[str]]] = {
    "wrinkle_score": {
        "grade1": [
            "a photo of wrinkled aged skin with deep lines and creases",
            "a face covered in pronounced wrinkles and heavy age lines",
            "skin showing severe wrinkling and deep creases from aging",
        ],
        "grade2": [
            "a face with some fine lines and early signs of aging",
            "skin showing a few light wrinkles and developing fine lines",
            "a face with mild wrinkles and moderate skin aging",
        ],
        "grade3": [
            "a photo of smooth youthful skin without wrinkles",
            "a face with no wrinkles or fine lines",
            "skin that appears completely smooth and unwrinkled",
        ],
    },
    "texture_smoothness": {
        # Key fix: use specific pore-size and surface-grain language CLIP can
        # visually distinguish; previous prompts were too similar across grades.
        "grade1": [
            "a close-up photo of rough bumpy skin with very large visible pores and uneven surface",
            "skin with an orange-peel texture, enlarged pores, and a coarse uneven surface",
            "a face with extremely rough skin where pores are clearly visible from a normal viewing distance",
        ],
        "grade2": [
            "a face with average skin texture where pores are visible but not excessively large",
            "skin with moderate texture and pores of normal size visible in good lighting",
            "a face with typical skin texture showing some pore visibility across the cheeks",
        ],
        "grade3": [
            "a close-up photo of extremely smooth skin with very small nearly invisible pores",
            "skin with a fine silky surface texture where individual pores are barely perceptible",
            "a face with porcelain-smooth skin and a uniformly fine-grained even surface",
        ],
    },
    "tone_evenness": {
        "grade1": [
            "a photo of blotchy uneven skin tone with strong discoloration",
            "skin with obvious patches of very different colors",
            "heavily uneven skin with prominent discoloration patches",
        ],
        "grade2": [
            "skin with some minor tone variation and slight unevenness",
            "a face with mild uneven tone in a few areas",
            "skin with occasional slight discoloration but mostly even",
        ],
        "grade3": [
            "a photo of even-toned skin with perfectly uniform color",
            "skin with consistent uniform tone throughout",
            "evenly colored skin with no patches or discoloration",
        ],
    },
    "redness_score": {
        "grade1": [
            "a photo of red flushed irritated skin",
            "skin showing strong visible redness and heavy flushing",
            "skin with red blotchy inflamed appearance",
        ],
        "grade2": [
            "skin with mild redness and slight pinkness in some areas",
            "a face with some light redness in certain areas",
            "skin showing minor flush or light redness",
        ],
        "grade3": [
            "a photo of skin with calm even neutral color",
            "skin completely without redness or irritation",
            "skin with perfectly neutral non-red tone",
        ],
    },
    "pigmentation_score": {
        "grade1": [
            "a photo of skin heavily covered in dark spots, freckles, and sun damage",
            "skin showing significant hyperpigmentation with many age spots and dark marks across the face",
            "skin densely covered in freckles and dark patches across the nose and cheeks",
        ],
        "grade2": [
            "skin with a handful of clearly visible dark spots or pigmentation marks",
            "a face with several small but noticeable pigmentation marks or mild discoloration",
            "skin showing a few scattered sun spots or small areas of darker pigmentation",
        ],
        "grade3": [
            "a photo of skin with perfectly even pigmentation and no visible dark spots",
            "skin with completely uniform color and no dark spots or pigmentation marks",
            "skin with no visible hyperpigmentation or discoloration of any kind",
        ],
    },
    "blemish_score": {
        "grade1": [
            "a photo of skin with active acne, many pimples, and multiple blemishes",
            "skin showing numerous visible pimples, pustules, and inflamed spots",
            "heavily blemished skin with widespread active breakouts",
        ],
        "grade2": [
            "skin with one or two noticeable pimples or small active blemishes",
            "a face with a few clearly visible spots or small blemishes",
            "skin showing a small number of individual blemishes or minor spots",
        ],
        "grade3": [
            "a photo of completely clear skin without any pimples or blemishes",
            "skin with no visible spots, pimples, or blemishes whatsoever",
            "perfectly clear unblemished skin with no active breakouts",
        ],
    },
    "under_eye_score": {
        # Key fix: grade2 needed more specific visual language (colour/shadow)
        # so CLIP can distinguish it from grade3; previous grade2 was too vague.
        "grade1": [
            "a face with deep dark purple or brown circles clearly visible under both eyes",
            "a person with prominent dark hollow bags under the eyes and heavy under-eye discoloration",
            "under-eye skin that is noticeably and significantly darker than the surrounding face",
        ],
        "grade2": [
            "a face with mild bluish or brownish shadows visible under the eyes",
            "a person with faint but noticeable under-eye circles showing light discoloration",
            "a face with light under-eye shadows that are visible in normal daylight",
        ],
        "grade3": [
            "a face with bright even-toned under-eye skin that matches the surrounding skin color",
            "a person with no discoloration or shadows visible in the under-eye area",
            "a well-rested face with completely uniform under-eye skin tone and no darkness",
        ],
    },
    "shine_score": {
        "grade1": [
            "a photo of very oily shiny skin with strong visible sheen",
            "skin with heavy excess oil and greasy shine",
            "extremely shiny oily skin with strong reflection",
        ],
        "grade2": [
            "skin with some shine and slight oiliness",
            "a face with mild shine especially in the T-zone area",
            "skin with moderate oiliness and some gloss",
        ],
        "grade3": [
            "a photo of skin with matte balanced finish",
            "skin without any excess shine or oil",
            "skin with healthy non-shiny matte appearance",
        ],
    },
    "overall_skin_quality": {
        # Run3 lesson: tone-neutral grade3 prompts collapsed std from 0.231→0.090
        # because grades 2/3 became visually indistinct. Fix: sharpen the contrast
        # between grades with more specific visual anchors while keeping tone-neutrality.
        "grade1": [
            "a face with clearly visible acne, blemishes, redness, and rough uneven skin texture",
            "skin showing multiple obvious problems: spots, discoloration, enlarged pores, and uneven tone",
            "a face with noticeably problematic skin showing several visible issues such as pimples or dark spots",
        ],
        "grade2": [
            "a face with a couple of small blemishes or minor skin imperfections visible on otherwise average skin",
            "skin with one or two visible spots and slight unevenness but no severe or widespread problems",
            "a face with ordinary skin quality — a few minor marks visible but nothing dramatically wrong",
        ],
        "grade3": [
            "a face with completely clear skin — no pimples, no spots, no redness, smooth and even-toned",
            "skin with zero visible blemishes, an even tone, and fine smooth texture across the whole face",
            "a face showing flawless clear skin with no marks, spots, or texture problems of any kind",
        ],
    },
}


# Order matters for output structure consistency
ATTRIBUTE_ORDER = [
    "texture_smoothness",
    "wrinkle_score",
    "tone_evenness",
    "redness_score",
    "pigmentation_score",
    "blemish_score",
    "under_eye_score",
    "shine_score",
    "overall_skin_quality",
]


def get_all_prompts() -> List[str]:
    """Flat list of every prompt across all attributes — useful for batch encoding."""
    out = []
    for attr in ATTRIBUTE_ORDER:
        out.extend(GRADE_PROMPTS[attr]["grade1"])
        out.extend(GRADE_PROMPTS[attr]["grade2"])
        out.extend(GRADE_PROMPTS[attr]["grade3"])
    return out


def get_prompt_indices() -> Dict[str, Dict[str, slice]]:
    """
    Returns a mapping from attribute -> {"grade1": slice, "grade2": slice, "grade3": slice}
    indicating positions in the flat prompt list for each grade.
    """
    indices = {}
    cursor = 0
    for attr in ATTRIBUTE_ORDER:
        n1 = len(GRADE_PROMPTS[attr]["grade1"])
        n2 = len(GRADE_PROMPTS[attr]["grade2"])
        n3 = len(GRADE_PROMPTS[attr]["grade3"])
        indices[attr] = {
            "grade1": slice(cursor, cursor + n1),
            "grade2": slice(cursor + n1, cursor + n1 + n2),
            "grade3": slice(cursor + n1 + n2, cursor + n1 + n2 + n3),
        }
        cursor += n1 + n2 + n3
    return indices
