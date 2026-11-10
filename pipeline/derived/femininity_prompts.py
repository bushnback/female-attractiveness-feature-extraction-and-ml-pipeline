"""
Prompt definitions for CLIP-based femininity scoring.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3-grade ordinal scoring per attribute: grade1 (least feminine) / grade2 (average)
/ grade3 (most feminine). 3 paraphrases per grade for prompt ensembling — the
CLIP text embeddings are averaged within each grade before scoring.

  Score = p_grade1 * 0.0 + p_grade2 * 0.5 + p_grade3 * 1.0

Score convention: 0.0 = least feminine, 1.0 = most feminine. Higher = more feminine.

Attributes scored:
  jaw_softness       — square/wide jaw vs soft/tapered jaw
  eye_prominence     — small/narrow eyes vs large/expressive eyes
  lip_fullness       — thin lips vs full/plump lips
  nose_delicacy      — large/prominent nose vs small/delicate nose
  face_softness      — angular/sharp features vs soft/rounded features
  overall_femininity — holistic femininity impression

Note on correlation: these features co-occur in CLIP's training data (feminine
faces tend to have all features at once), so sub-scores will not be fully
independent of each other. overall_femininity is the most reliable single signal;
sub-scores add interpretability at the cost of partial redundancy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT DESIGN — V1 (ORIGINAL) AND THE CLUSTERING PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V1 prompts described grade1 using masculine male faces as the low anchor
(e.g. "a face with a strong square masculine jaw structure"). When applied to
an all-female dataset, this caused severe score clustering toward 1.0 — every
woman looked highly feminine relative to a male reference point, producing an
uninformative score distribution (most scores 0.75–0.98).

This is the same ceiling effect observed in folder 24 (skin quality) when a
binary good/bad prompt was used: if the bad anchor is too extreme relative to
the actual population, CLIP assigns high probability to the good anchor for
virtually every image.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT DESIGN — V2 (CURRENT) — WITHIN-FEMALE ANCHORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The core fix: all three grade anchors are explicitly framed as *women*, with
grade1 describing the less-feminine end of the female spectrum rather than a
male face. Key phrases used throughout:

  grade1: "a woman with relatively [less feminine feature] for a woman"
            / "compared to other women" / "on the less feminine end among women"
  grade2: "a woman with typical / average [feature] for a woman"
            / "neither notably feminine nor notably androgynous"
  grade3: "a woman with notably [feminine feature] even among women"
            / "more [feminine trait] than typical for a woman"

This anchors CLIP's softmax competition within the female population rather
than between sexes, distributing scores across the full [0, 1] range even in
an all-female dataset. The approach is analogous to the 3-grade middle-anchor
technique in folder 24 that resolved the binary clustering problem for skin
quality scoring.
"""

from typing import Dict, List


GRADE_PROMPTS: Dict[str, Dict[str, List[str]]] = {

    # ── jaw_softness ──────────────────────────────────────────────────────────
    # V1 grade1 described male jaws. V2 describes the less-feminine end of the
    # female jaw spectrum — wider or squarer than typical for a woman, but still
    # framed as a woman. This keeps all three anchors within the female range.
    "jaw_softness": {
        "grade1": [
            "a woman with a relatively wide square jaw compared to other women",
            "a woman whose jaw is broader and more angular than typical for a woman",
            "a woman with a noticeably square jawline that is wide for a female face",
        ],
        "grade2": [
            "a woman with an average jaw width that is typical for a woman",
            "a woman with a jaw of moderate definition — neither very square nor very tapered",
            "a woman with a jaw shape that is unremarkable and typical among women",
        ],
        "grade3": [
            "a woman with a notably soft tapered jawline — more delicate than typical for a woman",
            "a woman with an unusually narrow V-shaped chin and very soft jawline even among women",
            "a woman with an exceptionally delicate refined jaw that tapers to a pointed chin",
        ],
    },

    # ── eye_prominence ────────────────────────────────────────────────────────
    # V1 grade1 described male-typical small eyes. V2 describes women with
    # relatively smaller eyes within the female range — still clearly a woman,
    # just with less prominent eyes than average for women.
    "eye_prominence": {
        "grade1": [
            "a woman with relatively small narrow eyes compared to other women",
            "a woman whose eyes are less prominent than typical for a woman",
            "a woman with small eyes that are smaller relative to her face than is typical for women",
        ],
        "grade2": [
            "a woman with eyes of typical size and prominence for a woman",
            "a woman with moderately sized eyes that are average among women",
            "a woman with eyes that are neither notably large nor notably small for a woman",
        ],
        "grade3": [
            "a woman with notably large wide eyes — larger and more open than typical for a woman",
            "a woman with unusually prominent expressive eyes that stand out even among women",
            "a woman with strikingly large eyes that are more prominent than typical for women",
        ],
    },

    # ── lip_fullness ──────────────────────────────────────────────────────────
    # V1 grade1 described generically thin lips. V2 explicitly frames thin lips
    # as thin-for-a-woman, anchoring within the female distribution.
    "lip_fullness": {
        "grade1": [
            "a woman with relatively thin lips compared to other women",
            "a woman whose lips are less full than typical for a woman",
            "a woman with lips that are noticeably thin for a female face",
        ],
        "grade2": [
            "a woman with lips of typical fullness for a woman",
            "a woman with moderately full lips that are average among women",
            "a woman with lip volume that is neither thin nor noticeably full for a woman",
        ],
        "grade3": [
            "a woman with notably full plump lips — fuller than typical for a woman",
            "a woman with unusually voluminous lips that are more prominent than typical among women",
            "a woman with exceptionally full lips that are lush even by female standards",
        ],
    },

    # ── nose_delicacy ─────────────────────────────────────────────────────────
    # V1 grade1 described large noses without female framing. V2 explicitly
    # frames larger noses as relatively larger within the female population.
    "nose_delicacy": {
        "grade1": [
            "a woman with a relatively large or wide nose compared to other women",
            "a woman whose nose is more prominent than typical for a woman",
            "a woman with a nose that is broader or larger than the average for women",
        ],
        "grade2": [
            "a woman with a nose of typical size and proportions for a woman",
            "a woman with an average nose that is unremarkable among women",
            "a woman with a nose that is neither large nor small relative to typical women",
        ],
        "grade3": [
            "a woman with a notably small delicate nose — more refined than typical for a woman",
            "a woman with an unusually petite narrow nose that is more delicate than most women",
            "a woman with an exceptionally small fine-featured nose that is dainty even among women",
        ],
    },

    # ── face_softness ─────────────────────────────────────────────────────────
    # V1 grade1 described male-typical angular bone structure. V2 describes
    # women with relatively more defined/angular features within the female range.
    "face_softness": {
        "grade1": [
            "a woman with relatively angular defined facial features compared to other women",
            "a woman whose bone structure is more prominent and angular than typical for a woman",
            "a woman with sharper more defined facial contours than is typical among women",
        ],
        "grade2": [
            "a woman with facial feature definition that is average and typical for a woman",
            "a woman with moderate bone definition — neither very angular nor very soft for a woman",
            "a woman with facial contours that are unremarkable and typical among women",
        ],
        "grade3": [
            "a woman with notably soft rounded facial features — softer than typical for a woman",
            "a woman with unusually gentle smooth facial contours and soft cheeks even among women",
            "a woman with exceptionally soft delicate facial features that are very feminine even for a woman",
        ],
    },

    # ── overall_femininity ────────────────────────────────────────────────────
    # V1 grade1 used a male face as the low anchor — the most severe case of
    # the cross-sex anchoring problem. V2 grades describe the range of femininity
    # WITHIN women: from androgynous-for-a-woman through to very feminine-for-a-woman.
    "overall_femininity": {
        "grade1": [
            "a woman with relatively androgynous features — less feminine than most women",
            "a woman on the less feminine end of the female spectrum with a broader jaw and less defined features",
            "a woman whose overall facial features are less stereotypically feminine compared to other women",
        ],
        "grade2": [
            "a woman with typical feminine features that are average and unremarkable among women",
            "a woman with a moderate degree of femininity — neither strongly feminine nor androgynous for a woman",
            "a woman with facial femininity that is typical and average for a female face",
        ],
        "grade3": [
            "a woman with strongly pronounced feminine features — more feminine than most women",
            "a woman with highly feminine facial proportions that stand out as very feminine even among women",
            "a woman with exceptionally feminine features: very soft jaw, large eyes, full lips, and a delicate nose",
        ],
    },
}


ATTRIBUTE_ORDER = [
    "jaw_softness",
    "eye_prominence",
    "lip_fullness",
    "nose_delicacy",
    "face_softness",
    "overall_femininity",
]


def get_all_prompts() -> List[str]:
    out = []
    for attr in ATTRIBUTE_ORDER:
        out.extend(GRADE_PROMPTS[attr]["grade1"])
        out.extend(GRADE_PROMPTS[attr]["grade2"])
        out.extend(GRADE_PROMPTS[attr]["grade3"])
    return out


def get_prompt_indices() -> Dict[str, Dict[str, slice]]:
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
