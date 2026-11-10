"""
Prompt definitions for CLIP-based neoteny / cuteness scoring.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSTRUCT: NEOTENY (babyfacedness / cuteness)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Neoteny refers to the retention of juvenile facial proportions in adult faces:
large eyes relative to face size, small nose, round face, prominent forehead,
and short lower face. These cues are associated with "cuteness" rather than
"beauty" — a deliberate distinction in this scorer.

In CLIP's training data "cute" and "beautiful" occupy meaningfully different
embedding regions. "Beautiful" clusters with mature, symmetrical, feminine
faces; "cute" clusters with childlike, youthful, doll-like proportions. Using
"cute" framing here intentionally separates this construct from the femininity
scorer in folder 27, so the two scores provide complementary rather than
redundant information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3-grade ordinal scoring per attribute: grade1 (least neotenous / most mature)
/ grade2 (average) / grade3 (most neotenous / most childlike). 3 paraphrases
per grade — text embeddings are averaged within each grade before scoring.

  Score = p_grade1 * 0.0 + p_grade2 * 0.5 + p_grade3 * 1.0

Score convention: 0.0 = least neotenous (mature-looking), 1.0 = most
neotenous (cute/childlike). Higher = more neotenous.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WITHIN-FEMALE ANCHORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The low anchor (grade1) describes a woman who looks mature and striking rather
than cute — NOT an elderly or unattractive person. This keeps all three anchors
within the range of adult women in the dataset and prevents the ceiling effect
seen in folder 27 V1 (where an out-of-range low anchor caused every subject to
score near 1.0).

The contrast pair is: cute / youthful / childlike  vs  striking / mature / elegant.
Both ends are positive framings — the scorer measures a style dimension, not
an attractiveness dimension.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ATTRIBUTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  eye_size           — small/narrow eyes vs large/wide/doll-like eyes
  nose_size          — prominent nose vs small button nose
  face_roundness     — angular/elongated face vs round/circular face
  lower_face_length  — long lower face vs short lower face (neotenous)
  overall_cuteness   — holistic cute/youthful impression vs mature/striking
"""

from typing import Dict, List


GRADE_PROMPTS: Dict[str, Dict[str, List[str]]] = {

    # ── eye_size ──────────────────────────────────────────────────────────────
    # Grade3 now describes the perceptual quality of wide-open, bright, innocent
    # eyes rather than demanding geometrically enormous eyes — so naturally
    # youthful faces with moderately sized but open eyes can still score high.
    "eye_size": {
        "grade1": [
            "a woman with narrow hooded eyes — small relative to her face with heavy lids that reduce their apparent size",
            "a woman with small deep-set eyes that have a defined lid fold and appear narrow and less open than most women",
            "a woman with close-set narrow eyes that are clearly smaller than average for a woman's face",
        ],
        "grade2": [
            "a woman with eyes of typical size — average and unremarkable among women",
            "a woman with moderately sized eyes that are neither large nor small for a woman",
            "a woman with eyes that are average in size and prominence for a female face",
        ],
        "grade3": [
            "a woman with wide-open bright eyes that give her face a youthful innocent and wide-eyed look",
            "a woman with large open eyes that look fresh and childlike — giving her a wide-eyed innocent expression",
            "a woman with notably open expressive eyes that make her look youthful, wide-eyed, and innocent",
        ],
    },

    # ── nose_size ─────────────────────────────────────────────────────────────
    # Small button nose is the core neoteny signal. Grade1 names the specific
    # anatomy of a prominent adult nose rather than using vague size language.
    "nose_size": {
        "grade1": [
            "a woman with a strong prominent nose — a defined bridge, noticeable width, and clear presence on her face",
            "a woman with a large nose that has a long defined bridge and wide base, clearly standing out on her face",
            "a woman with a broad nose with a pronounced bridge that draws the eye and dominates the centre of her face",
        ],
        "grade2": [
            "a woman with a nose of average size and prominence for a woman",
            "a woman with a nose that is neither small nor large — typical for a female face",
            "a woman with an ordinary nose that does not draw attention for its size",
        ],
        "grade3": [
            "a woman with a very small button nose — compact and barely projecting, giving her face a childlike quality",
            "a woman with a small upturned nose that is barely noticeable — soft and small like a child's nose",
            "a woman with a tiny delicate nose that is small even for a woman, making her face look young and babyfaced",
        ],
    },

    # ── face_roundness ────────────────────────────────────────────────────────
    # Round face is a baby-face cue. Grade1 names specific geometry: clearly
    # longer than wide, defined jawline, prominent chin — not just "angular."
    "face_roundness": {
        "grade1": [
            "a woman with a clearly oval or rectangular face — noticeably longer than it is wide, with a defined jawline",
            "a woman with a long narrow face that is distinctly taller than wide, with a prominent chin and angular jaw",
            "a woman with an elongated face shape where the forehead, cheeks, and jaw form a clearly vertical oval",
        ],
        "grade2": [
            "a woman with a face shape that is averagely round — neither clearly oval nor clearly circular",
            "a woman with a moderate face shape that is typical and unremarkable among women",
            "a woman with an average face shape that is neither notably round nor notably angular",
        ],
        "grade3": [
            "a woman with a round soft face — wide and full with gentle rounded contours that give her a youthful look",
            "a woman with a round face shape — soft full cheeks and a rounded chin that make her look young and babyfaced",
            "a woman with a noticeably round face — soft and full, giving her an approachable youthful babyface quality",
        ],
    },

    # ── lower_face_length ─────────────────────────────────────────────────────
    # Short lower face (nose-to-chin) is a classic neoteny cue. Grade1 names
    # specific geometry: long jaw, prominent chin, large philtrum-to-chin span.
    "lower_face_length": {
        "grade1": [
            "a woman with a noticeably long lower face — a large gap between her nose and chin with a prominent elongated jaw",
            "a woman whose chin sits far below her nose, giving her a long adult lower face with a strong jawline",
            "a woman with a long lower third to her face — her nose-to-chin distance is clearly longer than average",
        ],
        "grade2": [
            "a woman with a lower face of typical length — average nose-to-chin proportion",
            "a woman with lower face proportions that are unremarkable and average for a woman",
            "a woman with a moderate lower face length — neither short nor long for a woman",
        ],
        "grade3": [
            "a woman with a short compact lower face — her nose and chin are close together, giving her face a rounded youthful look",
            "a woman with a notably short lower face — the space from her nose to her chin is small, making her face look young and childlike",
            "a woman with a compressed lower face where the chin sits close to the nose — a classic babyface proportion",
        ],
    },

    # ── overall_babyface ──────────────────────────────────────────────────────
    # Renamed from overall_cuteness. Grade3 now describes someone who holistically
    # looks young and babyfaced — not requiring all extreme doll features at once.
    # A naturally youthful face like e0202 should land in grade3 even without
    # cartoonishly large eyes or exaggerated proportions.
    "overall_babyface": {
        "grade1": [
            "a woman with strong defined adult features — high cheekbones, a prominent nose, a defined jaw, and a long lower face. She looks like a grown adult, not young or babyfaced",
            "a woman with sharp adult facial geometry: a long face, a strong jaw, a noticeable nose, and small eyes. She looks mature and adult-featured",
            "a woman with mature prominent features — a long lower face, a wide-bridged nose, narrow eyes, and a defined angular jawline. She looks like a fully developed adult face",
        ],
        "grade2": [
            "a woman with an average face — she looks like a typical adult woman, neither particularly young nor mature looking",
            "a woman with a face that is neither notably young and babyfaced nor notably mature and adult — just typical",
            "a woman with a neutral appearance — a standard adult female face with no strong youthful or mature quality",
        ],
        "grade3": [
            "a woman who looks young and babyfaced — her soft features and youthful proportions make her look younger than she probably is",
            "a woman with a genuine babyface — soft features, a small nose, and a youthful freshness that makes her look childlike even as an adult",
            "a woman who has a naturally youthful babyfaced appearance — soft rounded features and fresh young-looking skin that give her a childlike quality",
        ],
    },
}


ATTRIBUTE_ORDER = [
    "eye_size",
    "nose_size",
    "face_roundness",
    "lower_face_length",
    "overall_babyface",
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
