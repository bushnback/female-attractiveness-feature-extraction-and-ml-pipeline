# Female Facial Attractiveness Feature Extraction Pipeline

Extracts ~150+ interpretable 2D/3D geometric and vision-model features
from a folder of face photos, producing a single `ml_features.csv` ready
for modelling — plus an optional Lasso/ElasticNet/XGBoost+SHAP/Cohen's d
modeling stage if you supply attractiveness labels.

**Author:** Andrew Hale
**License:** [MIT](LICENSE) — free to use, modify, and distribute.

This is a from-scratch reusable version of the pipeline built for the
accompanying thesis ("Explainable Prediction of Female Facial
Attractiveness using 2D/3D Geometric and Vision-Derived Features"). No
datasets, outputs, or proprietary images are included here — only the
feature-extraction code, so you can run it on your own dataset.

## Architecture

Three parallel feature streams feed one aggregation step, with an
optional modeling stage after:

```
Photos ──┬──→ 2D landmarks ──→ 2D geometric features ────┐
         │                     + GPA averageness          │
         │                                                 │
         ├──→ DECA 3D mesh ──→ DAD-3DHeads 445 lm ──┬──→ face shape     │
         │                                          ├──→ 3D depth      ├─→ Feature
         │                                          └──→ 3D surface    │  Aggregation
         │                                                             │  (ml_features.csv)
         └──→ derived (vision-model) features ─────────────────────────┘
              age, skin quality, femininity, neoteny, adiposity,
              hair colour/length, glasses, tattoo/piercing,
              composite femininity/neoteny (2D+3D combined)
                                   │
                                   ▼
                 Modeling & Explainability
                 Lasso / ElasticNet / XGBoost+SHAP / Cohen's d
```

### `pipeline/geometry_2d/`
- `landmarks_68.py` / `landmarks_106.py` — 2D landmark detection (FAN-68
  with BiSeNet-refined jawline, or InsightFace-106)
- `features_2d_68.py` / `features_2d_106.py` — ~50 geometric features
  per landmark scheme (proportions, eyes, symmetry, jaw/face shape,
  spatial relationships), all IOD-normalised
- `averageness_gpa.py` — Generalised Procrustes Analysis distinctiveness
  score, with solvePnP head-pose exclusion

**106pt landmarks are the recommended default** — they measure the same
features more precisely and meaningfully improved every result tested
(see the thesis's Future Work section and Table 17/22). 68pt is kept for
comparability with studies using the classic iBUG convention; pick either
with `--landmarks 68` / `--landmarks 106`.

### `pipeline/geometry_3d/`
- `deca_mesh.py` — DECA FLAME mesh reconstruction (5023 vertices)
- `dad_landmarks_445.py` — 445 semantic landmarks indexed from the mesh
  (region definitions bundled in `dad_3dheads_keypoints/`)
- `face_shape.py` — face shape classification (oval/round/square/heart/
  diamond/oblong/triangle) from percentile-based 3D measurements
- `depth_curvature.py` — nose protrusion, orbital depth, cheekbone
  prominence, chin projection, forehead slope, facial convexity, etc.
- `surface_geodesic.py` — surface area ratios, geodesic/Euclidean path
  ratios, regional angularity, surface normal variance

MICA mesh support was dropped from this public version — in testing it
had no working inference path (pre-computed meshes only), so DECA is the
only supported 3D mesh source here.

### `pipeline/derived/`
Vision-model features, all CLIP/SigLIP2-based except age:
`age_mivolo.py`, `skin_quality_clip.py` (+ MediaPipe quality flags),
`femininity_clip.py`, `neoteny_clip.py`, `adiposity_clip.py`,
`hair_color.py` / `hair_length.py`, `glasses_clip.py`,
`tattoo_piercing_clip.py`, and `composite_femininity.py` (combines 2D +
3D geometric features into relative femininity/neoteny scores, separate
from the CLIP-based `femininity_clip.py`/`neoteny_clip.py`).

### `pipeline/aggregate/` and `pipeline/modeling/`
`aggregate.py` merges every stage's per-subject CSV into one
`ml_features.csv`, prefixed by source (e.g. `lm2d__face_width_jaw`,
`skin__overall_skin_quality`). `modeling/` runs Lasso, ElasticNet,
XGBoost+SHAP, and a Cohen's d bottom-vs-top-20% analysis against that CSV
if you have attractiveness labels.

## Setup

See `setup/README.md` — three conda environments, plus which model
weights download automatically vs. need a one-time manual step (DECA's
FLAME model and MiVOLO's checkpoints are both gated behind external
registration/release pages, so they can't be auto-downloaded by a
script).

## Usage

Drop your photos into `input/` (each filename, minus extension, becomes
that subject's ID throughout the pipeline), then:

```bash
conda activate attractiveness-pipeline
python run_pipeline.py \
    --landmarks 106 \
    --deca-repo /path/to/DECA \
    --mivolo-repo /path/to/MiVOLO \
    --mivolo-detector-weights /path/to/yolov8x_person_face.pt \
    --mivolo-checkpoint /path/to/model_imdb_cross_person_4.24_99.46.pth.tar
```

This reads from `input/` and writes everything — every stage's
intermediate CSVs/JSONs in their own subfolder, plus the final
`ml_features.csv` — into `output/` by default. Pass `--input` / `--work-dir`
to use different folders instead.

Quick smoke test with no external repos needed (2D-only):
```bash
python run_pipeline.py --landmarks 106 --skip-3d --skip-derived
```

Add `--labels labels.txt` (whitespace-separated `filename score` per
line, or a 2-column CSV) to also run the modeling stage.

Each stage module is also runnable standalone via its own CLI (`python
pipeline/geometry_2d/landmarks_106.py --input ... --output ...`) if you
only need one feature group.

## What's NOT included

- Any images, datasets, or `ml_features.csv` output — bring your own photos
- Model weights — downloaded automatically (CLIP/SigLIP2/MediaPipe) or via
  a documented one-time manual step (DECA/FLAME, MiVOLO)
- The Lasso/ElasticNet/XGBoost model-selection narrative and dataset-specific
  analysis from the thesis itself — `modeling/` runs the same algorithms but
  interpreting results is up to you and your own labels
