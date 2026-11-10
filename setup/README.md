# Setup

This pipeline ships as code only — no datasets, no model weights. Every
stage either downloads its model automatically on first use, or needs one
manual step below (because the model is gated behind free registration,
or isn't hosted somewhere a script can reliably fetch it from).

## 1. Conda environments

Three separate environments, because each has a different (and sometimes
conflicting) set of pinned dependencies:

| Env | Used for | Install |
|---|---|---|
| `attractiveness-pipeline` | 2D landmarks, DAD-3DHeads, depth/surface, GPA averageness, modeling | `pip install -r requirements-pipeline.txt` |
| `attractiveness-clip` | All CLIP/SigLIP2 vision-feature stages | `pip install -r requirements-clip-stages.txt` |
| `deca-env` | DECA 3D mesh reconstruction | see `requirements-deca-env.txt` — install torch first, then pytorch3d, then the rest |

Run each stage from the appropriate environment, e.g.:
```bash
conda activate attractiveness-pipeline
python run_pipeline.py --input photos/ --work-dir work/ ...
```

## 2. Automatic downloads (no action needed)

- **CLIP / SigLIP2 models** (skin, femininity, neoteny, adiposity, glasses,
  tattoo/piercing, hair) — downloaded by `open_clip` into `~/.cache` the
  first time each stage runs.
- **MediaPipe face landmarker** (used for quality flags in the skin stage)
  — downloaded automatically by `pipeline/derived/skin_quality_check.py`
  into `~/.cache/mediapipe/`.

## 3. Manual downloads (one-time)

### DECA (3D mesh reconstruction)
1. Clone the repo: `git clone https://github.com/yfeng95/DECA`
2. DECA needs the FLAME model, which requires free registration at
   https://flame.is.tue.mpg.de — follow DECA's own README
   ("Prepare data") for exactly which files go where under `DECA/data/`.
3. Pass the cloned path to the pipeline: `--deca-repo /path/to/DECA`

### MiVOLO (perceived age estimation)
1. Clone the repo: `git clone https://github.com/WildChlamydia/MiVOLO`
2. Download the two checkpoints MiVOLO's own README links to:
   - `yolov8x_person_face.pt` (YOLOv8 face/person detector)
   - `model_imdb_cross_person_4.24_99.46.pth.tar` (MiVOLO age/gender model)
3. Pass all three paths to the pipeline:
   `--mivolo-repo /path/to/MiVOLO --mivolo-detector-weights /path/to/yolov8x_person_face.pt --mivolo-checkpoint /path/to/model_imdb_cross_person_4.24_99.46.pth.tar`

### BiSeNet face parsing (68pt landmarks only — not needed if using `--landmarks 106`)
1. Get the `79999_iter.pth` checkpoint from
   https://github.com/zllrunning/face-parsing.PyTorch (see its README for
   the current download link).
2. `pip install face-parsing` (or install that repo directly) so the
   `face_parsing` CLI command is on PATH.
3. Pass the checkpoint path: `--face-parsing-ckpt /path/to/79999_iter.pth`

## 4. Quick smoke test without any of the above

The 2D-only path with 106pt landmarks needs nothing beyond
`requirements-pipeline.txt`:
```bash
python run_pipeline.py --input photos/ --work-dir work/ \
    --landmarks 106 --skip-3d --skip-derived
```
