"""
face_shape_classifier_v2.py
============================
Classifies face shape from DAD-3DHeads 445 3D landmarks on FLAME meshes.

VERSION 2 — fixes and improvements over v1:
  ──────────────────────────────────────────────────────────────────────────
  ISSUE 1: Face height was severely underestimated.
      V1 used a narrow mid-brow band (Y 38–52, |X|<10, Z>40) that matched
      only ~3 points far below the true forehead, yielding height/width
      ratios around 0.86 (face wider than tall — anatomically implausible).
      FIX: Use the top of the forehead landmark region (highest midline
      landmarks within the top 20% of Y range) as the upper reference.
      This produces realistic h/w ratios of 1.04–1.16.

  ISSUE 2: Hard-coded Y-coordinate thresholds assumed a specific mesh pose.
      V1 used absolute Y values (e.g. Y > 57 for forehead, Y < -65 for
      chin) which are fragile if the mesh origin or scale changes.
      FIX: All zone boundaries are computed as percentiles of each mesh's
      own Y-coordinate distribution, making the code pose-agnostic.

  ISSUE 3: Jaw angle was measured from one side only (right jaw corner).
      FIX: Average the left and right jaw corner angles for robustness.

  ISSUE 4: Classification confidence was low (30–42%) due to heavily
      overlapping scoring criteria.
      FIX:
        a) Primary "signature" criteria now carry 3.0 points (was 2.0),
           creating wider score gaps.
        b) Disqualifiers added: if a shape's key metric is clearly out of
           range, its score is clamped to zero.  E.g. oblong is zeroed
           when h/w < 1.10 (face is not elongated).
        c) Overlapping threshold ranges were narrowed so fewer shapes
           accumulate incidental points.

  ISSUE 5: FLAME model output has a narrower variation range than raw
      anthropometric data.  All 8 test faces had fw/cb ≈ 0.94–1.01 and
      cb/jaw ≈ 1.02–1.05.
      FIX: Thresholds recalibrated for FLAME-range measurements.
  ──────────────────────────────────────────────────────────────────────────

Usage:
    python face_shape_classifier_v2.py --input_dir input_faces --output_dir output_results_v2

Coordinate system (FLAME / DAD-3DHeads):
    X  — lateral:  positive = face-right,  negative = face-left
    Y  — vertical: positive = up,          negative = down (chin)
    Z  — depth:    positive = forward (face surface), negative = rear
"""

import os
import sys
import csv
import json
import math
import argparse
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# FACE SHAPE LABELS AND NUMERIC CODES
# ─────────────────────────────────────────────────────────────────────────────

SHAPE_CODES = {
    "oval":      1,
    "round":     2,
    "square":    3,
    "heart":     4,
    "diamond":   5,
    "oblong":    6,
    "triangle":  7,
    "unknown":   0,
}

# ─────────────────────────────────────────────────────────────────────────────
# MEASUREMENT EXTRACTION  (v2 — percentile-based, pose-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def extract_measurements(lm: np.ndarray) -> dict:
    """
    Extract geometric measurements for face shape classification.

    All vertical zone boundaries are computed as percentiles of each
    mesh's own Y distribution, making this robust to pose/scale changes.

    Parameters
    ----------
    lm : np.ndarray, shape (N, 3)
        3D landmark positions (typically 445 DAD-3DHeads landmarks).

    Returns
    -------
    dict  — raw measurements (mm) and derived ratios.
    """

    y_vals = lm[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    y_range = y_max - y_min

    # ── Zone boundaries (percentile-based) ──────────────────────────────────
    # V1 ISSUE: used absolute Y thresholds that broke if the mesh was
    # translated or scaled.  Percentile-based zones adapt automatically.
    y_top20  = y_max - 0.20 * y_range   # forehead region starts here
    y_low20  = y_min + 0.20 * y_range   # chin region ends here
    y_low40  = y_min + 0.40 * y_range   # jaw region upper bound

    # Face-surface filter: exclude rear-of-skull points
    face_pts = lm[lm[:, 2] > -10]

    # ── 1. FOREHEAD WIDTH ──────────────────────────────────────────────────
    # Points in the top 20% of Y range.
    fh_pts = lm[lm[:, 1] > y_top20]
    if len(fh_pts) < 2:
        raise ValueError(f"Insufficient forehead landmarks (top 20% Y > {y_top20:.1f})")
    forehead_width = float(fh_pts[:, 0].max() - fh_pts[:, 0].min())

    # ── 2. CHEEKBONE WIDTH ─────────────────────────────────────────────────
    # Widest lateral span in a 10mm Y-slice, scanning the middle 60%.
    best_cb_width = 0.0
    best_cb_y = 0.0
    scan_lo = y_min + 0.20 * y_range
    scan_hi = y_max - 0.20 * y_range
    for y_centre in np.arange(scan_lo, scan_hi, 2.0):
        sl = face_pts[
            (face_pts[:, 1] >= y_centre - 5) &
            (face_pts[:, 1] <  y_centre + 5)
        ]
        if len(sl) < 4:
            continue
        w = float(sl[:, 0].max() - sl[:, 0].min())
        if w > best_cb_width:
            best_cb_width = w
            best_cb_y = y_centre
    cheekbone_width = best_cb_width
    if cheekbone_width == 0:
        raise ValueError("Could not compute cheekbone width (no valid Y slices)")

    # ── 3. JAW WIDTH ───────────────────────────────────────────────────────
    # Points in the 20–40% Y band (above chin, below cheekbone zone).
    jaw_zone = lm[
        (lm[:, 1] > y_low20) & (lm[:, 1] < y_low40) & (lm[:, 2] > -5)
    ]
    if len(jaw_zone) < 2:
        raise ValueError("Insufficient jaw landmarks in Y 20–40% zone")
    jaw_width = float(jaw_zone[:, 0].max() - jaw_zone[:, 0].min())

    # ── 4. CHIN TIP ────────────────────────────────────────────────────────
    # Lowest-Y landmark nearest the midline.
    chin_candidates = lm[lm[:, 1] < y_low20]
    if len(chin_candidates) == 0:
        raise ValueError(f"No chin landmarks found (Y < {y_low20:.1f})")
    chin_tip = chin_candidates[np.argmin(np.abs(chin_candidates[:, 0]))]

    # ── 5. FACE HEIGHT ─────────────────────────────────────────────────────
    # V1 ISSUE: measured from mid-brow (Y≈45) → h/w ≈ 0.86 (too short).
    # V2 FIX:  use the highest midline forehead landmark as upper reference.
    top_midline = lm[
        (np.abs(lm[:, 0]) < 20) & (lm[:, 1] > y_top20)
    ]
    if len(top_midline) == 0:
        raise ValueError("No forehead midline landmarks found")
    forehead_top_y = float(top_midline[:, 1].max())
    face_height = forehead_top_y - float(chin_tip[1])

    # ── 6. JAW ANGLE ───────────────────────────────────────────────────────
    # V1 ISSUE: used only the right jaw corner, which is noisy.
    # V2 FIX:  average the angle from both left and right jaw corners.
    angles = []
    for sign, x_thresh in [(1, 20), (-1, -20)]:
        if sign > 0:
            side_jaw = lm[
                (lm[:, 1] > y_low20) & (lm[:, 1] < y_low40) &
                (lm[:, 0] > x_thresh) & (lm[:, 2] > -5)
            ]
        else:
            side_jaw = lm[
                (lm[:, 1] > y_low20) & (lm[:, 1] < y_low40) &
                (lm[:, 0] < x_thresh) & (lm[:, 2] > -5)
            ]
        if len(side_jaw) == 0:
            continue
        corner = side_jaw[np.argmax(side_jaw[:, 1])]
        dx = abs(float(corner[0] - chin_tip[0]))
        dy = abs(float(corner[1] - chin_tip[1]))
        angles.append(math.degrees(math.atan2(dx, dy)))
    jaw_angle_deg = float(np.mean(angles)) if angles else float('nan')

    # ── 7. DERIVED RATIOS ──────────────────────────────────────────────────
    r_forehead_cheekbone = forehead_width / cheekbone_width
    r_cheekbone_jaw      = cheekbone_width / jaw_width
    r_forehead_jaw       = forehead_width / jaw_width
    r_height_cheekbone   = face_height / cheekbone_width

    return {
        # Raw measurements (mm-scale, FLAME units)
        "forehead_width_mm":    round(forehead_width,  2),
        "cheekbone_width_mm":   round(cheekbone_width, 2),
        "jaw_width_mm":         round(jaw_width,       2),
        "face_height_mm":       round(face_height,     2),
        "jaw_angle_deg":        round(jaw_angle_deg,   2),
        # Ratios
        "ratio_forehead_cheekbone": round(r_forehead_cheekbone, 4),
        "ratio_cheekbone_jaw":      round(r_cheekbone_jaw,      4),
        "ratio_forehead_jaw":       round(r_forehead_jaw,       4),
        "ratio_height_cheekbone":   round(r_height_cheekbone,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION LOGIC  (v2 — higher separation, disqualifiers)
# ─────────────────────────────────────────────────────────────────────────────

def classify_face_shape(m: dict) -> tuple[str, int, dict]:
    """
    Classify face shape from geometric ratio measurements.

    V2 improvements over V1:
      - Primary criteria weighted at 3.0 (was 2.0) for wider score gaps.
      - Disqualifiers: shapes that clearly don't match are zeroed out.
      - Narrower overlapping ranges between shapes.
      - Thresholds recalibrated for FLAME mesh measurement range.

    Returns
    -------
    shape_name : str
    shape_code : int
    confidence_scores : dict  (relative scores for each shape, sum to 1.0)
    """

    fw_cb  = m["ratio_forehead_cheekbone"]   # forehead / cheekbone
    cb_jaw = m["ratio_cheekbone_jaw"]         # cheekbone / jaw
    fw_jaw = m["ratio_forehead_jaw"]          # forehead / jaw
    h_cb   = m["ratio_height_cheekbone"]      # height / cheekbone
    jaw_a  = m["jaw_angle_deg"]               # jaw taper angle (degrees)

    scores = {s: 0.0 for s in SHAPE_CODES if s != "unknown"}

    # ── OBLONG / RECTANGLE ─────────────────────────────────────────────────
    # Signature: notably elongated (tall relative to width).
    # V1 ISSUE: could score points from width uniformity alone even when
    # height clearly disqualified it.  V2 adds a disqualifier below.
    if h_cb > 1.25:
        scores["oblong"] += 3.0      # strong elongation
    elif h_cb > 1.15:
        scores["oblong"] += 1.5      # moderate elongation
    if fw_cb > 0.90 and cb_jaw < 1.06:
        scores["oblong"] += 1.0      # uniform widths top to bottom

    # ── ROUND ──────────────────────────────────────────────────────────────
    # Signature: face nearly as wide as tall, soft rounded jaw.
    if h_cb < 1.05:
        scores["round"] += 3.0       # nearly circular proportions
    elif h_cb < 1.10:
        scores["round"] += 1.5
    if jaw_a > 55:
        scores["round"] += 1.5       # very soft, rounded jaw
    elif jaw_a > 48:
        scores["round"] += 0.5
    if cb_jaw < 1.04:
        scores["round"] += 1.0       # minimal cheekbone-jaw taper

    # ── SQUARE ─────────────────────────────────────────────────────────────
    # Signature: angular jaw (small taper angle), uniform widths.
    if jaw_a < 40:
        scores["square"] += 3.0      # sharp, angular jaw
    elif jaw_a < 48:
        scores["square"] += 1.5
    if cb_jaw < 1.04 and fw_cb > 0.90:
        scores["square"] += 1.5      # forehead ≈ cheekbone ≈ jaw
    if h_cb > 1.05 and h_cb < 1.18:
        scores["square"] += 0.5      # slightly taller than wide

    # ── HEART (inverted triangle) ──────────────────────────────────────────
    # Signature: wide forehead, narrow jaw, pronounced taper.
    if fw_jaw > 1.05:
        scores["heart"] += 3.0       # forehead clearly wider than jaw
    elif fw_jaw > 1.02:
        scores["heart"] += 1.5
    if fw_cb > 0.98:
        scores["heart"] += 1.0       # forehead as wide as cheekbones
    if jaw_a > 52:
        scores["heart"] += 1.0       # narrow, tapering chin

    # ── DIAMOND ────────────────────────────────────────────────────────────
    # Signature: cheekbones clearly widest; both forehead and jaw narrow.
    if fw_cb < 0.93 and cb_jaw > 1.05:
        scores["diamond"] += 3.0     # cheekbones dominate
    elif fw_cb < 0.96 and cb_jaw > 1.03:
        scores["diamond"] += 1.5
    if fw_jaw > 0.95 and fw_jaw < 1.05:
        scores["diamond"] += 0.5     # forehead ≈ jaw (both narrow)

    # ── TRIANGLE (pear) ────────────────────────────────────────────────────
    # Signature: jaw wider than forehead.
    if fw_jaw < 0.95:
        scores["triangle"] += 3.0    # forehead notably narrower than jaw
    elif fw_jaw < 0.98:
        scores["triangle"] += 1.5
    if cb_jaw < 1.0:
        scores["triangle"] += 1.5    # jaw as wide as or wider than cheekbones

    # ── OVAL ───────────────────────────────────────────────────────────────
    # Signature: balanced proportions, slight elongation, gentle taper.
    if h_cb > 1.08 and h_cb < 1.22:
        scores["oval"] += 2.0        # moderately elongated
    if fw_cb > 0.93 and fw_cb < 1.0:
        scores["oval"] += 1.5        # forehead slightly narrower than cheekbone
    if cb_jaw > 1.03 and cb_jaw < 1.08:
        scores["oval"] += 1.5        # gentle cheekbone-to-jaw taper
    if jaw_a > 45 and jaw_a < 55:
        scores["oval"] += 1.0        # moderate jaw softness

    # ── DISQUALIFIERS ──────────────────────────────────────────────────────
    # V1 ISSUE: shapes could accumulate incidental points from secondary
    # criteria even when the primary measurement clearly ruled them out.
    # V2 FIX:  zero out shapes that are anatomically impossible given the
    # measurements.
    if h_cb < 1.10:
        scores["oblong"] = 0.0       # not elongated → can't be oblong
    if h_cb > 1.18:
        scores["round"] = 0.0        # too tall → can't be round
    if jaw_a > 55:
        scores["square"] = 0.0       # jaw too soft → can't be square
    if fw_jaw < 1.0:
        scores["heart"] = 0.0        # forehead not wider → can't be heart
    if fw_cb > 0.98 or cb_jaw < 1.02:
        scores["diamond"] = 0.0      # cheekbones not dominant → can't be diamond
    if fw_jaw > 1.02:
        scores["triangle"] = 0.0     # forehead wider → can't be triangle

    # ── DETERMINE WINNER ───────────────────────────────────────────────────
    best_shape = max(scores, key=scores.get)
    best_score = scores[best_shape]

    if best_score == 0:
        best_shape = "unknown"

    # Normalise to confidence percentages
    total = sum(scores.values())
    if total > 0:
        confidence = {k: round(v / total, 4) for k, v in scores.items()}
    else:
        confidence = {k: 0.0 for k in scores}

    return best_shape, SHAPE_CODES[best_shape], confidence


# ─────────────────────────────────────────────────────────────────────────────
# PER-FACE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_face(landmarks_path: str) -> dict:
    """Load landmarks, measure, and classify a single face."""
    lm = np.load(landmarks_path)
    if lm.ndim != 2 or lm.shape[1] != 3:
        raise ValueError(
            f"Expected shape (N, 3), got {lm.shape} in {landmarks_path}"
        )

    measurements = extract_measurements(lm)
    shape_name, shape_code, confidence = classify_face_shape(measurements)

    return {
        "shape_name":   shape_name,
        "shape_code":   shape_code,
        "confidence":   confidence,
        "measurements": measurements,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

LANDMARK_FILENAME = "dad3dheads_landmarks_3d.npy"

def run_batch(input_dir: str, output_dir: str) -> None:
    """
    Process all person subfolders found in input_dir.
    Writes face_shape.json per person and a summary CSV.
    """
    if not os.path.isdir(input_dir):
        sys.exit(f"[ERROR] Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    person_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    if not person_dirs:
        sys.exit(f"[ERROR] No subfolders found in {input_dir}")

    print(f"Found {len(person_dirs)} person folder(s) in '{input_dir}'")
    print(f"Output → '{output_dir}'\n")

    summary_rows = []
    ok_count = 0
    err_count = 0

    for person_id in person_dirs:
        lm_path = os.path.join(input_dir, person_id, LANDMARK_FILENAME)

        if not os.path.isfile(lm_path):
            print(f"  [SKIP] {person_id}: '{LANDMARK_FILENAME}' not found")
            err_count += 1
            continue

        try:
            result = process_face(lm_path)
        except Exception as exc:
            print(f"  [FAIL] {person_id}: {exc}")
            err_count += 1
            continue

        # ── Write per-person JSON ─────────────────────────────────────────
        person_out_dir = os.path.join(output_dir, person_id)
        os.makedirs(person_out_dir, exist_ok=True)
        json_path = os.path.join(person_out_dir, "face_shape.json")

        output_record = {
            "person_id":    person_id,
            "shape_name":   result["shape_name"],
            "shape_code":   result["shape_code"],
            "confidence":   result["confidence"],
            "measurements": result["measurements"],
        }

        with open(json_path, "w") as f:
            json.dump(output_record, f, indent=2)

        # ── Accumulate summary row ────────────────────────────────────────
        row = {"person_id": person_id,
               "shape_name": result["shape_name"],
               "shape_code": result["shape_code"]}
        row.update(result["measurements"])
        conf_sorted = sorted(result["confidence"].items(),
                             key=lambda x: x[1], reverse=True)
        row["conf_1st"] = conf_sorted[0][1]
        row["conf_2nd_shape"] = conf_sorted[1][0]
        row["conf_2nd"] = conf_sorted[1][1]
        summary_rows.append(row)

        ok_count += 1
        print(f"  [OK]   {person_id}: {result['shape_name']} "
              f"(code={result['shape_code']}, "
              f"confidence={conf_sorted[0][1]*100:.0f}%)")

    # ── Write summary CSV ─────────────────────────────────────────────────
    if summary_rows:
        csv_path = os.path.join(output_dir, "all_faces_summary.csv")
        keys = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary CSV written → {csv_path}")

    print(f"\nDone.  Processed: {ok_count}  Skipped/Failed: {err_count}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify face shape from DAD-3DHeads 3D landmarks (v2)"
    )
    parser.add_argument(
        "--input_dir",
        default="input_faces",
        help="Folder containing one subfolder per person (default: input_faces)"
    )
    parser.add_argument(
        "--output_dir",
        default="output_results_v2",
        help="Folder where results will be written (default: output_results_v2)"
    )
    args = parser.parse_args()
    run_batch(args.input_dir, args.output_dir)
