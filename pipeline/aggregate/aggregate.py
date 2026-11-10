#!/usr/bin/env python3
"""
aggregate.py
============
Merge every stage's per-subject CSV output into one ML-ready
ml_features.csv, keyed by subject_id, with each source's columns
prefixed (e.g. lm2d__face_width_jaw, skin__overall_skin_quality).

This mirrors the "Feature Aggregation" box in the pipeline architecture:
every 2D/3D/derived feature stream merges here into a single row per
subject.

Usage (as a library — call from run_pipeline.py):
    from aggregate import aggregate, Source
    aggregate([
        Source("lm2d", "path/to/2d_features_summary.csv", id_col="subject"),
        Source("shape", "path/to/all_faces_summary.csv", id_col="person_id",
               drop_cols=["conf_2nd_shape"]),
        ...
    ], output_csv="ml_features.csv")
"""

import argparse
import csv
import json
from collections import OrderedDict, namedtuple
from pathlib import Path

# prefix     — column prefix in the merged CSV (e.g. "lm2d", "skin")
# csv_path   — path to the source CSV
# id_col     — name of the subject-id column in that CSV
# drop_cols  — columns to exclude from that source (e.g. redundant labels)
# id_strip_ext — if True, strip a file extension from the id column
#   (some CLIP scripts write "image.jpg" as the id; everything else uses
#   the bare subject id)
Source = namedtuple("Source", ["prefix", "csv_path", "id_col", "drop_cols", "id_strip_ext"])


def source(prefix, csv_path, id_col="subject_id", drop_cols=None, id_strip_ext=False):
    """Convenience constructor with sensible defaults for Source."""
    return Source(prefix, Path(csv_path), id_col, drop_cols or [], id_strip_ext)


def aggregate(sources, output_csv):
    """
    sources: list of Source namedtuples (use `source(...)` to build them).
    Any source whose csv_path doesn't exist is skipped with a warning —
    useful when only a subset of stages were run.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_fields = []
    all_data = {}

    for src in sources:
        if not src.csv_path.exists():
            print(f"  [{src.prefix}] CSV not found: {src.csv_path} — skipped")
            continue

        drop_set = set(src.drop_cols)
        with open(src.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            feature_cols = [c for c in reader.fieldnames if c != src.id_col and c not in drop_set]
            prefixed = [f"{src.prefix}__{c}" for c in feature_cols]
            all_fields.extend(prefixed)

            n_rows = 0
            for row in reader:
                sid = row[src.id_col].strip()
                if src.id_strip_ext:
                    sid = sid.rsplit(".", 1)[0]
                if not sid:
                    continue
                if sid not in all_data:
                    all_data[sid] = OrderedDict()
                for orig, pref in zip(feature_cols, prefixed):
                    all_data[sid][pref] = row[orig]
                n_rows += 1

        print(f"  [{src.prefix}] {len(prefixed)} features, {n_rows} subjects")

    if not all_data:
        print("[ERROR] No data to aggregate")
        return None

    subject_ids = sorted(all_data.keys())
    header = ["subject_id"] + all_fields

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sid in subject_ids:
            writer.writerow([sid] + [all_data[sid].get(col, "") for col in all_fields])

    n_complete = sum(1 for sid in subject_ids if all(col in all_data[sid] for col in all_fields))
    print(f"\n  Output: {output_csv}")
    print(f"  Subjects: {len(subject_ids)}")
    print(f"  Features: {len(all_fields)}")
    print(f"  Fully complete rows: {n_complete}/{len(subject_ids)}")

    return output_csv


def main():
    """
    CLI form: pass a JSON manifest describing the sources, e.g.:
      [
        {"prefix": "lm2d", "csv_path": "...", "id_col": "subject"},
        {"prefix": "skin", "csv_path": "...", "id_col": "image_id"}
      ]
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Path to a JSON list of source definitions")
    ap.add_argument("--output", required=True, help="Output ml_features.csv path")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    sources = [source(**entry) for entry in manifest]
    aggregate(sources, args.output)


if __name__ == "__main__":
    main()
