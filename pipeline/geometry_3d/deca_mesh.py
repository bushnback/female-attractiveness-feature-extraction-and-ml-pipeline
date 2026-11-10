#!/usr/bin/env python3
"""
deca_mesh.py
============
Run DECA 3D face reconstruction (FLAME-topology mesh: 5023 vertices, 9976
faces) on a folder of face photos.

REQUIRES (not bundled — see setup/README.md):
  - A clone of the DECA repo (https://github.com/yfeng95/DECA), importable
    as `decalib` — pass its path via --deca-repo or set DECA_REPO_DIR.
  - DECA's pretrained model data (FLAME model + DECA checkpoint) placed in
    <deca-repo>/data/. FLAME requires free registration before download
    (licensing), so this step cannot be scripted — see DECA's own README.
  - conda env with pytorch3d, torch, etc. (see requirements/deca-env.txt)

Output per subject (in <output>/<subject>/):
  mesh.ply, mesh.obj  — FLAME mesh (vertices in mm)
  kpt68.npy           — 68 3D landmarks
  identity.npy        — FLAME shape parameters

Usage:
    python deca_mesh.py --input /path/to/photos --output /path/to/out \
        --deca-repo /path/to/DECA
"""

import argparse
import struct
import io
import sys
from pathlib import Path

import numpy as np
import cv2
import torch


def vertices_faces_to_ply_bytes(vertices, faces):
    """Convert numpy vertices and faces to binary PLY format."""
    nv = len(vertices)
    nf = len(faces)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    buf = io.BytesIO()
    buf.write(header.encode('ascii'))
    for v in vertices:
        buf.write(struct.pack('<fff', float(v[0]), float(v[1]), float(v[2])))
    for f in faces:
        buf.write(struct.pack('<B', 3))
        buf.write(struct.pack('<iii', int(f[0]), int(f[1]), int(f[2])))
    return buf.getvalue()


def run(input_dir, output_dir, deca_repo, device="cpu"):
    """Run DECA on every image in input_dir, skipping subjects that already
    have mesh.ply in output_dir."""
    deca_repo = Path(deca_repo).resolve()
    sys.path.insert(0, str(deca_repo))
    from decalib.deca import DECA
    from decalib.datasets import datasets
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    missing = []
    for p in image_paths:
        if (output_dir / p.stem / "mesh.ply").exists():
            print(f"  {p.stem}: already has output, skipping")
        else:
            missing.append(p)

    if not missing:
        print("All subjects already have DECA output.")
        return {p.stem: output_dir / p.stem for p in image_paths}

    print(f"Initialising DECA (device={device})...")
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)

    results = {}
    for idx, img_path in enumerate(missing, 1):
        name = img_path.stem
        print(f"[{idx}/{len(missing)}] Processing: {name}")

        testdata = datasets.TestData(str(img_path), iscrop=True, face_detector='fan')
        if len(testdata) == 0:
            print(f"  [ERROR] No face detected in {img_path.name}")
            continue

        images = testdata[0]['image'].to(device)[None, ...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)

        out_dir = output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        vertices = opdict['verts'][0].cpu().numpy() * 1000.0  # metres -> mm
        faces = deca.render.faces[0].cpu().numpy()

        try:
            texture = util.tensor2image(opdict['uv_texture_gt'][0])
            uvcoords = deca.render.raw_uvcoords[0].cpu().numpy()
            uvfaces = deca.render.uvfaces[0].cpu().numpy()
            normal_map = util.tensor2image(opdict['uv_detail_normals'][0] * 0.5 + 0.5)
            util.write_obj(str(out_dir / f"{name}.obj"), vertices, faces,
                            texture=texture, uvcoords=uvcoords,
                            uvfaces=uvfaces, normal_map=normal_map)
            util.write_obj(str(out_dir / "mesh.obj"), vertices, faces)
        except Exception as e:
            print(f"  [WARN] OBJ save issue: {e}")
            util.write_obj(str(out_dir / "mesh.obj"), vertices, faces)

        ply_bytes = vertices_faces_to_ply_bytes(vertices, faces)
        with open(out_dir / "mesh.ply", "wb") as f:
            f.write(ply_bytes)
        print(f"  Saved mesh.ply ({len(vertices)} verts, {len(faces)} faces)")

        landmarks3d = opdict['landmarks3d'][0].cpu().numpy()
        np.save(str(out_dir / "kpt68.npy"), landmarks3d)

        shape_params = codedict['shape'][0].cpu().numpy()
        np.save(str(out_dir / "identity.npy"), shape_params)

        try:
            vis_img = deca.visualize(visdict)
            cv2.imwrite(str(out_dir / f"{name}_vis.jpg"), vis_img)
        except Exception as e:
            print(f"  [WARN] Vis save issue: {e}")

        results[name] = out_dir

    print(f"Done. {len(results)}/{len(image_paths)} subjects have DECA output.")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Folder of .jpg/.png photos")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--deca-repo", required=True, help="Path to a clone of github.com/yfeng95/DECA")
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda' (GPU strongly recommended)")
    args = ap.parse_args()
    run(args.input, args.output, args.deca_repo, device=args.device)


if __name__ == "__main__":
    main()
