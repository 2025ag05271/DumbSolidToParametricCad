"""Predict a design history for a STEP file.

Usage:
    .venv\\Scripts\\python.exe -m s2pnet.predict --ckpt runs/base/best.pt --step part.stp --out out/part

Writes <out>.json (design history), <out>.txt (readable tokens) and <out>_rebuilt.step, and prints
the IoU between the rebuilt solid and the input.

Note: the model was trained on synthetic parts in their generated placement (absolute mm, axis-
aligned sketch planes). Real parts in other placements need placement canonicalization (step 6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np
import torch

from cadseq.build import QUANTIZED_EDGE_TOL, build, build_lenient
from cadseq.dataset import COORD_SCALE, collate, to_device
from cadseq.graph import brep_to_graph_arrays
from cadseq.metrics import mc_iou
from cadseq.tokenizer import DecodeError, decode, to_text

from .model import Solid2Param


def graph_item(shape: cq.Shape, uv: int = 10) -> dict:
    a = brep_to_graph_arrays(shape.wrapped, uv)
    a["node_x"][..., :3] *= COORD_SCALE
    a["edge_x"][..., :3] *= COORD_SCALE
    return {"id": "input", "node_x": torch.from_numpy(a["node_x"]), "edge_x": torch.from_numpy(a["edge_x"]),
            "edge_index": torch.from_numpy(a["edge_index"].astype(np.int64)),
            "tokens": torch.zeros(1, dtype=torch.long)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--step", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output path prefix")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Solid2Param.from_checkpoint(ckpt["model"]).to(device).eval()

    solids = cq.importers.importStep(str(args.step)).solids().vals()
    if len(solids) != 1:
        raise SystemExit(f"expected one solid in {args.step}, found {len(solids)}")
    shape = solids[0]
    batch = to_device(collate([graph_item(shape)]), device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        seq = model.generate(batch)[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".txt").write_text(to_text(seq), encoding="utf-8")
    try:
        pred = decode(seq)
    except DecodeError as e:
        raise SystemExit(f"prediction did not decode: {e}")
    args.out.with_suffix(".json").write_text(json.dumps(pred.model_dump(), indent=1), encoding="utf-8")
    print(to_text(seq))
    skipped: list[str] = []
    try:
        body = build(pred, edge_tol=QUANTIZED_EDGE_TOL)
    except Exception as e:
        print(f"strict rebuild failed ({e}); using lenient rebuild")
        try:
            body, skipped = build_lenient(pred)
        except Exception as e2:
            raise SystemExit(f"prediction did not rebuild: {e2}")
    rebuilt = args.out.parent / f"{args.out.name}_rebuilt.step"
    cq.exporters.export(cq.Workplane().add(body), str(rebuilt), cq.exporters.ExportTypes.STEP)
    print(f"features {len(pred.features)} (skipped {skipped or 'none'}) | "
          f"IoU vs input {mc_iou(body, shape):.4f} | wrote {rebuilt}")


if __name__ == "__main__":
    main()
