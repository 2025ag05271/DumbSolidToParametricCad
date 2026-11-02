"""Predict a design history for a STEP file.

Usage:
    .venv\\Scripts\\python.exe -m s2pnet.predict --ckpt runs/base/last.pt --step part.stp --out out/part
    .venv\\Scripts\\python.exe -m s2pnet.predict ... --samples 8 --temperature 0.8   # generate-and-verify

Writes <out>.json (design history), <out>.txt (readable tokens) and <out>_rebuilt.step, and prints
the IoU between the rebuilt solid and the input. With ``--samples N`` the greedy prediction and
N-1 sampled ones are rebuilt and the one that best matches the input is kept.

Note: the model was trained on synthetic parts in their generated placement (absolute mm, axis-
aligned sketch planes). Real parts in other placements need placement canonicalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np
import torch

from cadseq.dataset import COORD_SCALE, collate, to_device
from cadseq.graph import brep_to_graph_arrays
from cadseq.metrics import mc_iou
from cadseq.tokenizer import to_text

from .model import Solid2Param
from .verify import decode_candidates, generate_candidates, select_best


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
    ap.add_argument("--samples", type=int, default=1, help="candidates to generate and verify")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Solid2Param.from_checkpoint(ckpt["model"]).to(device).eval()

    solids = cq.importers.importStep(str(args.step)).solids().vals()
    if len(solids) != 1:
        raise SystemExit(f"expected one solid in {args.step}, found {len(solids)}")
    shape = solids[0]
    batch = to_device(collate([graph_item(shape)]), device)
    cands = generate_candidates(model, batch, args.samples, args.temperature, args.seed)[0]
    keep, seqs = decode_candidates(cands)
    if not seqs:
        raise SystemExit("no prediction decoded")

    best, body, history, infos = select_best(seqs, shape)
    if best is None:
        raise SystemExit(f"no prediction rebuilt: {infos[0].get('error_lenient') or infos[0].get('error')}")
    tokens, info = cands[keep[best]], infos[best]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".txt").write_text(to_text(tokens), encoding="utf-8")
    # Saved history = the one that strictly rebuilds the output solid (repaired if lenient).
    args.out.with_suffix(".json").write_text(json.dumps(history.model_dump(), indent=1), encoding="utf-8")
    print(to_text(tokens))
    rebuilt = args.out.parent / f"{args.out.name}_rebuilt.step"
    cq.exporters.export(cq.Workplane().add(body), str(rebuilt), cq.exporters.ExportTypes.STEP)
    mode = "strict" if info["built_strict"] else f"lenient, skipped {info['skipped']} feature(s), repaired history saved"
    print(f"candidate {keep[best]} of {len(cands)} unique ({mode}) | features {len(history.features)} | "
          f"IoU vs input {mc_iou(body, shape):.4f} | wrote {rebuilt}")


if __name__ == "__main__":
    main()
