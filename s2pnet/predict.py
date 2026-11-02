"""Predict a design history for a STEP file.

Usage:
    .venv\\Scripts\\python.exe -m s2pnet.predict --ckpt runs/base/last.pt --step part.stp --out out/part
    .venv\\Scripts\\python.exe -m s2pnet.predict ... --samples 8 --temperature 0.8   # generate-and-verify

Writes <out>.json (design history), <out>.txt (readable tokens) and <out>_rebuilt.step, and prints
the IoU between the rebuilt solid and the input. With ``--samples N`` the greedy prediction and
N-1 sampled ones are rebuilt and the one that best matches the input is kept.

Note: v1 models were trained on synthetic parts in their generated placement (absolute mm, axis-
aligned sketch planes). v2 copy models are trained with axis-aligned rotations/reflections and
±100 mm translations, and the input is re-centred on a 1 mm grid (``--no-center`` to disable);
arbitrarily rotated parts still need axis alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np
import torch

from cadseq.dataset import COORD_SCALE, collate, collate_copy, scale_features, to_device
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


def copy_item(shape: cq.Shape, uv: int = 10) -> dict:
    """Input item for v2 (copy) models: graph + analytic attributes + candidate values, exactly as
    ``CopySeqDataset`` builds them (no target tokens)."""
    from cadseq.candidates import candidate_values, extract_primitives, value_features
    from cadseq.dataset import CopySeqDataset

    MAX_CANDIDATES = CopySeqDataset.MAX_CANDIDATES
    a = brep_to_graph_arrays(shape.wrapped, uv, with_attributes=True)
    raw = scale_features({k: a[k] for k in ("node_x", "edge_x", "face_attr", "edge_attr")})
    values, tags = candidate_values(extract_primitives(shape))
    if len(values) > MAX_CANDIDATES:
        keep = np.sort(np.argsort(np.abs(values))[:MAX_CANDIDATES])
        values, tags = values[keep], tags[keep]
    return {"id": "input", "pose": (np.eye(3), np.zeros(3)),
            "edge_index": torch.from_numpy(a["edge_index"].astype(np.int64)),
            "tokens": torch.zeros(1, dtype=torch.long),
            **{k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in raw.items()},
            "cand_feat": torch.from_numpy(np.concatenate([value_features(values), tags.astype(np.float32)], 1)),
            "cand_quant": torch.from_numpy(np.round(values * 20).astype(np.int64)),
            "cand_value": torch.from_numpy(values),
            "num_pos": torch.zeros(0, dtype=torch.long), "copy_pairs": torch.zeros(0, 2, dtype=torch.long)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--step", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output path prefix")
    ap.add_argument("--samples", type=int, default=1, help="candidates to generate and verify")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-center", action="store_true",
                    help="copy models: don't move the bounding-box centre (rounded to 1 mm) to the origin")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Solid2Param.from_checkpoint(ckpt["model"]).to(device).eval()

    solids = cq.importers.importStep(str(args.step)).solids().vals()
    if len(solids) != 1:
        raise SystemExit(f"expected one solid in {args.step}, found {len(solids)}")
    shape = solids[0]
    if model.cfg.copy:
        # Training covered placements within ~±100 mm of the generator's; far-away parts are moved
        # near the origin by a whole-mm translation (keeps every input value on the 0.05 mm grid).
        t = np.zeros(3)
        if not args.no_center:
            t = -np.round(np.array(shape.BoundingBox().center.toTuple()))
        posed = shape.translate(cq.Vector(*t)) if np.any(t) else shape
        batch = to_device(collate_copy([copy_item(posed)]), device)
    else:
        batch = to_device(collate([graph_item(shape)]), device)
    cands = generate_candidates(model, batch, args.samples, args.temperature, args.seed)[0]
    keep, seqs = decode_candidates(cands)
    if model.cfg.copy and np.any(t):
        from cadseq.pose import transform_sequence

        # Back to the input placement (may lie outside the tokenizer range, so no re-tokenizing).
        seqs = [transform_sequence(s, np.eye(3), -t, canonical=False) for s in seqs]
    if not seqs:
        raise SystemExit("no prediction decoded")

    best, body, history, infos = select_best(seqs, shape)
    if best is None:
        raise SystemExit(f"no prediction rebuilt: {infos[0].get('error_lenient') or infos[0].get('error')}")
    tokens, info = cands[keep[best]], infos[best]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = to_text(tokens)
    if model.cfg.copy and np.any(t):
        text = f"# tokens in the centred frame: input translated by {t.tolist()} mm\n" + text
    args.out.with_suffix(".txt").write_text(text, encoding="utf-8")
    # Saved history = the one that strictly rebuilds the output solid (repaired if lenient).
    args.out.with_suffix(".json").write_text(json.dumps(history.model_dump(), indent=1), encoding="utf-8")
    print(text)
    rebuilt = args.out.parent / f"{args.out.name}_rebuilt.step"
    cq.exporters.export(cq.Workplane().add(body), str(rebuilt), cq.exporters.ExportTypes.STEP)
    mode = "strict" if info["built_strict"] else f"lenient, skipped {info['skipped']} feature(s), repaired history saved"
    print(f"candidate {keep[best]} of {len(cands)} unique ({mode}) | features {len(history.features)} | "
          f"IoU vs input {mc_iou(body, shape):.4f} | wrote {rebuilt}")


if __name__ == "__main__":
    main()
