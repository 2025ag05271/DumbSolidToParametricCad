"""CAD-level evaluation: predict histories for a split, rebuild them and compare with the input solid.

Usage:
    .venv\\Scripts\\python.exe -m s2pnet.evaluate --ckpt runs/base/best.pt --split test --n 500

Metrics per sample (written to <ckpt dir>/eval_<split>.jsonl, summary printed and saved):
    exact      predicted tokens == ground-truth tokens
    valid      tokens decode to a valid CADSequence
    built_strict  the sequence rebuilds exactly as predicted (edge refs within 0.3 mm)
    built      rebuilds with ``build_lenient`` (edge refs snapped, failing features skipped)
    iou        Monte Carlo volume IoU of the (lenient) rebuild vs the ground-truth STEP (0 if not built)
    iou_strict IoU counting only strict rebuilds
    chamfer    Chamfer distance / GT bbox diagonal (built samples only)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import torch

from cadseq.dataset import CADGraphSeqDataset, collate, to_device
from cadseq.prepare import load_index, sample_path
from cadseq.tokenizer import DecodeError, decode

from .model import Solid2Param


def _cad_check(args):
    """Runs in a worker process: rebuild prediction and compare it with the ground-truth STEP."""
    import warnings

    warnings.filterwarnings("ignore")
    import cadquery as cq

    from cadseq.build import QUANTIZED_EDGE_TOL, build, build_lenient
    from cadseq.metrics import chamfer, mc_iou
    from cadseq.schema import CADSequence

    seq_json, gt_step = args
    seq = CADSequence.model_validate_json(seq_json)
    res = {"built_strict": False, "built": False, "iou": 0.0, "iou_strict": 0.0, "skipped": 0}
    try:
        pred = build(seq, edge_tol=QUANTIZED_EDGE_TOL)
        res["built_strict"] = True
    except Exception as e:
        res["error"] = str(e)[:200]
        try:
            pred, skipped = build_lenient(seq)
            res["skipped"] = len(skipped)
        except Exception as e2:
            res["error_lenient"] = str(e2)[:200]
            return res
    res["built"] = True
    try:
        gt = cq.importers.importStep(gt_step).val()
        res.update({"iou": mc_iou(pred, gt), "chamfer": chamfer(pred, gt),
                    "faces": len(pred.Faces()), "gt_faces": len(gt.Faces())})
        if res["built_strict"]:
            res["iou_strict"] = res["iou"]
    except Exception as e:
        res["error_metric"] = str(e)[:200]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/synth_v2"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--unconstrained", action="store_true", help="disable grammar-constrained decoding")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Solid2Param.from_checkpoint(ckpt["model"]).to(device).eval()
    ds = CADGraphSeqDataset(args.data, args.split, limit=args.n)
    index = load_index(args.data)

    t0 = time.time()
    preds = []
    for s in range(0, len(ds), args.bs):
        batch = to_device(collate([ds[i] for i in range(s, min(len(ds), s + args.bs))]), device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outs = model.generate(batch, constrained=not args.unconstrained)
        for sid, seq, gt, L in zip(batch["ids"], outs, batch["tokens"].tolist(), batch["lengths"].tolist()):
            preds.append((sid, seq, gt[:L]))
        print(f"  generated {len(preds)}/{len(ds)}", flush=True)
    gen_s = time.time() - t0

    rows, jobs = [], []
    for sid, seq, gt in preds:
        r = {"id": sid, "family": index[sid]["family"], "exact": seq == gt, "n_pred_tokens": len(seq),
             "n_gt_tokens": len(gt), "valid": False, "built_strict": False, "built": False,
             "iou": 0.0, "iou_strict": 0.0, "skipped": 0}
        try:
            job = (decode(seq).model_dump_json(), str(sample_path(args.data, "step", sid)))
            r["valid"] = True
            jobs.append((len(rows), job))
        except DecodeError as e:
            r["error"] = str(e)[:200]
        rows.append(r)

    t1 = time.time()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, maxtasksperchild=50)
    pending = [(i, pool.apply_async(_cad_check, (job,))) for i, job in jobs]
    for k, (i, res) in enumerate(pending, 1):
        try:
            rows[i].update(res.get(timeout=args.timeout))
        except mp.TimeoutError:
            rows[i].update({"error": "timeout"})
        if k % 50 == 0:
            print(f"  rebuilt {k}/{len(pending)}", flush=True)
    pool.terminate()
    cad_s = time.time() - t1

    out = args.ckpt.parent / f"eval_{args.split}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def agg(rs):
        n = len(rs)
        ious = sorted(r["iou"] for r in rs)
        cds = sorted(r["chamfer"] for r in rs if "chamfer" in r)
        built = [r for r in rs if r["built"]]
        return {
            "n": n,
            "exact": round(sum(r["exact"] for r in rs) / n, 4),
            "valid": round(sum(r["valid"] for r in rs) / n, 4),
            "built_strict": round(sum(r["built_strict"] for r in rs) / n, 4),
            "built": round(len(built) / n, 4),
            "iou_strict_mean": round(sum(r["iou_strict"] for r in rs) / n, 4),
            "iou_mean": round(sum(ious) / n, 4),
            "iou_median": round(ious[n // 2], 4),
            "iou>0.9": round(sum(i > 0.9 for i in ious) / n, 4),
            "iou>0.95": round(sum(i > 0.95 for i in ious) / n, 4),
            "iou>0.99": round(sum(i > 0.99 for i in ious) / n, 4),
            "chamfer_median_built": round(cds[len(cds) // 2], 5) if cds else None,
            "mean_skipped_features": round(sum(r["skipped"] for r in built) / max(1, len(built)), 2),
        }

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    summary = {"ckpt": str(args.ckpt), "split": args.split, "constrained": not args.unconstrained,
               **agg(rows), "generate_seconds": round(gen_s, 1), "cad_seconds": round(cad_s, 1),
               "by_family": {f: agg(rs) for f, rs in sorted(by_fam.items())}}
    (args.ckpt.parent / f"eval_{args.split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
