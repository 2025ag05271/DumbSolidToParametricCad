"""CAD-level evaluation: predict histories for a split, rebuild them and compare with the input solid.

Usage:
    .venv\\Scripts\\python.exe -m s2pnet.evaluate --ckpt runs/base/last.pt --split test --n 1000
    .venv\\Scripts\\python.exe -m s2pnet.evaluate --ckpt runs/base/last.pt --samples 8 --temperature 0.8

With ``--samples N`` (generate-and-verify) each part gets the greedy prediction plus N-1 sampled
ones; every candidate is rebuilt and the one with the best IoU against the **input** solid is kept
(selection uses its own random points, seed 1; the final IoU below uses seed 0).

Metrics per sample (written to <ckpt dir>/eval_<split>[_s<N>].jsonl, summary printed and saved):
    exact      chosen tokens == ground-truth tokens
    valid      at least one candidate decodes to a valid CADSequence
    built_strict  the chosen sequence rebuilds exactly as predicted (edge refs within 0.3 mm)
    built      rebuilds with ``build_lenient`` (edge refs snapped, failing features skipped)
    iou        Monte Carlo volume IoU of the chosen rebuild vs the ground-truth STEP (0 if not built)
    iou_strict IoU counting only strict rebuilds
    chamfer    Chamfer distance / GT bbox diagonal (built samples only)
    chosen     index of the chosen candidate (0 = greedy)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cadseq.dataset import CADGraphSeqDataset, CopySeqDataset, collate, collate_copy, to_device
from cadseq.pose import transform_sequence
from cadseq.prepare import load_index, sample_path
from cadseq.workers import SupervisedPool

from .model import Solid2Param
from .train import keep_awake
from .verify import cad_check, decode_candidates, generate_candidates


def unpose(keep, seqs, pose):
    """Map histories predicted for a posed input (x -> R x + t) back to the original frame.
    Returns (kept candidate indices, mapped sequences); candidates that leave the range are dropped."""
    R, t = pose
    if np.allclose(R, np.eye(3)) and not np.any(t):
        return keep, seqs
    Ri = R.T
    ti = -(Ri @ t)
    k_out, s_out = [], []
    for k, s in zip(keep, seqs):
        try:
            s_out.append(transform_sequence(s, Ri, ti))
            k_out.append(k)
        except Exception:
            pass
    return k_out, s_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/synth_v2"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--samples", type=int, default=1, help="candidates per part (1 = greedy only)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--select-points", type=int, default=2048, help="MC points for candidate selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per candidate")
    ap.add_argument("--pose", action="store_true",
                    help="copy models: give each part a fixed random axis-aligned pose + translation; "
                         "predictions are mapped back before comparing with the original STEP")
    ap.add_argument("--tag", default=None, help="output file suffix (default: s<samples>[_t<temp>])")
    args = ap.parse_args()

    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Solid2Param.from_checkpoint(ckpt["model"]).to(device).eval()
    if model.cfg.copy:
        ds = CopySeqDataset(args.data, args.split, limit=args.n, pose_aug=1.0 if args.pose else 0.0, fixed_pose=True)
        coll = collate_copy
    else:
        if args.pose:
            raise SystemExit("--pose needs a copy model (v1 models were trained in the generated placement)")
        ds, coll = CADGraphSeqDataset(args.data, args.split, limit=args.n), collate
    index = load_index(args.data)
    tag = args.tag or (f"s{args.samples}" + (f"_t{args.temperature:g}" if args.samples > 1 else "")
                       + ("_posed" if args.pose else ""))

    t0 = time.time()
    preds = []
    for s in range(0, len(ds), args.bs):
        batch = to_device(coll([ds[i] for i in range(s, min(len(ds), s + args.bs))]), device)
        cands = generate_candidates(model, batch, args.samples, args.temperature, seed=args.seed + s)
        poses = batch.get("poses") or [(np.eye(3), np.zeros(3))] * len(cands)
        for sid, cs, gt, L, pose in zip(batch["ids"], cands, batch["tokens"].tolist(), batch["lengths"].tolist(), poses):
            preds.append((sid, cs, gt[:L], pose))
        print(f"  generated {len(preds)}/{len(ds)}", flush=True)
    gen_s = time.time() - t0

    rows, jobs = [], []
    for sid, cands, gt, pose in preds:
        keep, seqs = decode_candidates(cands)
        keep, seqs = unpose(keep, seqs, pose)
        r = {"id": sid, "family": index[sid]["family"], "exact": False, "n_gt_tokens": len(gt),
             "n_unique": len(cands), "valid": bool(seqs), "built_strict": False, "built": False,
             "iou": 0.0, "iou_strict": 0.0, "skipped": 0, "chosen": None}
        r["_cands"], r["_keep"], r["_gt"] = cands, keep, gt
        if seqs:
            jobs.append((len(rows), ([q.model_dump_json() for q in seqs],
                                     str(sample_path(args.data, "step", sid)), args.select_points, True)))
        else:
            r["error"] = "no candidate decodes"
        rows.append(r)

    t1 = time.time()
    pool = SupervisedPool(cad_check, args.workers, timeout=lambda job: args.timeout * len(job[0]))
    failures = {"crash": 0, "timeout": 0, "error": 0}
    for k, (j, status, res) in enumerate(pool.run([job for _, job in jobs]), 1):
        i = jobs[j][0]
        if status == "ok":
            rows[i].update(res)
        else:
            failures[status] += 1
            rows[i].update({"error": f"{status}: {res}"})
        if k % 50 == 0:
            print(f"  rebuilt {k}/{len(jobs)}", flush=True)
    pool.close()
    cad_s = time.time() - t1

    for r in rows:
        cands, keep, gt = r.pop("_cands"), r.pop("_keep"), r.pop("_gt")
        if r["chosen"] is not None:
            chosen_tokens = cands[keep[r["chosen"]]]
            r["chosen"] = keep[r["chosen"]]  # index among all candidates (0 = greedy)
            r["exact"] = chosen_tokens == gt
            r["n_pred_tokens"] = len(chosen_tokens)

    out = args.ckpt.parent / f"eval_{args.split}_{tag}.jsonl"
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
            "chosen_greedy": round(sum(r["chosen"] == 0 for r in built) / max(1, len(built)), 4),
            "mean_unique_candidates": round(sum(r["n_unique"] for r in rs) / n, 2),
        }

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    summary = {"ckpt": str(args.ckpt), "split": args.split, "samples": args.samples,
               "temperature": args.temperature if args.samples > 1 else None,
               **agg(rows), "worker_failures": failures,
               "generate_seconds": round(gen_s, 1), "cad_seconds": round(cad_s, 1),
               "by_family": {f: agg(rs) for f, rs in sorted(by_fam.items())}}
    (args.ckpt.parent / f"eval_{args.split}_{tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_family"}, indent=2))


if __name__ == "__main__":
    main()
