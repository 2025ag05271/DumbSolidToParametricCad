"""Train the Solid2Param encoder–decoder.

Usage (from the repository root):
    .venv\\Scripts\\python.exe -m s2pnet.train --data data/synth_v2 --out runs/base --epochs 60

Resumes automatically from <out>/last.pt and logs to <out>/log.jsonl. Every ``--gen-every`` epochs:
greedy generation on ``--gen-samples`` val parts (validity, feature-type match, normalized token
edit distance) and a CAD-level check on the first ``--cad-samples`` of them (rebuild + IoU against
the input, in ``--cad-workers`` processes). ``best.pt`` is the checkpoint with the highest CAD score
(mean IoU, 0 for parts that do not rebuild), because validation loss is a poor proxy: it starts
rising long before generation quality stops improving. With ``--cad-samples 0`` best.pt falls back
to the lowest validation loss.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cadseq.dataset import CADGraphSeqDataset, LengthBucketSampler, collate, to_device
from cadseq.prepare import sample_path
from cadseq.tokenizer import EOS_ID, DecodeError, decode
from cadseq.workers import SupervisedPool
from cadseq.dataset import CopySeqDataset, collate_copy

from .model import ModelConfig, Solid2Param
from .verify import cad_check


def _reseed_worker(worker_id: int) -> None:
    """Distinct pose-augmentation random streams per DataLoader worker."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def keep_awake() -> None:
    """Stop Windows idle sleep / Modern Standby (which suspends training) while this process runs.
    Closing the lid or a manual sleep still suspends it."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def lr_at(step, total, warmup, peak, floor=0.05):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def validate(model, loader, device, amp_dtype):
    model.eval()
    if model.cfg.copy:  # mixture (copy / generate) likelihood: average the training loss + metrics
        tot, n, agg = 0.0, 0, {}
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss, m = model.loss(batch)
            tot, n = tot + loss.item(), n + 1
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        return {"val_loss": tot / n, **{f"val_{k}": v / n for k, v in agg.items()}}
    tot, n_tok, correct, seq_ok, n_seq = 0.0, 0, 0, 0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(batch)
        target = batch["tokens"][:, 1:]
        valid = target != 0
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0, reduction="sum")
        hit = (logits.argmax(-1) == target) & valid
        tot += loss.item()
        n_tok += int(valid.sum())
        correct += int(hit.sum())
        seq_ok += int((hit | ~valid).all(1).sum())
        n_seq += target.shape[0]
    return {"val_loss": tot / n_tok, "val_tok_acc": correct / n_tok, "val_tf_seq_acc": seq_ok / n_seq}


def token_edit_distance(a, b) -> int:
    """Levenshtein distance between two token sequences (row-vectorized DP)."""
    a, b = np.asarray(a), np.asarray(b)
    prev = np.arange(len(b) + 1)
    idx = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        tmp = np.empty_like(prev)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[:-1] + (b != x), prev[1:] + 1)  # substitution / deletion
        prev = np.minimum.accumulate(tmp - idx) + idx  # insertions (prefix closure)
    return int(prev[-1])


@torch.no_grad()
def generation_eval(model, ds, device, amp_dtype, n: int, batch_size: int = 32, coll=collate):
    """Grammar-constrained greedy decoding on the first n samples (no CAD kernel needed).

    Returns (metrics, predicted token sequences). ``gen_tok_err`` is the token edit distance
    divided by the ground-truth length (0 = identical); unlike exact match it moves as the model
    improves.
    """
    model.eval()
    n = min(n, len(ds))
    exact = valid = feat_ok = 0
    errs, seqs = [], []
    copied = numbers = 0
    t0 = time.time()
    for s in range(0, n, batch_size):
        batch = to_device(coll([ds[i] for i in range(s, min(n, s + batch_size))]), device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            outs = model.generate(batch, max_len=model.cfg.max_len)
        st = getattr(model.decoder, "copy_stats", None)
        if st:
            copied, numbers = copied + st["copied"], numbers + st["numbers"]
        for seq, gt, L in zip(outs, batch["tokens"].tolist(), batch["lengths"].tolist()):
            gt = gt[:L]
            seqs.append(seq)
            exact += seq == gt
            errs.append(token_edit_distance(seq, gt) / len(gt))
            if seq[-1] == EOS_ID:
                try:
                    pred = decode(seq)
                    valid += 1
                    feat_ok += [f.type for f in pred.features] == [f.type for f in decode(gt).features]
                except DecodeError:
                    pass
    metrics = {"gen_n": n, "gen_exact": exact / n, "gen_valid": valid / n,
               "gen_feature_types_match": feat_ok / n, "gen_tok_err": float(np.mean(errs)),
               "gen_seconds": round(time.time() - t0, 1)}
    if numbers:
        metrics["gen_copy_rate"] = copied / numbers
    return metrics, seqs


class CadEvaluator:
    """Rebuilds predictions in a persistent, supervised pool of CAD worker processes, scores by IoU."""

    def __init__(self, workers: int, timeout: float = 120.0):
        self.pool = SupervisedPool(cad_check, workers, timeout)

    def __call__(self, seqs, step_paths) -> dict:
        t0 = time.time()
        tasks, slots = [], []
        for k, (seq, path) in enumerate(zip(seqs, step_paths)):
            try:
                tasks.append(([decode(seq).model_dump_json()], str(path), 0, False))
                slots.append(k)
            except DecodeError:
                pass
        res = [{"built": False, "built_strict": False, "iou": 0.0} for _ in seqs]
        for j, status, r in self.pool.run(tasks):
            if status == "ok":
                res[slots[j]] = r
        ious = sorted(r["iou"] for r in res)
        n = len(res)
        return {"cad_n": n, "cad_built_strict": sum(r["built_strict"] for r in res) / n,
                "cad_built": sum(r["built"] for r in res) / n, "cad_iou_mean": float(np.mean(ious)),
                "cad_iou_median": ious[n // 2], "cad_seconds": round(time.time() - t0, 1)}

    def close(self):
        self.pool.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/synth_v2"))
    ap.add_argument("--out", type=Path, default=Path("runs/base"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--max-seq", type=int, default=768, help="drop training samples with more tokens")
    ap.add_argument("--gen-every", type=int, default=5, help="epochs between generation + CAD evals")
    ap.add_argument("--gen-samples", type=int, default=128)
    ap.add_argument("--cad-samples", type=int, default=64,
                    help="val parts (subset of gen samples) rebuilt for the CAD score; 0 = select best.pt by val loss")
    ap.add_argument("--cad-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="use only N train samples (debug/overfit)")
    ap.add_argument("--val-split", default="val", help="split used for validation (e.g. 'train' to check overfitting)")
    ap.add_argument("--seed", type=int, default=0)
    for f, v in asdict(ModelConfig()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{f.replace('_', '-')}", action="store_true", default=None)
        else:
            ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=None)
    ap.add_argument("--pose-aug", type=float, default=0.0,
                    help="v2: probability of a random axis-aligned pose + translation per training item")
    ap.add_argument("--loader-workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.out / "last.pt"
    resume = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else None
    if resume:
        cfg = ModelConfig(**resume["model"]["config"])
    else:
        cfg = ModelConfig(**{f: (getattr(args, f) if getattr(args, f) is not None else v)
                             for f, v in asdict(ModelConfig()).items()})

    if cfg.copy:
        train_ds = CopySeqDataset(args.data, "train", pose_aug=args.pose_aug, max_len=args.max_seq,
                                  limit=args.limit, seed=args.seed)
        val_ds = CopySeqDataset(args.data, args.val_split, max_len=args.max_seq,
                                limit=args.limit if args.val_split == "train" else None)
        coll = collate_copy
    else:
        train_ds = CADGraphSeqDataset(args.data, "train", max_len=args.max_seq, limit=args.limit)
        val_ds = CADGraphSeqDataset(args.data, args.val_split, max_len=args.max_seq,
                                    limit=args.limit if args.val_split == "train" else None)
        coll = collate
    sampler = LengthBucketSampler(train_ds, args.bs, shuffle=True, seed=args.seed)
    loader_kw = {}
    if args.loader_workers > 0:
        loader_kw = {"num_workers": args.loader_workers, "persistent_workers": True,
                     "worker_init_fn": _reseed_worker}
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=coll, pin_memory=True, **loader_kw)
    # Copy scoring over thousands of candidates makes eval as memory-hungry as training: don't double the batch.
    val_bs = args.bs if cfg.copy else args.bs * 2
    val_dl = DataLoader(val_ds, batch_sampler=LengthBucketSampler(val_ds, val_bs, shuffle=False),
                        collate_fn=coll, pin_memory=True)

    model = Solid2Param(cfg).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos" not in n and "tok" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "pos" not in n and "tok" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
    start_epoch, step = 0, 0
    use_cad = args.cad_samples > 0
    # best_score is "higher is better": CAD IoU score, or -val_loss when CAD evaluation is off.
    best_score = -float("inf")
    if resume:
        model.load_state_dict(resume["model"]["state_dict"])
        opt.load_state_dict(resume["opt"])
        start_epoch, step = resume["epoch"] + 1, resume["step"]
        best_score = resume.get("best_score", -float("inf"))  # older checkpoints stored val loss
        sampler.epoch = start_epoch
        print(f"resumed from epoch {resume['epoch']} (step {step})")
    cad_eval = CadEvaluator(args.cad_workers) if use_cad else None
    cad_n = min(args.cad_samples, args.gen_samples, len(val_ds))
    cad_paths = [sample_path(args.data, "step", sid) for sid, _ in val_ds.items[:cad_n]]

    n_params = sum(p.numel() for p in model.parameters())
    total_steps = args.epochs * len(sampler)
    print(f"device {device} | params {n_params / 1e6:.1f}M | train {len(train_ds)} val {len(val_ds)} | "
          f"{len(sampler)} steps/epoch, {total_steps} total", flush=True)
    (args.out / "config.json").write_text(json.dumps({"model": asdict(cfg), "args": {
        k: str(v) for k, v in vars(args).items()}}, indent=2), encoding="utf-8")

    log = open(args.out / "log.jsonl", "a", encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, nb, ntok = time.time(), 0.0, 0, 0
        run_m: dict = {}
        for batch in train_dl:
            batch = to_device(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.warmup, args.lr)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss, m = model.loss(batch, args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            nb += 1
            ntok += int(batch["lengths"].sum())
            run_loss += loss.item()
            for k, v in m.items():
                run_m[k] = run_m.get(k, 0.0) + v
            if step % 100 == 0:
                extra = " ".join(f"{k} {v / nb:.4f}" for k, v in run_m.items() if k != "seq_acc")
                print(f"  ep {epoch} step {step} loss {run_loss / nb:.4f} {extra} "
                      f"gnorm {gnorm:.2f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{ntok / (time.time() - t0):.0f} tok/s", flush=True)

        rec = {"epoch": epoch, "step": step, "train_loss": run_loss / nb,
               **{f"train_{k}": v / nb for k, v in run_m.items()},
               "epoch_seconds": round(time.time() - t0, 1)}
        rec.update(validate(model, val_dl, device, amp_dtype))
        score = None if use_cad else -rec["val_loss"]
        if (epoch + 1) % args.gen_every == 0 or epoch + 1 == args.epochs:
            gen_metrics, seqs = generation_eval(model, val_ds, device, amp_dtype, args.gen_samples, coll=coll)
            rec.update(gen_metrics)
            if use_cad:
                rec.update(cad_eval(seqs[:cad_n], cad_paths))
                score = rec["cad_iou_mean"]
        if device.type == "cuda":
            rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

        is_best = score is not None and score > best_score
        if is_best:
            best_score = score
        rec["is_best"] = is_best
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

        state = {"model": model.checkpoint(), "opt": opt.state_dict(), "epoch": epoch, "step": step,
                 "best_score": best_score}
        torch.save(state, args.out / "last.pt.tmp")
        Path(args.out / "last.pt.tmp").replace(ckpt_path)
        if is_best:
            torch.save({"model": model.checkpoint(), "epoch": epoch, "score": best_score,
                        "score_metric": "cad_iou_mean" if use_cad else "-val_loss"}, args.out / "best.pt")
    if cad_eval is not None:
        cad_eval.close()


if __name__ == "__main__":
    main()
