"""Train the Solid2Param encoder–decoder.

Usage (from solid2param/, with the training venv):
    .venv\\Scripts\\python.exe -m s2pnet.train --data data/synth_v2 --out runs/base --epochs 60

Resumes automatically from <out>/last.pt. Logs to <out>/log.jsonl; keeps best.pt (lowest val loss).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cadseq.dataset import CADGraphSeqDataset, LengthBucketSampler, collate, to_device
from cadseq.tokenizer import EOS_ID, DecodeError, decode

from .model import ModelConfig, Solid2Param


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


@torch.no_grad()
def generation_eval(model, ds, device, amp_dtype, n: int, batch_size: int = 32):
    """Grammar-constrained greedy decoding on the first n samples (no CAD kernel needed)."""
    model.eval()
    n = min(n, len(ds))
    exact = valid = feat_ok = 0
    t0 = time.time()
    for s in range(0, n, batch_size):
        batch = to_device(collate([ds[i] for i in range(s, min(n, s + batch_size))]), device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            outs = model.generate(batch, max_len=model.cfg.max_len)
        for seq, gt, L in zip(outs, batch["tokens"].tolist(), batch["lengths"].tolist()):
            gt = gt[:L]
            exact += seq == gt
            if seq[-1] == EOS_ID:
                try:
                    pred = decode(seq)
                    valid += 1
                    feat_ok += [f.type for f in pred.features] == [f.type for f in decode(gt).features]
                except DecodeError:
                    pass
    return {"gen_n": n, "gen_exact": exact / n, "gen_valid": valid / n, "gen_feature_types_match": feat_ok / n,
            "gen_seconds": round(time.time() - t0, 1)}


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
    ap.add_argument("--gen-every", type=int, default=5, help="epochs between generation evals")
    ap.add_argument("--gen-samples", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="use only N train samples (debug/overfit)")
    ap.add_argument("--val-split", default="val", help="split used for validation (e.g. 'train' to check overfitting)")
    ap.add_argument("--seed", type=int, default=0)
    for f, v in asdict(ModelConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = CADGraphSeqDataset(args.data, "train", max_len=args.max_seq, limit=args.limit)
    val_ds = CADGraphSeqDataset(args.data, args.val_split, max_len=args.max_seq,
                                limit=args.limit if args.val_split == "train" else None)
    sampler = LengthBucketSampler(train_ds, args.bs, shuffle=True, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_sampler=LengthBucketSampler(val_ds, args.bs * 2, shuffle=False),
                        collate_fn=collate, pin_memory=True)

    ckpt_path = args.out / "last.pt"
    resume = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path.exists() else None
    if resume:
        cfg = ModelConfig(**resume["model"]["config"])
    else:
        cfg = ModelConfig(**{f: (getattr(args, f) if getattr(args, f) is not None else v)
                             for f, v in asdict(ModelConfig()).items()})
    model = Solid2Param(cfg).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos" not in n and "tok" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "pos" not in n and "tok" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
    start_epoch, step, best = 0, 0, float("inf")
    if resume:
        model.load_state_dict(resume["model"]["state_dict"])
        opt.load_state_dict(resume["opt"])
        start_epoch, step, best = resume["epoch"] + 1, resume["step"], resume["best"]
        sampler.epoch = start_epoch
        print(f"resumed from epoch {resume['epoch']} (step {step})")

    n_params = sum(p.numel() for p in model.parameters())
    total_steps = args.epochs * len(sampler)
    print(f"device {device} | params {n_params / 1e6:.1f}M | train {len(train_ds)} val {len(val_ds)} | "
          f"{len(sampler)} steps/epoch, {total_steps} total", flush=True)
    (args.out / "config.json").write_text(json.dumps({"model": asdict(cfg), "args": {
        k: str(v) for k, v in vars(args).items()}}, indent=2), encoding="utf-8")

    log = open(args.out / "log.jsonl", "a", encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, run_acc, nb, ntok = time.time(), 0.0, 0.0, 0, 0
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
            run_acc += m["tok_acc"]
            if step % 100 == 0:
                print(f"  ep {epoch} step {step} loss {run_loss / nb:.4f} tok_acc {run_acc / nb:.4f} "
                      f"gnorm {gnorm:.2f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{ntok / (time.time() - t0):.0f} tok/s", flush=True)

        rec = {"epoch": epoch, "step": step, "train_loss": run_loss / nb, "train_tok_acc": run_acc / nb,
               "epoch_seconds": round(time.time() - t0, 1)}
        rec.update(validate(model, val_dl, device, amp_dtype))
        if (epoch + 1) % args.gen_every == 0 or epoch + 1 == args.epochs:
            rec.update(generation_eval(model, val_ds, device, amp_dtype, args.gen_samples))
        if device.type == "cuda":
            rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

        is_best = rec["val_loss"] < best
        best = min(best, rec["val_loss"])
        state = {"model": model.checkpoint(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "best": best}
        torch.save(state, args.out / "last.pt.tmp")
        Path(args.out / "last.pt.tmp").replace(ckpt_path)
        if is_best:
            torch.save({"model": model.checkpoint(), "epoch": epoch, "val_loss": best}, args.out / "best.pt")


if __name__ == "__main__":
    main()
