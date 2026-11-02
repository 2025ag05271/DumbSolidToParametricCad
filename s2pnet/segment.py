"""Feature identification: per-face feature class + feature instances (UV-Net / AAGNet style).

    .venv\\Scripts\\python.exe -m s2pnet.segment train --data data/synth_v2 --out runs/seg
    .venv\\Scripts\\python.exe -m s2pnet.segment eval  --ckpt runs/seg/best.pt --split test

Model: ``BRepEncoder`` (UV-grid CNNs + exact face / edge attributes + message passing + face
Transformer) with two per-face heads:
- semantic head → one of ``cadseq.provenance.FACE_CLASSES`` (extrude side/end, cut side/end,
  simple / counterbore / countersink hole, fillet, chamfer);
- instance head → embedding; two faces belong to the same feature instance when the sigmoid of
  their scaled dot product > 0.5 (pairwise BCE within each part, balanced positives/negatives).

Training uses random rotation / translation / scale augmentation, so identification does not
depend on how a STEP file is placed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cadseq.dataset import CADGraphSeqDataset, LengthBucketSampler, collate, to_device
from cadseq.provenance import FACE_CLASSES

from .encoder import BRepEncoder, _mlp


@dataclass
class SegConfig:
    d_model: int = 256
    heads: int = 8
    graph_layers: int = 4
    enc_layers: int = 2
    dropout: float = 0.1
    emb_dim: int = 64
    use_attrs: bool = True


class FaceSegmenter(nn.Module):
    def __init__(self, cfg: SegConfig = SegConfig()):
        super().__init__()
        from step_to_graph import EDGE_ATTR_DIM, FACE_ATTR_DIM

        self.cfg = cfg
        self.encoder = BRepEncoder(cfg.d_model, cfg.graph_layers, cfg.enc_layers, cfg.heads, cfg.dropout,
                                   FACE_ATTR_DIM if cfg.use_attrs else 0, EDGE_ATTR_DIM if cfg.use_attrs else 0)
        self.cls_head = _mlp(cfg.d_model, cfg.d_model, len(FACE_CLASSES), cfg.dropout)
        self.emb_head = _mlp(cfg.d_model, cfg.d_model, cfg.emb_dim)
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, batch):
        _, _, h = self.encoder(batch["node_x"], batch["edge_x"], batch["edge_index"], batch["node_batch"],
                               int(batch["node_batch"].max()) + 1,
                               batch.get("face_attr"), batch.get("edge_attr"), return_nodes=True)
        return self.cls_head(h), self.emb_head(h)

    def affinity_logits(self, emb_a, emb_b):
        return emb_a @ emb_b.T / math.sqrt(self.cfg.emb_dim) + self.bias

    def loss(self, batch, cls_weight=None):
        logits, emb = self(batch)
        target = batch["face_cls"]
        l_cls = F.cross_entropy(logits.float(), target, weight=cls_weight)
        l_inst, n = 0.0, 0
        for b in range(int(batch["node_batch"].max()) + 1):
            m = batch["node_batch"] == b
            e = emb[m].float()
            inst = batch["face_instance"][m]
            same = (inst[:, None] == inst[None, :]).float()
            bce = F.binary_cross_entropy_with_logits(self.affinity_logits(e, e), same, reduction="none")
            pos, neg = same.sum(), (1 - same).sum()
            l_b = (bce * same).sum() / pos.clamp_min(1) + (bce * (1 - same)).sum() / neg.clamp_min(1)
            l_inst = l_inst + 0.5 * l_b
            n += 1
        l_inst = l_inst / max(n, 1)
        acc = (logits.argmax(-1) == target).float().mean().item()
        return l_cls + l_inst, {"cls": l_cls.item(), "inst": float(l_inst), "acc": acc}

    def checkpoint(self) -> dict:
        return {"config": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "FaceSegmenter":
        m = cls(SegConfig(**ckpt["config"]))
        m.load_state_dict(ckpt["state_dict"])
        return m


# ------------------------------------------------------------- inference --

# Instances group faces by feature *type*: a pad's side and end faces are one extrude instance.
FEATURE_TYPES = ["extrude", "cut", "hole_simple", "hole_counterbore", "hole_countersink", "fillet", "chamfer"]
CLASS_TO_TYPE = np.array([FEATURE_TYPES.index(c.rsplit("_", 1)[0] if c.startswith(("extrude", "cut")) else c)
                          for c in FACE_CLASSES])


def connected_regions(inst: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Split instances into connected face regions (faces of one instance linked by adjacency).

    Instance identification is evaluated on regions: one history feature may fillet several separate
    edges, and how a history groups equal-radius fillets into features is arbitrary."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(inst)
    same = inst[edge_index[0]] == inst[edge_index[1]]
    a, b = edge_index[0][same], edge_index[1][same]
    g = csr_matrix((np.ones(len(a), dtype=bool), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def group_instances(cls_pred: np.ndarray, aff: np.ndarray, edge_index: np.ndarray,
                    threshold: float = 0.5) -> np.ndarray:
    """Faces → instance (region) ids: connected components of adjacent faces with the same feature
    type and affinity > threshold. Keeps identical neighbouring features (e.g. a hole pattern) apart."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(cls_pred)
    ftype = CLASS_TO_TYPE[cls_pred]
    a, b = edge_index
    keep = (ftype[a] == ftype[b]) & (aff[a, b] > threshold)
    g = csr_matrix((np.ones(int(keep.sum()), dtype=bool), (a[keep], b[keep])), shape=(n, n))
    return connected_components(g, directed=False)[1]


@torch.no_grad()
def predict(model: FaceSegmenter, batch, threshold: float = 0.5) -> list[dict]:
    """Per part: face class ids, class probabilities, instance ids and the affinity matrix."""
    model.eval()
    logits, emb = model(batch)
    probs = logits.float().softmax(-1)
    out = []
    nb = batch["node_batch"]
    ei = batch["edge_index"].cpu().numpy()
    starts = torch.cumsum(torch.bincount(nb), 0).cpu().numpy()
    for b in range(int(nb.max()) + 1):
        m = nb == b
        lo, hi = (0 if b == 0 else starts[b - 1]), starts[b]
        local = ei[:, (ei[0] >= lo) & (ei[0] < hi)] - lo
        e = emb[m].float()
        aff = torch.sigmoid(model.affinity_logits(e, e).float()).cpu().numpy()
        cls = probs[m].argmax(-1).cpu().numpy()
        out.append({"cls": cls, "prob": probs[m].cpu().numpy(), "aff": aff,
                    "instance": group_instances(cls, aff, local, threshold)})
    return out


# -------------------------------------------------------------- metrics --

def instance_f1(gt_inst, gt_type, pr_inst, pr_type, iou_thr: float = 0.5):
    """Feature-level matching (AAGNet style): a predicted instance is a true positive if it has
    the same feature type as an unmatched ground-truth instance and face IoU ≥ iou_thr."""
    gts = [(set(np.nonzero(gt_inst == g)[0]), np.bincount(gt_type[gt_inst == g]).argmax())
           for g in np.unique(gt_inst)]
    prs = [(set(np.nonzero(pr_inst == p)[0]), np.bincount(pr_type[pr_inst == p]).argmax())
           for p in np.unique(pr_inst)]
    used, tp = set(), 0
    for faces, c in prs:
        for k, (gf, gc) in enumerate(gts):
            if k in used or gc != c:
                continue
            if len(faces & gf) / len(faces | gf) >= iou_thr:
                used.add(k)
                tp += 1
                break
    return tp, len(prs), len(gts)


@torch.no_grad()
def evaluate(model, ds, device, bs: int = 64, limit: int | None = None, threshold: float = 0.5) -> dict:
    model.eval()
    n_cls, n_typ = len(FACE_CLASSES), len(FEATURE_TYPES)
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    tp = n_pred = n_gt = 0
    per_type = np.zeros((n_typ, 3), dtype=np.int64)  # tp, pred, gt instances per feature type
    n = min(len(ds), limit or len(ds))
    for s in range(0, n, bs):
        items = [ds[i] for i in range(s, min(n, s + bs))]
        batch = to_device(collate(items), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            preds = predict(model, batch, threshold)
        for it, p in zip(items, preds):
            gt_c = it["face_cls"].numpy()
            gt_i = connected_regions(it["face_instance"].numpy(), it["edge_index"].numpy())
            gt_t, pr_t = CLASS_TO_TYPE[gt_c], CLASS_TO_TYPE[p["cls"]]
            np.add.at(conf, (gt_c, p["cls"]), 1)
            t, npr, ngt = instance_f1(gt_i, gt_t, p["instance"], pr_t)
            tp, n_pred, n_gt = tp + t, n_pred + npr, n_gt + ngt
            for c in range(n_typ):
                gi = np.unique(gt_i[gt_t == c])
                if len(gi) == 0:
                    continue
                sel = np.isin(gt_i, gi)  # faces of this type's GT instances
                per_type[c] += instance_f1(gt_i[sel], gt_t[sel], p["instance"][sel], pr_t[sel])
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    prec, rec = tp / max(n_pred, 1), tp / max(n_gt, 1)
    return {
        "parts": n, "face_acc": float(inter.sum() / conf.sum()), "mIoU": float(iou.mean()),
        "class_IoU": {c: round(float(v), 4) for c, v in zip(FACE_CLASSES, iou)},
        "instance_precision": prec, "instance_recall": rec,
        "instance_F1": 2 * prec * rec / max(prec + rec, 1e-9),
        "type_instance_recall": {c: round(float(t / max(g, 1)), 4) for c, (t, _, g) in zip(FEATURE_TYPES, per_type)},
        "confusion": conf.tolist(),
    }


# ------------------------------------------------------------- training --

def train(args) -> None:
    from .train import keep_awake, lr_at

    keep_awake()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    train_ds = CADGraphSeqDataset(args.data, "train", augment=not args.no_augment, seed=args.seed, limit=args.limit)
    val_ds = CADGraphSeqDataset(args.data, "val", augment=not args.no_augment, seed=args.seed + 1)
    if not train_ds.has_labels:
        raise SystemExit(f"no face labels in {args.data}: run `python -m cadseq.labels {args.data}` first")
    cfg = SegConfig(d_model=args.d_model, enc_layers=args.enc_layers, use_attrs=not args.no_attrs)
    model = FaceSegmenter(cfg).to(device)
    sampler = LengthBucketSampler(train_ds, args.bs, shuffle=True, seed=args.seed)
    dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total, step, best = args.epochs * len(sampler), 0, -1.0

    # class weights: inverse sqrt frequency (fillets / chamfers / cut ends are rarer)
    counts = np.bincount(np.asarray(train_ds.extra["face_cls"]), minlength=len(FACE_CLASSES)).astype(float)
    w = 1 / np.sqrt(np.maximum(counts, 1))
    cls_weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | train {len(train_ds)} "
          f"val {len(val_ds)} | attrs {cfg.use_attrs} | augment {not args.no_augment}", flush=True)
    (args.out / "config.json").write_text(json.dumps({"model": asdict(cfg), "args": {
        k: str(v) for k, v in vars(args).items()}}, indent=2), encoding="utf-8")
    log = open(args.out / "log.jsonl", "a", encoding="utf-8")
    for epoch in range(args.epochs):
        model.train()
        t0, agg, nb = time.time(), {"cls": 0.0, "inst": 0.0, "acc": 0.0}, 0
        for batch in dl:
            batch = to_device(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.warmup, args.lr)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, m = model.loss(batch, cls_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step, nb = step + 1, nb + 1
            for k in agg:
                agg[k] += m[k]
        rec = {"epoch": epoch, **{f"train_{k}": v / nb for k, v in agg.items()},
               "epoch_seconds": round(time.time() - t0, 1)}
        ev = evaluate(model, val_ds, device, limit=args.val_parts)
        rec.update({k: ev[k] for k in ("face_acc", "mIoU", "instance_F1")})
        score = ev["mIoU"] + ev["instance_F1"]
        rec["is_best"] = score > best
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save({"model": model.checkpoint(), "epoch": epoch}, args.out / "last.pt")
        if score > best:
            best = score
            torch.save({"model": model.checkpoint(), "epoch": epoch, "val": ev}, args.out / "best.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=Path("data/synth_v2"))
    t.add_argument("--out", type=Path, default=Path("runs/seg"))
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--wd", type=float, default=0.05)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--enc-layers", type=int, default=2)
    t.add_argument("--val-parts", type=int, default=500)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--no-attrs", action="store_true", help="ablation: UV-grids only")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--data", type=Path, default=Path("data/synth_v2"))
    e.add_argument("--split", default="test")
    e.add_argument("--no-augment", action="store_true", help="evaluate in the generated placement only")
    args = ap.parse_args()

    if args.cmd == "train":
        train(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceSegmenter.from_checkpoint(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    model.to(device)
    ds = CADGraphSeqDataset(args.data, args.split, augment=not args.no_augment, seed=123)
    res = evaluate(model, ds, device)
    tag = "fixedpose" if args.no_augment else "randompose"
    (args.ckpt.parent / f"eval_{args.split}_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()
