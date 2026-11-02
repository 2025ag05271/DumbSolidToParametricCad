"""PyTorch dataset: (UV-Net B-Rep graph, design-history tokens) pairs. Pure torch (no DGL/OCC).

Reads the packed arrays written by ``python -m cadseq.prepare <dataset>`` (memory-mapped, so
``num_workers=0`` is fast and Windows worker start-up is avoided).

    from torch.utils.data import DataLoader
    from cadseq.dataset import CADGraphSeqDataset, collate, LengthBucketSampler

    ds = CADGraphSeqDataset("data/synth_v2", split="train")
    dl = DataLoader(ds, batch_sampler=LengthBucketSampler(ds, 32, shuffle=True), collate_fn=collate)
    b = next(iter(dl))
    b["node_x"]      # (ΣF, 10, 10, 7) float32   face UV-grids
    b["edge_x"]      # (ΣE, 10, 6)     float32   edge U-grids (directed edges)
    b["edge_index"]  # (2, ΣE)         int64     global face ids (src, dst)
    b["node_batch"]  # (ΣF,)           int64     sample id per face
    b["tokens"]      # (B, L)          int64     <bos> … <eos>, padded with PAD_ID
    b["lengths"], b["ids"]

Point coordinates are multiplied by ``coord_scale`` (0.01 → 1 unit = 100 mm) but NOT centred:
target tokens are absolute mm in the world frame, so the encoder sees the same frame.
(For a DGLGraph, use ``cadseq.graph.to_dgl`` on the per-sample .npz files instead.)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .tokenizer import PAD_ID

COORD_SCALE = 0.01
LABEL_KEYS = ["face_cls", "face_role", "face_instance", "face_feature", "face_mirror"]


def scale_features(item: dict, coord_scale: float = COORD_SCALE) -> dict:
    """Raw (mm) numpy arrays → normalized model inputs (in place). Positions and lengths are scaled
    by ``coord_scale``; log-area is divided by 10."""
    item["node_x"][..., :3] *= coord_scale
    item["edge_x"][..., :3] *= coord_scale
    if "face_attr" in item:
        from step_to_graph import FACE_ATTR_LENGTHS

        item["face_attr"][:, FACE_ATTR_LENGTHS] *= coord_scale
        item["face_attr"][:, 16] /= 10.0
    if "edge_attr" in item:
        from step_to_graph import EDGE_ATTR_LENGTHS

        item["edge_attr"][:, EDGE_ATTR_LENGTHS] *= coord_scale
    return item


class CADGraphSeqDataset(Dataset):
    """Graph + token items. If ``cadseq.labels`` has been run on the dataset, items also carry the
    analytic attributes (``face_attr``, ``edge_attr``) and per-face labels (``LABEL_KEYS``).
    ``augment=True`` applies a random rotation / translation / scale to the graph features (only
    valid for tasks whose targets are pose-invariant, e.g. face segmentation — not tokens)."""

    def __init__(self, root, split: str = "train", coord_scale: float = COORD_SCALE,
                 max_len: int | None = None, limit: int | None = None, augment: bool = False,
                 seed: int = 0):
        self.root = Path(root)
        self.coord_scale = coord_scale
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        z = np.load(self.root / "tokens.npz")
        self.tokens, self.offsets = z["tokens"], z["offsets"]
        pos = {str(sid): i for i, sid in enumerate(z["ids"])}
        g = self.root / "graphs"
        self.node_x = np.load(g / "node_x.npy", mmap_mode="r")
        self.edge_x = np.load(g / "edge_x.npy", mmap_mode="r")
        self.edge_index = np.load(g / "edge_index.npy", mmap_mode="r")
        self.node_off = np.load(g / "node_off.npy")
        self.edge_off = np.load(g / "edge_off.npy")
        self.extra = {k: np.load(g / f"{k}.npy", mmap_mode="r")
                      for k in LABEL_KEYS + ["face_attr", "edge_attr"] if (g / f"{k}.npy").exists()}

        split_ids = json.loads((self.root / "splits.json").read_text(encoding="utf-8"))[split]
        self.items = [(sid, pos[sid]) for sid in split_ids]
        if max_len is not None:
            self.items = [(s, k) for s, k in self.items if self.offsets[k + 1] - self.offsets[k] <= max_len]
        if limit is not None:
            self.items = self.items[:limit]

    @property
    def has_labels(self) -> bool:
        return all(k in self.extra for k in LABEL_KEYS)

    @property
    def has_attrs(self) -> bool:
        return "face_attr" in self.extra and "edge_attr" in self.extra

    def __len__(self) -> int:
        return len(self.items)

    def length(self, i: int) -> int:
        _, k = self.items[i]
        return int(self.offsets[k + 1] - self.offsets[k])

    def __getitem__(self, i: int) -> dict:
        sid, k = self.items[i]
        a, b = self.node_off[k], self.node_off[k + 1]
        c, d = self.edge_off[k], self.edge_off[k + 1]
        raw = {"node_x": np.array(self.node_x[a:b]), "edge_x": np.array(self.edge_x[c:d])}
        if self.has_attrs:
            raw["face_attr"] = np.array(self.extra["face_attr"][a:b])
            raw["edge_attr"] = np.array(self.extra["edge_attr"][c:d])
        if self.augment:
            from .augment import random_pose

            raw = random_pose(raw, self.rng)
        raw = scale_features(raw, self.coord_scale)
        item = {
            "id": sid,
            "edge_index": torch.from_numpy(np.array(self.edge_index[:, c:d], dtype=np.int64)),
            "tokens": torch.from_numpy(self.tokens[self.offsets[k]:self.offsets[k + 1]].astype(np.int64)),
            **{key: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for key, v in raw.items()},
        }
        for key in LABEL_KEYS:
            if key in self.extra:
                item[key] = torch.from_numpy(np.array(self.extra[key][a:b], dtype=np.int64))
        return item


def collate(items: list[dict]) -> dict:
    node_x, edge_x, edge_index, node_batch = [], [], [], []
    base = 0
    for i, it in enumerate(items):
        n = it["node_x"].shape[0]
        node_x.append(it["node_x"])
        edge_x.append(it["edge_x"])
        edge_index.append(it["edge_index"] + base)
        node_batch.append(torch.full((n,), i, dtype=torch.long))
        base += n
    lengths = torch.tensor([len(it["tokens"]) for it in items], dtype=torch.long)
    tokens = torch.full((len(items), int(lengths.max())), PAD_ID, dtype=torch.long)
    for i, it in enumerate(items):
        tokens[i, :len(it["tokens"])] = it["tokens"]
    out = {
        "node_x": torch.cat(node_x), "edge_x": torch.cat(edge_x),
        "edge_index": torch.cat(edge_index, dim=1), "node_batch": torch.cat(node_batch),
        "tokens": tokens, "lengths": lengths, "ids": [it["id"] for it in items],
    }
    for key in ["face_attr", "edge_attr"] + LABEL_KEYS:
        if all(key in it for it in items):
            out[key] = torch.cat([it[key] for it in items])
    return out


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class LengthBucketSampler(Sampler):
    """Batches of similar token length (less padding); order randomized per epoch."""

    def __init__(self, ds: CADGraphSeqDataset, batch_size: int, shuffle: bool = True,
                 bucket_batches: int = 50, seed: int = 0):
        self.lengths = [ds.length(i) for i in range(len(ds))]
        self.bs, self.shuffle, self.bucket, self.seed, self.epoch = batch_size, shuffle, bucket_batches, seed, 0

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        if self.shuffle:
            rng.shuffle(idx)
        chunk = self.bs * self.bucket
        batches = []
        for s in range(0, len(idx), chunk):
            part = sorted(idx[s:s + chunk], key=self.lengths.__getitem__)
            batches += [part[j:j + self.bs] for j in range(0, len(part), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs
