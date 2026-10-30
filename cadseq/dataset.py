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


class CADGraphSeqDataset(Dataset):
    def __init__(self, root, split: str = "train", coord_scale: float = COORD_SCALE,
                 max_len: int | None = None, limit: int | None = None):
        self.root = Path(root)
        self.coord_scale = coord_scale
        z = np.load(self.root / "tokens.npz")
        self.tokens, self.offsets = z["tokens"], z["offsets"]
        pos = {str(sid): i for i, sid in enumerate(z["ids"])}
        g = self.root / "graphs"
        self.node_x = np.load(g / "node_x.npy", mmap_mode="r")
        self.edge_x = np.load(g / "edge_x.npy", mmap_mode="r")
        self.edge_index = np.load(g / "edge_index.npy", mmap_mode="r")
        self.node_off = np.load(g / "node_off.npy")
        self.edge_off = np.load(g / "edge_off.npy")

        split_ids = json.loads((self.root / "splits.json").read_text(encoding="utf-8"))[split]
        self.items = [(sid, pos[sid]) for sid in split_ids]
        if max_len is not None:
            self.items = [(s, k) for s, k in self.items if self.offsets[k + 1] - self.offsets[k] <= max_len]
        if limit is not None:
            self.items = self.items[:limit]

    def __len__(self) -> int:
        return len(self.items)

    def length(self, i: int) -> int:
        _, k = self.items[i]
        return int(self.offsets[k + 1] - self.offsets[k])

    def __getitem__(self, i: int) -> dict:
        sid, k = self.items[i]
        a, b = self.node_off[k], self.node_off[k + 1]
        c, d = self.edge_off[k], self.edge_off[k + 1]
        node_x = np.array(self.node_x[a:b])
        edge_x = np.array(self.edge_x[c:d])
        node_x[..., :3] *= self.coord_scale
        edge_x[..., :3] *= self.coord_scale
        return {
            "id": sid,
            "node_x": torch.from_numpy(node_x),
            "edge_x": torch.from_numpy(edge_x),
            "edge_index": torch.from_numpy(np.array(self.edge_index[:, c:d], dtype=np.int64)),
            "tokens": torch.from_numpy(self.tokens[self.offsets[k]:self.offsets[k + 1]].astype(np.int64)),
        }


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
    return {
        "node_x": torch.cat(node_x), "edge_x": torch.cat(edge_x),
        "edge_index": torch.cat(edge_index, dim=1), "node_batch": torch.cat(node_batch),
        "tokens": tokens, "lengths": lengths, "ids": [it["id"] for it in items],
    }


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
