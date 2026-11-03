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
                 seed: int = 0, tokens: str = "tokens.npz"):
        self.root = Path(root)
        self.coord_scale = coord_scale
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        z = np.load(self.root / tokens)
        self.tokens, self.offsets = z["tokens"], z["offsets"]
        pos = {str(sid): i for i, sid in enumerate(z["ids"])}
        g = self.root / "graphs"
        self.node_off = np.load(g / "node_off.npy")
        self.edge_off = np.load(g / "edge_off.npy")
        self._open()

        split_ids = json.loads((self.root / "splits.json").read_text(encoding="utf-8"))[split]
        self.items = [(sid, pos[sid]) for sid in split_ids]
        if max_len is not None:
            self.items = [(s, k) for s, k in self.items if self.offsets[k + 1] - self.offsets[k] <= max_len]
        if limit is not None:
            self.items = self.items[:limit]

    _MMAPS = ("node_x", "edge_x", "edge_index", "extra")

    def _open(self) -> None:
        """(Re)open the memory-mapped arrays (also called in DataLoader worker processes)."""
        g = self.root / "graphs"
        self.node_x = np.load(g / "node_x.npy", mmap_mode="r")
        self.edge_x = np.load(g / "edge_x.npy", mmap_mode="r")
        self.edge_index = np.load(g / "edge_index.npy", mmap_mode="r")
        self.extra = {k: np.load(g / f"{k}.npy", mmap_mode="r")
                      for k in LABEL_KEYS + ["face_attr", "edge_attr"] if (g / f"{k}.npy").exists()}

    def __getstate__(self):  # pickling a memmap copies the whole file: drop and reopen instead
        return {k: v for k, v in self.__dict__.items() if k not in self._MMAPS}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

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


class CopySeqDataset(CADGraphSeqDataset):
    """Items for the v2 copy decoder: graph + analytic attributes + labels + **candidate values**
    (exact numbers of the input B-Rep, see ``cadseq.candidates``) and, for every number in the target
    tokens, the candidates holding its value (copy targets).

    ``pose_aug`` = probability of a random axis-aligned rotation / reflection + grid translation,
    applied consistently to the history (re-tokenized), the graph features and the candidates.
    ``fixed_pose=True`` draws the pose from the item index (reproducible posed evaluation); the pose
    is returned as ``pose = (R, t)``.

    ``edge_ptr=True`` (phase 3b) uses the cleaned / re-targeted histories of ``cadseq.edgefix``
    (``tokens_ef.npz``) and adds the part's **virtual edges** (sharp edges under blend faces) with,
    for every fillet / chamfer edge ref, the virtual edges it may point to (see ``vedge_item``).
    """

    MAX_CANDIDATES = 4096
    MATCH_TOL = 0.0249  # copying a candidate emits its value rounded to the 0.05 mm grid
    _MMAPS = CADGraphSeqDataset._MMAPS + ("prim", "vedge")

    def __init__(self, root, split: str = "train", pose_aug: float = 0.0, translate: float = 100.0,
                 fixed_pose: bool = False, edge_ptr: bool = False, **kw):
        if edge_ptr:
            kw.setdefault("tokens", "tokens_ef.npz")
        super().__init__(root, split, **kw)
        g = self.root / "graphs"
        self.prim_off = {k: np.load(g / f"prim_{k}_off.npy") for k in ("points", "scalars", "planes")}
        self.pose_aug, self.translate, self.fixed_pose = pose_aug, translate, fixed_pose
        self.edge_ptr = edge_ptr
        if edge_ptr:
            self.vedge_off = np.load(g / "vedge_off.npy")
        self._open()

    def _open(self) -> None:
        super()._open()
        if hasattr(self, "prim_off"):
            g = self.root / "graphs"
            self.prim = {k: np.load(g / f"prim_{k}.npy", mmap_mode="r")
                         for k in ("points", "point_kind", "scalars", "scalar_kind", "plane_normal", "plane_point")}
            if self.edge_ptr:
                self.vedge = {k: np.load(g / f"vedge_{k}.npy", mmap_mode="r")
                              for k in ("point", "curve", "face", "sup", "group")}

    def primitives(self, k: int) -> dict:
        a, b = self.prim_off["points"][k], self.prim_off["points"][k + 1]
        c, d = self.prim_off["scalars"][k], self.prim_off["scalars"][k + 1]
        e, f = self.prim_off["planes"][k], self.prim_off["planes"][k + 1]
        return {"points": np.array(self.prim["points"][a:b]), "point_kind": np.array(self.prim["point_kind"][a:b]),
                "scalars": np.array(self.prim["scalars"][c:d]), "scalar_kind": np.array(self.prim["scalar_kind"][c:d]),
                "plane_normal": np.array(self.prim["plane_normal"][e:f]),
                "plane_point": np.array(self.prim["plane_point"][e:f])}

    def __getitem__(self, i: int) -> dict:
        from .augment import transform
        from .candidates import TAGS, candidate_values, transform_primitives, value_features
        from .pose import random_pose, try_transform
        from .tokenizer import COARSE_BASE, FINE_BASE, NUM_COARSE, NUM_FINE, NUM_OFFSET, decode, dequantize, encode

        sid, k = self.items[i]
        a, b = self.node_off[k], self.node_off[k + 1]
        c, d = self.edge_off[k], self.edge_off[k + 1]
        raw = {"node_x": np.array(self.node_x[a:b]), "edge_x": np.array(self.edge_x[c:d]),
               "face_attr": np.array(self.extra["face_attr"][a:b]), "edge_attr": np.array(self.extra["edge_attr"][c:d])}
        tokens = self.tokens[self.offsets[k]:self.offsets[k + 1]].astype(np.int64)
        prims = self.primitives(k)

        rng = np.random.default_rng(i) if self.fixed_pose else self.rng
        R, t = np.eye(3), np.zeros(3)
        if self.pose_aug > 0 and rng.random() < self.pose_aug:
            R2, t2 = random_pose(rng, self.translate)
            seq2 = try_transform(decode(tokens.tolist()), R2, t2)
            if seq2 is not None:
                R, t = R2, t2
                tokens = np.asarray(encode(seq2), dtype=np.int64)
                raw = transform(raw, R, t.astype(np.float32), 1.0)
                prims = transform_primitives(prims, R, t)

        values, tags = candidate_values(prims)
        if len(values) > self.MAX_CANDIDATES:  # keep the smallest-magnitude values (rare)
            keep = np.sort(np.argsort(np.abs(values))[: self.MAX_CANDIDATES])
            values, tags = values[keep], tags[keep]
        quant = np.round(values * 20).astype(np.int64)  # tokenizer grid (Q_PER_MM = 20)
        num_pos, pairs = [], []
        coarse = (tokens >= COARSE_BASE) & (tokens < COARSE_BASE + NUM_COARSE)
        for p in np.nonzero(coarse)[0]:
            v = dequantize((tokens[p] - COARSE_BASE) * NUM_FINE + (tokens[p + 1] - FINE_BASE) - NUM_OFFSET)
            lo = np.searchsorted(values, v - self.MATCH_TOL)
            hi = np.searchsorted(values, v + self.MATCH_TOL)
            for cidx in range(lo, hi):
                pairs.append((len(num_pos), cidx))
            num_pos.append(p)

        raw = scale_features(raw, self.coord_scale)
        item = {
            "id": sid, "pose": (R, t),
            "edge_index": torch.from_numpy(np.array(self.edge_index[:, c:d], dtype=np.int64)),
            "tokens": torch.from_numpy(tokens),
            **{key: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for key, v in raw.items()},
            "cand_feat": torch.from_numpy(np.concatenate([value_features(values), tags.astype(np.float32)], 1)),
            "cand_quant": torch.from_numpy(quant),
            "cand_value": torch.from_numpy(values),
            "num_pos": torch.tensor(num_pos, dtype=torch.long),
            "copy_pairs": torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
        }
        for key in LABEL_KEYS:
            if key in self.extra:
                item[key] = torch.from_numpy(np.array(self.extra[key][a:b], dtype=np.int64))
        if self.edge_ptr:
            e, f = self.vedge_off[k], self.vedge_off[k + 1]
            point = np.array(self.vedge["point"][e:f], dtype=np.float64)
            point = point @ R.T + t
            item.update(vedge_item(point, np.array(self.vedge["curve"][e:f]), np.array(self.vedge["face"][e:f]),
                                   np.array(self.vedge["sup"][e:f]), tokens, np.array(self.vedge["group"][e:f])))
        return item


def vedge_item(point, curve, face, sup, tokens=None, group=None) -> dict:
    """Virtual-edge inputs of one part (points already in the item's pose) and, given the target
    ``tokens``, the edge-ref pointer targets:

    ``ve_feat`` (V, 3·VALUE_FEAT_DIM + 5) Fourier features of x, y, z + curve one-hot; ``ve_quant``
    (V, 3) grid integers; ``ve_curve`` / ``ve_face`` / ``ve_sup``; ``er_pos`` position of each edge
    ref's first coordinate token; ``er_curve``; ``er_pairs`` (ref, virtual edge) — the virtual edges
    at exactly the ref's point plus the others of their group (same intermediate-body edge)."""
    from .candidates import value_features
    from .edgefix import CURVES
    from .tokenizer import COARSE_BASE, EDGE_TYPES, FINE_BASE, NUM_FINE, NUM_OFFSET, TOKEN_ID

    point = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    quant = np.round(point * 20).astype(np.int64)
    curve = np.asarray(curve, dtype=np.int64)
    onehot = np.eye(len(CURVES), dtype=np.float32)[curve]
    feat = np.concatenate([value_features(point[:, 0]), value_features(point[:, 1]), value_features(point[:, 2]),
                           onehot], 1).astype(np.float32)
    out = {"ve_feat": torch.from_numpy(feat), "ve_quant": torch.from_numpy(quant), "ve_curve": torch.from_numpy(curve),
           "ve_face": torch.from_numpy(np.asarray(face, dtype=np.int64)),
           "ve_sup": torch.from_numpy(np.asarray(sup, dtype=np.int64).reshape(-1, 2))}
    if tokens is None:
        return out
    e_ids = {TOKEN_ID[v]: CURVES.index(k) for k, v in EDGE_TYPES.items() if k in CURVES}
    er_pos, er_curve, pairs = [], [], []
    for p in np.nonzero(np.isin(tokens, list(e_ids)))[0]:
        q = np.array([(tokens[p + 1 + 2 * c] - COARSE_BASE) * NUM_FINE + (tokens[p + 2 + 2 * c] - FINE_BASE) - NUM_OFFSET
                      for c in range(3)])
        cid = e_ids[int(tokens[p])]
        m = np.nonzero(np.all(quant == q, 1) & (curve == cid))[0]
        if len(m) and group is not None:
            g = group[m]
            g = g[g >= 0]
            if len(g):
                m = np.union1d(m, np.nonzero((group == g[0]) & (curve == cid))[0])
        pairs += [(len(er_pos), int(j)) for j in m]
        er_pos.append(int(p) + 1)
        er_curve.append(cid)
    out.update({"er_pos": torch.tensor(er_pos, dtype=torch.long), "er_curve": torch.tensor(er_curve, dtype=torch.long),
                "er_pairs": torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)})
    return out


def collate_copy(items: list[dict]) -> dict:
    """``collate`` + padded candidates (B, C, F) with pad mask, numbers and copy-target pairs."""
    out = collate(items)
    C = max(len(it["cand_quant"]) for it in items)
    F = items[0]["cand_feat"].shape[1]
    B = len(items)
    feat = torch.zeros(B, C, F)
    quant = torch.zeros(B, C, dtype=torch.long)
    value = torch.zeros(B, C, dtype=torch.float64)
    pad = torch.ones(B, C, dtype=torch.bool)
    num_b, num_pos, pair_num, pair_cand = [], [], [], []
    n_base = 0
    for b, it in enumerate(items):
        n = len(it["cand_quant"])
        feat[b, :n], quant[b, :n], value[b, :n], pad[b, :n] = it["cand_feat"], it["cand_quant"], it["cand_value"], False
        num_b.append(torch.full((len(it["num_pos"]),), b, dtype=torch.long))
        num_pos.append(it["num_pos"])
        if len(it["copy_pairs"]):
            pair_num.append(it["copy_pairs"][:, 0] + n_base)
            pair_cand.append(it["copy_pairs"][:, 1])
        n_base += len(it["num_pos"])
    out.update({
        "cand_feat": feat, "cand_quant": quant, "cand_value": value, "cand_pad": pad,
        "num_b": torch.cat(num_b), "num_pos": torch.cat(num_pos),
        "pair_num": torch.cat(pair_num) if pair_num else torch.zeros(0, dtype=torch.long),
        "pair_cand": torch.cat(pair_cand) if pair_cand else torch.zeros(0, dtype=torch.long),
        "poses": [it["pose"] for it in items],
    })
    if "ve_feat" in items[0]:
        out.update(_collate_vedges(items))
    return out


def _collate_vedges(items: list[dict]) -> dict:
    B = len(items)
    V = max(1, max(len(it["ve_curve"]) for it in items))
    F = items[0]["ve_feat"].shape[1]
    feat, quant = torch.zeros(B, V, F), torch.zeros(B, V, 3, dtype=torch.long)
    curve, face = torch.full((B, V), -1, dtype=torch.long), torch.zeros(B, V, dtype=torch.long)
    sup, pad = torch.zeros(B, V, 2, dtype=torch.long), torch.ones(B, V, dtype=torch.bool)
    er_b, er_pos, er_curve, pr, pv = [], [], [], [], []
    base = 0
    for b, it in enumerate(items):
        n = len(it["ve_curve"])
        feat[b, :n], quant[b, :n], curve[b, :n] = it["ve_feat"], it["ve_quant"], it["ve_curve"]
        face[b, :n], sup[b, :n], pad[b, :n] = it["ve_face"], it["ve_sup"], False
        if "er_pos" in it:
            er_b.append(torch.full((len(it["er_pos"]),), b, dtype=torch.long))
            er_pos.append(it["er_pos"])
            er_curve.append(it["er_curve"])
            if len(it["er_pairs"]):
                pr.append(it["er_pairs"][:, 0] + base)
                pv.append(it["er_pairs"][:, 1])
            base += len(it["er_pos"])
    z = torch.zeros(0, dtype=torch.long)
    return {"ve_feat": feat, "ve_quant": quant, "ve_curve": curve, "ve_face": face, "ve_sup": sup, "ve_pad": pad,
            "er_b": torch.cat(er_b) if er_b else z, "er_pos": torch.cat(er_pos) if er_pos else z,
            "er_curve": torch.cat(er_curve) if er_curve else z,
            "er_pair_ref": torch.cat(pr) if pr else z, "er_pair_ve": torch.cat(pv) if pv else z}


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
