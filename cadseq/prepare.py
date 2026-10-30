"""Pack a generated dataset for training: tokens, splits and vocab.

Usage:
    python -m cadseq.prepare data/synth_v2

Writes into the dataset folder:
    tokens.npz    tokens (uint16, concatenated), offsets (N+1), ids (N)
    graphs/       packed UV-Net arrays in tokens.npz order (memory-mappable .npy):
                  node_x (ΣF,N,N,7) f32, node_off (N+1), edge_x (ΣE,N,6) f32,
                  edge_index (2,ΣE) int32 (local face ids), edge_off (N+1)
    splits.json   {"train": [...], "val": [...], "test": [...]} stratified by family (90/5/5)
    vocab.json    token strings in id order
    tokens_stats.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .schema import CADSequence
from .tokenizer import VOCAB, decode, encode


def sample_path(root: Path, kind: str, sid: str) -> Path:
    """kind: 'step' | 'json' | 'graph'."""
    ext = {"step": "step", "json": "json", "graph": "npz"}[kind]
    return Path(root) / kind / (sid[:-3] or "0") / f"{sid}.{ext}"


def load_index(root: Path) -> dict[str, dict]:
    recs = {}
    for line in (Path(root) / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("failed"):
                recs[r["id"]] = r
    return dict(sorted(recs.items()))


def make_splits(index: dict[str, dict], seed: int = 0, val: float = 0.05, test: float = 0.05) -> dict:
    by_family = defaultdict(list)
    for sid, r in index.items():
        by_family[r["family"]].append(sid)
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for fam in sorted(by_family):
        ids = sorted(by_family[fam])
        rng.shuffle(ids)
        n_val, n_test = round(len(ids) * val), round(len(ids) * test)
        out["val"] += ids[:n_val]
        out["test"] += ids[n_val:n_val + n_test]
        out["train"] += ids[n_val + n_test:]
    return {k: sorted(v) for k, v in out.items()}


def pack_graphs(root: Path, ids: list[str]) -> dict:
    """Concatenate per-sample graph .npz files into memory-mappable arrays (two passes, low RAM)."""
    out = Path(root) / "graphs"
    out.mkdir(exist_ok=True)
    nf, ne = [], []
    for sid in ids:
        with np.load(sample_path(root, "graph", sid)) as z:
            nf.append(z["node_x"].shape[0])
            ne.append(z["edge_x"].shape[0])
            n = z["node_x"].shape[1]
    node_off = np.concatenate([[0], np.cumsum(nf)]).astype(np.int64)
    edge_off = np.concatenate([[0], np.cumsum(ne)]).astype(np.int64)
    nf_total, ne_total = int(node_off[-1]), int(edge_off[-1])  # plain ints: np.int64 breaks .npy headers
    node_x = np.lib.format.open_memmap(out / "node_x.npy", "w+", np.float32, (nf_total, n, n, 7))
    edge_x = np.lib.format.open_memmap(out / "edge_x.npy", "w+", np.float32, (ne_total, n, 6))
    edge_index = np.lib.format.open_memmap(out / "edge_index.npy", "w+", np.int32, (2, ne_total))
    for k, sid in enumerate(ids):
        with np.load(sample_path(root, "graph", sid)) as z:
            node_x[node_off[k]:node_off[k + 1]] = z["node_x"]
            edge_x[edge_off[k]:edge_off[k + 1]] = z["edge_x"]
            edge_index[:, edge_off[k]:edge_off[k + 1]] = z["edge_index"]
    for a in (node_x, edge_x, edge_index):
        a.flush()
    np.save(out / "node_off.npy", node_off)
    np.save(out / "edge_off.npy", edge_off)
    return {"faces": int(node_off[-1]), "directed_edges": int(edge_off[-1]), "uv": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = args.dataset

    index = load_index(root)
    ids, chunks, bad = [], [], []
    for sid in index:
        data = json.loads(sample_path(root, "json", sid).read_text(encoding="utf-8"))
        toks = encode(CADSequence.model_validate(data))
        # Lossless check: decoding the tokens must give back the stored JSON exactly.
        if json.loads(json.dumps(decode(toks).model_dump())) != data:
            bad.append(sid)
            continue
        ids.append(sid)
        chunks.append(np.asarray(toks, dtype=np.uint16))
    if bad:
        print(f"WARNING: {len(bad)} samples are not token-lossless and were skipped: {bad[:10]}")

    lengths = np.array([len(c) for c in chunks])
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(root / "tokens.npz", tokens=np.concatenate(chunks), offsets=offsets, ids=np.array(ids))
    graph_stats = pack_graphs(root, ids)

    splits = make_splits({sid: index[sid] for sid in ids}, args.seed)
    (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (root / "vocab.json").write_text(json.dumps(VOCAB), encoding="utf-8")
    stats = {
        "samples": len(ids),
        "not_lossless": len(bad),
        "vocab_size": len(VOCAB),
        "splits": {k: len(v) for k, v in splits.items()},
        "length": {p: int(np.percentile(lengths, p)) for p in (0, 50, 90, 99, 100)},
        "total_tokens": int(lengths.sum()),
        "graphs": graph_stats,
    }
    (root / "tokens_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
