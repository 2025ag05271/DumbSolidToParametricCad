"""Phase 3b data pass: clean and re-target fillet / chamfer edge references.

For every sample:

1. **Virtual edges**: ``cadseq.blends.sharp_edges`` for *every* face of the input STEP (label-free,
   so inference computes exactly the same set); points snapped to the tokenizer grid.
2. **Replay** the (quantized) history. For each fillet / chamfer edge ref the matched edge of the
   intermediate body is known:
   - refs to smooth edges (seams, tangent joins — no-ops that leave no trace in the final solid,
     ~21 % of all refs in synth_v2) are dropped, and fillets / chamfers left without refs too;
   - virtual edges lying on that exact edge (≤ ``ON_EDGE`` mm) form the ref's **group** (pointer
     targets) and the ref point is moved to the first one's snapped point.
3. The new history is re-tokenized and rebuilt strictly; it must reproduce the STEP volume, else
   the sample keeps its original tokens (``status = "fallback"``).

Writes ``tokens_ef.npz`` (same ids / order as tokens.npz) and ``graphs/vedge_*.npy``:
``point`` (V, 3) float64 snapped, ``curve`` (V,) int8 (index in ``CURVES``), ``face`` / ``sup``
(V,) / (V, 2) int32 face indices, ``group`` (V,) int32 (−1 = no ref), ``off`` (N+1,) int64.

    .venv\\Scripts\\python.exe -m cadseq.edgefix data\\synth_v2 --workers 16
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .prepare import sample_path

CURVES = ["line", "circle", "ellipse", "bspline", "other"]
ON_EDGE = 0.05      # virtual point (snapped) within this distance of the referenced edge → same edge
VOLUME_TOL = 1e-3   # relative volume difference allowed between the new rebuild and the STEP


def snap(p) -> np.ndarray:
    return np.round(np.asarray(p, dtype=np.float64) * 20.0) / 20.0


def virtual_edges(shape) -> dict:
    """Label-free virtual edge arrays for one solid (also used at inference)."""
    from .blends import sharp_edges
    from .tokenizer import NUM_OFFSET

    n_faces = len(shape.Faces())
    ve = sharp_edges(shape, range(n_faces))
    lim = (NUM_OFFSET - 1) / 20.0
    keep = [v for v in ve if np.all(np.abs(v.point) < lim)]
    return {"point": snap(np.array([v.point for v in keep]).reshape(-1, 3)),
            "curve": np.array([CURVES.index(v.curve) for v in keep], np.int8),
            "face": np.array([v.face for v in keep], np.int32),
            "sup": np.array([v.supports for v in keep], np.int32).reshape(-1, 2)}


def fix_sample(task) -> dict:
    import cadquery as cq

    from .build import QUANTIZED_EDGE_TOL, Builder, build, is_smooth_edge
    from .schema import CADSequence, Chamfer, EdgeRef, Fillet
    from .tokenizer import decode, encode

    root, sid, tokens = task
    root = Path(root)
    shape = cq.importers.importStep(str(sample_path(root, "step", sid))).val()
    ve = virtual_edges(shape)
    group = np.full(len(ve["curve"]), -1, np.int32)
    seq = decode(list(tokens))
    stats = Counter()
    try:
        b = Builder(edge_tol=QUANTIZED_EDGE_TOL)
        kept, n_group = [], 0
        for f in seq.features:
            if isinstance(f, (Fillet, Chamfer)):
                refs = []
                for r, e in zip(f.edges, b._edges(f.edges)):
                    stats["refs"] += 1
                    if is_smooth_edge(b.body, e):
                        stats["smooth"] += 1
                        continue
                    bb = e.BoundingBox()
                    lo = np.array([bb.xmin, bb.ymin, bb.zmin]) - 0.1
                    hi = np.array([bb.xmax, bb.ymax, bb.zmax]) + 0.1
                    cid = CURVES.index(r.curve)
                    near = np.nonzero((ve["curve"] == cid) & np.all((ve["point"] >= lo) & (ve["point"] <= hi), 1))[0]
                    match = [j for j in near
                             if e.distance(cq.Vertex.makeVertex(*ve["point"][j])) <= ON_EDGE]
                    if match:
                        free = [j for j in match if group[j] < 0]
                        group[free] = n_group
                        n_group += 1
                        r = EdgeRef(point=tuple(float(c) for c in ve["point"][match[0]]), curve=r.curve)
                        stats["matched"] += 1
                    refs.append(r)
                if not refs:
                    stats["features_dropped"] += 1
                    continue
                f = f.model_copy(update={"edges": refs})
            b.apply(f)
            kept.append(f)
        new = encode(CADSequence.model_validate({"features": [f.model_dump() for f in kept]}))
        body = build(decode(new), QUANTIZED_EDGE_TOL)
        if abs(body.Volume() - shape.Volume()) > VOLUME_TOL * shape.Volume():
            raise RuntimeError(f"volume {body.Volume():.3f} vs STEP {shape.Volume():.3f}")
        status = "ok"
    except Exception as e:  # keep the original history
        new, status = list(tokens), f"fallback: {str(e)[:120]}"
        group[:] = -1
        stats = Counter(refs=stats["refs"])
    return {"id": sid, "status": status, "tokens": np.asarray(new, np.int16), "group": group,
            "stats": dict(stats), **ve}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--limit", type=int, default=None, help="process the first N samples only (no packing)")
    args = ap.parse_args()

    from cadseq.workers import SupervisedPool
    from s2pnet.train import keep_awake

    keep_awake()
    root = args.dataset.resolve()
    z = np.load(root / "tokens.npz")
    ids, toks, offs = [str(s) for s in z["ids"]], z["tokens"], z["offsets"]
    n = len(ids) if args.limit is None else args.limit
    tasks = [(str(root), ids[k], toks[offs[k]:offs[k + 1]].tolist()) for k in range(n)]
    results: list = [None] * n
    t0, stats, status = time.time(), Counter(), Counter()
    pool = SupervisedPool(fix_sample, args.workers, args.timeout)
    for k, (i, st, res) in enumerate(pool.run(tasks), 1):
        if st != "ok":
            res = {"id": ids[i], "status": f"fallback: worker {st}", "tokens": np.asarray(tasks[i][2], np.int16),
                   "group": np.zeros(0, np.int32), "stats": {}, "point": np.zeros((0, 3)),
                   "curve": np.zeros(0, np.int8), "face": np.zeros(0, np.int32), "sup": np.zeros((0, 2), np.int32)}
        results[i] = res
        stats.update(res["stats"])
        status[res["status"].split(":")[0]] += 1
        if k % 500 == 0 or k == n:
            el = time.time() - t0
            print(f"[{k}/{n}] {dict(status)} {dict(stats)} {k / el:.1f}/s eta {(n - k) * el / k / 60:.1f} min",
                  flush=True)
    pool.close()

    summary = {"samples": n, "status": dict(status), **dict(stats),
               "fallbacks": [(r["id"], r["status"]) for r in results if r["status"] != "ok"][:50],
               "seconds": round(time.time() - t0, 1)}
    if args.limit is not None:
        print(json.dumps({k: v for k, v in summary.items() if k != "fallbacks"}, indent=1))
        print(summary["fallbacks"][:10])
        return
    lengths = np.array([len(r["tokens"]) for r in results])
    np.savez(root / "tokens_ef.npz", ids=z["ids"], tokens=np.concatenate([r["tokens"] for r in results]).astype(z["tokens"].dtype),
             offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(offs.dtype))
    g = root / "graphs"
    for key in ("point", "curve", "face", "sup", "group"):
        np.save(g / f"vedge_{key}.npy", np.concatenate([r[key] for r in results]))
    np.save(g / "vedge_off.npy", np.concatenate([[0], np.cumsum([len(r["curve"]) for r in results])]).astype(np.int64))
    (root / "edgefix_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "fallbacks"}, indent=1))


if __name__ == "__main__":
    main()
