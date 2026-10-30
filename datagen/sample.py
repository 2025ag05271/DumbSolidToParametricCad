"""Generate, validate and export one synthetic sample."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import cadquery as cq
import numpy as np

from cadseq.build import QUANTIZED_EDGE_TOL, build, summarize
from cadseq.graph import brep_to_graph_arrays, is_connected
from cadseq.schema import CADSequence
from cadseq.tokenizer import decode, encode

from . import features as F
from .context import Gen
from .families import FAMILIES


class SampleRejected(RuntimeError):
    pass


def generate(seed: int) -> tuple[CADSequence, cq.Shape, dict]:
    r = random.Random(seed)
    g = Gen(r)
    names = list(FAMILIES)
    family = r.choices(names, [FAMILIES[n][1] for n in names])[0]
    if not FAMILIES[family][0](g) or g.body is None:
        raise SampleRejected(f"base family {family} failed")
    turned = bool(g.hints.get("turned"))

    plan = [
        (F.add_boss, r.choices([0, 1, 2, 3], [0.5, 0.3, 0.15, 0.05])[0] if not turned else r.choices([0, 1], [0.8, 0.2])[0]),
        ("deferred", None),
        (F.add_pocket, r.choices([0, 1, 2, 3, 4], [0.15, 0.3, 0.25, 0.18, 0.12])[0]),
        (F.add_hole_group, r.choices([0, 1, 2, 3, 4], [0.1, 0.3, 0.3, 0.2, 0.1])[0]),
        (F.add_fillet, r.choices([0, 1, 2, 3, 4], [0.15, 0.3, 0.25, 0.2, 0.1])[0]),
        (F.add_chamfer, r.choices([0, 1, 2, 3], [0.35, 0.35, 0.2, 0.1])[0]),
        (F.add_mirror, int(r.random() < (0.08 if turned else 0.18))),
    ]
    for stage, n in plan:
        if stage == "deferred":
            for run in g.deferred_pockets:
                run(g)
            continue
        done, attempts = 0, 0
        while done < n and attempts < n * 4 and not g.full():
            attempts += 1
            done += bool(stage(g))

    seq = g.finalize()
    meta = {"family": family, "stats": dict(g.stats)}
    return seq, g.body, meta


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(1e-9, abs(b))


def validate_and_export(seq: CADSequence, body: cq.Shape, out_dir: Path, sid: str, uv_n: int = 10) -> dict:
    """Canonicalizes via the tokenizer, replays it from scratch, round-trips STEP and builds the graph.

    The saved JSON is exactly ``decode(encode(seq))`` and the STEP is built from it, so
    JSON ↔ tokens is lossless for every dataset sample.
    """
    tokens = encode(seq)
    canon = decode(tokens)
    if encode(canon) != tokens:
        raise SampleRejected("tokenizer not idempotent")
    rebuilt = build(canon, edge_tol=QUANTIZED_EDGE_TOL)
    if _rel(rebuilt.Volume(), body.Volume()) > 1e-6 or len(rebuilt.Faces()) != len(body.Faces()):
        raise SampleRejected("canonical replay differs from incremental build")

    shard = sid[:-3] or "0"
    step_path = out_dir / "step" / shard / f"{sid}.step"
    json_path = out_dir / "json" / shard / f"{sid}.json"
    graph_path = out_dir / "graph" / shard / f"{sid}.npz"
    for p in (step_path, json_path, graph_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    cq.exporters.export(cq.Workplane().add(rebuilt), str(step_path), cq.exporters.ExportTypes.STEP)
    solids = cq.importers.importStep(str(step_path)).solids().vals()
    if len(solids) != 1:
        raise SampleRejected(f"STEP re-import gave {len(solids)} solids")
    loaded = solids[0]
    if not loaded.isValid() or len(loaded.Faces()) != len(rebuilt.Faces()) \
            or _rel(loaded.Volume(), rebuilt.Volume()) > 1e-4:
        raise SampleRejected("STEP round-trip mismatch")

    arrays = brep_to_graph_arrays(loaded.wrapped, uv_n)
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise SampleRejected("NaN/inf in graph features")
    if not is_connected(arrays["node_x"].shape[0], arrays["edge_index"]):
        raise SampleRejected("face graph not connected")
    empty_mask = int((arrays["node_x"][..., 6].reshape(len(arrays["node_x"]), -1).sum(1) == 0).sum())
    np.savez_compressed(graph_path, **arrays)

    # JSON is written last: its presence marks a completed sample (resumable runs).
    json_path.write_text(json.dumps(canon.model_dump(), separators=(",", ":")), encoding="utf-8")

    ops = Counter(f.type for f in canon.features)
    return {
        **summarize(rebuilt),
        "ops": dict(ops),
        "n_features": len(canon.features),
        "n_tokens": len(tokens),
        "hole_kinds": dict(Counter(f.kind for f in canon.features if f.type == "hole")),
        "empty_mask_faces": empty_mask,
        "graph_edges": int(arrays["edge_index"].shape[1]),
    }


def make_sample(index: int, out_dir: Path, base_seed: int = 0, uv_n: int = 10, max_attempts: int = 6,
                attempt_offset: int = 0) -> dict:
    """Retries with derived seeds until one sample passes every check."""
    sid = f"{index:08d}"
    errors = []
    for attempt in range(attempt_offset, attempt_offset + max_attempts):
        seed = (base_seed * 1_000_003 + index) * 64 + attempt
        t0 = time.perf_counter()
        try:
            seq, body, meta = generate(seed)
            info = validate_and_export(seq, body, out_dir, sid, uv_n)
            return {"id": sid, "seed": seed, "attempt": attempt, **meta, **info,
                    "seconds": round(time.perf_counter() - t0, 2)}
        except Exception as e:  # any kernel/validation failure → retry with next seed
            errors.append(f"{type(e).__name__}: {str(e)[:160]}")
    return {"id": sid, "failed": True, "errors": errors}
