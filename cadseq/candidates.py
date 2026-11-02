"""Geometric number candidates: exact values in the input B-Rep that history numbers can copy.

A design history's numbers (sketch coordinates, plane offsets, hole positions and diameters, fillet
radii, pad lengths, ...) are, for the most part, exact values readable from the solid it produced:
vertex / centre coordinates, plane offsets, radii, angles and distances between parallel planes.
The v2 decoder *points* at these values instead of generating digits, which makes copied numbers
exact (the "geometry-informed" part of the hybrid decoder).

Two stages so that pose augmentation is exact and cheap:
    prims = extract_primitives(shape)          # geometry (points, planes, invariant scalars), mm
    prims = transform_primitives(prims, R, t)  # optional rigid transform (axis-aligned rotations)
    values, tags = candidate_values(prims)     # unique scalar candidates + multi-hot provenance
"""

from __future__ import annotations

import math

import numpy as np

# point kinds
VERTEX, CIRCLE_CENTER, ARC_MID, EDGE_MID, AXIS_POINT = range(5)
POINT_KINDS = ["vertex", "circle_center", "arc_mid", "edge_mid", "axis_point"]
# scalar kinds (pose-invariant)
RADIUS, CONE_ANGLE = range(2)

TAGS = (["coord_x", "coord_y", "coord_z", "negated", "plane"] + POINT_KINDS
        + ["bbox", "radius", "diameter", "angle", "diff_x", "diff_y", "diff_z"])
TAG = {t: i for i, t in enumerate(TAGS)}
N_TAGS = len(TAGS)
MERGE_DECIMALS = 4  # candidates closer than 1e-4 mm are the same value
MAX_ABS = 509.0     # tokenizer range is [-510, 510)


def extract_primitives(shape) -> dict:
    """Exact geometric primitives of a B-Rep solid (OCP TopoDS_Shape or cadquery Shape)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane, \
        GeomAbs_Sphere, GeomAbs_Torus
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool

    s = shape.wrapped if hasattr(shape, "wrapped") else shape
    pts, kinds, scal, skind, plane_n, plane_p = [], [], [], [], [], []

    def xyz(p):
        return (p.X(), p.Y(), p.Z())

    exp = TopExp_Explorer(s, TopAbs_VERTEX)
    seen = set()
    while exp.More():
        p = xyz(BRep_Tool.Pnt_s(TopoDS.Vertex_s(exp.Current())))
        key = tuple(round(c, 6) for c in p)
        if key not in seen:
            seen.add(key)
            pts.append(p)
            kinds.append(VERTEX)
        exp.Next()

    exp = TopExp_Explorer(s, TopAbs_EDGE)
    while exp.More():
        c = BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))
        t0, t1 = c.FirstParameter(), c.LastParameter()
        mid = xyz(c.Value(0.5 * (t0 + t1)))
        if c.GetType() == GeomAbs_Circle:
            circ = c.Circle()
            pts.append(xyz(circ.Location()))
            kinds.append(CIRCLE_CENTER)
            pts.append(mid)
            kinds.append(ARC_MID)
            scal.append(circ.Radius())
            skind.append(RADIUS)
        elif c.GetType() == GeomAbs_Line:
            pts.append(mid)
            kinds.append(EDGE_MID)
        exp.Next()

    exp = TopExp_Explorer(s, TopAbs_FACE)
    while exp.More():
        a = BRepAdaptor_Surface(TopoDS.Face_s(exp.Current()), True)
        t = a.GetType()
        if t == GeomAbs_Plane:
            pl = a.Plane()
            plane_n.append(xyz(pl.Axis().Direction()))
            plane_p.append(xyz(pl.Location()))
        elif t in (GeomAbs_Cylinder, GeomAbs_Cone):
            g = a.Cylinder() if t == GeomAbs_Cylinder else a.Cone()
            pts.append(xyz(g.Axis().Location()))
            kinds.append(AXIS_POINT)
            if t == GeomAbs_Cylinder:
                scal.append(g.Radius())
                skind.append(RADIUS)
            else:
                scal.append(g.SemiAngle())
                skind.append(CONE_ANGLE)
                pts.append(xyz(g.Apex()))
                kinds.append(AXIS_POINT)
        elif t == GeomAbs_Torus:
            g = a.Torus()
            scal += [g.MinorRadius(), g.MajorRadius()]
            skind += [RADIUS, RADIUS]
            pts.append(xyz(g.Axis().Location()))
            kinds.append(AXIS_POINT)
        elif t == GeomAbs_Sphere:
            g = a.Sphere()
            scal.append(g.Radius())
            skind.append(RADIUS)
            pts.append(xyz(g.Location()))
            kinds.append(CIRCLE_CENTER)
        exp.Next()

    return {
        "points": np.asarray(pts, dtype=np.float64).reshape(-1, 3),
        "point_kind": np.asarray(kinds, dtype=np.int8),
        "scalars": np.asarray(scal, dtype=np.float64),
        "scalar_kind": np.asarray(skind, dtype=np.int8),
        "plane_normal": np.asarray(plane_n, dtype=np.float64).reshape(-1, 3),
        "plane_point": np.asarray(plane_p, dtype=np.float64).reshape(-1, 3),
    }


def transform_primitives(prims: dict, R: np.ndarray, t: np.ndarray) -> dict:
    """x → R x + t (R orthogonal). Scalars (radii, angles) are invariant."""
    out = dict(prims)
    out["points"] = prims["points"] @ R.T + t
    out["plane_normal"] = prims["plane_normal"] @ R.T
    out["plane_point"] = prims["plane_point"] @ R.T + t
    return out


def candidate_values(prims: dict, with_diffs: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Unique scalar candidates (C,) float64 and their multi-hot provenance tags (C, N_TAGS) bool."""
    vals, tags = [], []

    def add(v, *tag_names):
        v = np.asarray(v, dtype=np.float64).ravel()
        m = np.zeros((len(v), N_TAGS), dtype=bool)
        for name in tag_names:
            m[:, TAG[name]] = True
        vals.append(v)
        tags.append(m)

    pts, kinds = prims["points"], prims["point_kind"]
    planes_by_axis: list[list[float]] = [[], [], []]
    for n, p in zip(prims["plane_normal"], prims["plane_point"]):
        k = int(np.argmax(np.abs(n)))
        if abs(abs(n[k]) - 1) < 1e-7:
            planes_by_axis[k].append(float(p[k]))
    lo = pts.min(0) if len(pts) else np.zeros(3)
    hi = pts.max(0) if len(pts) else np.zeros(3)
    axis_name = ["coord_x", "coord_y", "coord_z"]
    for k in range(3):
        coords_k = []
        for kind, name in enumerate(POINT_KINDS):
            v = pts[kinds == kind, k]
            add(v, axis_name[k], name)
            add(-v, axis_name[k], name, "negated")
            if kind == VERTEX:
                coords_k.append(v)
        pl = np.asarray(planes_by_axis[k])
        add(pl, axis_name[k], "plane")
        add(-pl, axis_name[k], "plane", "negated")
        bb = np.array([lo[k], hi[k], 0.5 * (lo[k] + hi[k])])
        add(bb, axis_name[k], "bbox")
        add(-bb, axis_name[k], "bbox", "negated")
        if with_diffs:
            u = np.unique(np.round(np.concatenate(coords_k + [pl]), MERGE_DECIMALS))
            if len(u) > 1:
                d = np.abs(u[:, None] - u[None, :])[np.triu_indices(len(u), 1)]
                add(d[d > 1e-6], f"diff_{'xyz'[k]}")
    sc, sk = prims["scalars"], prims["scalar_kind"]
    r = sc[sk == RADIUS]
    add(r, "radius")
    add(2 * r, "diameter")
    add(np.degrees(2 * sc[sk == CONE_ANGLE]), "angle")

    v = np.concatenate(vals)
    m = np.concatenate(tags)
    keep = np.abs(v) < MAX_ABS
    v, m = v[keep], m[keep]
    key = np.round(v, MERGE_DECIMALS)
    uniq, inv = np.unique(key, return_inverse=True)
    merged = np.zeros((len(uniq), N_TAGS), dtype=bool)
    np.logical_or.at(merged, inv, m)
    return uniq, merged


def match(values: np.ndarray, target: float, tol: float = 1e-3) -> np.ndarray:
    """Indices of candidates equal to ``target`` (sorted ``values``)."""
    lo, hi = np.searchsorted(values, target - tol), np.searchsorted(values, target + tol)
    return np.arange(lo, hi)


def value_features(v: np.ndarray, n_freq: int = 12) -> np.ndarray:
    """Fourier features of a value in mm at scales 0.05 mm .. ~1000 mm (for candidate embeddings)."""
    scales = 0.05 * (2.0 ** np.arange(n_freq + 2))  # 0.05 .. 819 mm
    x = np.asarray(v, dtype=np.float64)[:, None] / scales[None, :] * 2 * math.pi
    return np.concatenate([np.sin(x), np.cos(x), np.asarray(v)[:, None] / 100.0], 1).astype(np.float32)


VALUE_FEAT_DIM = 2 * 14 + 1
PRIM_KEYS = ["points", "point_kind", "scalars", "scalar_kind", "plane_normal", "plane_point"]


def _extract_task(task):
    import warnings

    warnings.filterwarnings("ignore")
    import cadquery as cq

    from .prepare import sample_path

    root, sid = task
    shape = cq.importers.importStep(str(sample_path(root, "step", sid))).val()
    return extract_primitives(shape)


def pack_dataset(root, workers: int = 16) -> dict:
    """Extract primitives from every STEP of a dataset and pack them (tokens.npz order) into
    graphs/prim_*.npy with per-sample offsets."""
    import time
    from pathlib import Path

    from .workers import SupervisedPool

    root = Path(root).resolve()
    ids = [str(s) for s in np.load(root / "tokens.npz")["ids"]]
    res: list = [None] * len(ids)
    t0 = time.time()
    pool = SupervisedPool(_extract_task, workers, timeout=120)
    for k, (i, status, r) in enumerate(pool.run([(str(root), sid) for sid in ids]), 1):
        if status != "ok":
            raise RuntimeError(f"{ids[i]}: {status}: {r}")
        res[i] = r
        if k % 2000 == 0:
            print(f"  [{k}/{len(ids)}] {time.time() - t0:.0f}s", flush=True)
    pool.close()
    g = root / "graphs"
    groups = {"points": ["points", "point_kind"], "scalars": ["scalars", "scalar_kind"],
              "planes": ["plane_normal", "plane_point"]}
    for name, keys in groups.items():
        sizes = [len(r[keys[0]]) for r in res]
        np.save(g / f"prim_{name}_off.npy", np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64))
        for key in keys:
            np.save(g / f"prim_{key}.npy", np.concatenate([r[key] for r in res]))
    return {"samples": len(ids), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Extract + pack candidate primitives for a dataset")
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    print(json.dumps(pack_dataset(a.dataset, a.workers)))
