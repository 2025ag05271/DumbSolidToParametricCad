"""Kernel-independent shape similarity metrics (robust where OCC booleans fail on
near-coincident solids, and usable for any STEP — e.g. CATIA rebuilds).

    mc_iou(a, b)       Monte Carlo volume IoU: share of random points (in the joint bounding box)
                       inside both solids vs inside either. Std. error ≈ 0.5 % at n=8192.
    chamfer(a, b)      symmetric Chamfer distance between area-uniform surface samples, divided by
                       the bounding-box diagonal of ``b`` (the reference).
"""

from __future__ import annotations

import cadquery as cq
import numpy as np
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN
from scipy.spatial import cKDTree


def _bounds(*shapes):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for s in shapes:
        bb = s.BoundingBox()
        lo = np.minimum(lo, [bb.xmin, bb.ymin, bb.zmin])
        hi = np.maximum(hi, [bb.xmax, bb.ymax, bb.zmax])
    return lo, hi


def inside(shape: cq.Shape, pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    clf = BRepClass3d_SolidClassifier(shape.wrapped)
    out = np.empty(len(pts), dtype=bool)
    for i, (x, y, z) in enumerate(pts):
        clf.Perform(gp_Pnt(float(x), float(y), float(z)), tol)
        out[i] = clf.State() == TopAbs_IN
    return out


def mc_iou(a: cq.Shape, b: cq.Shape, n: int = 8192, seed: int = 0) -> float:
    lo, hi = _bounds(a, b)
    pts = lo + (hi - lo) * np.random.default_rng(seed).random((n, 3))
    ia, ib = inside(a, pts), inside(b, pts)
    union = (ia | ib).sum()
    return float((ia & ib).sum() / union) if union else 0.0


def surface_samples(shape: cq.Shape, n: int = 4096, seed: int = 0, tol: float = 0.05) -> np.ndarray:
    verts, tris = shape.tessellate(tol, 0.2)
    v = np.array([p.toTuple() for p in verts])
    t = np.array(tris)
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    rng = np.random.default_rng(seed)
    k = rng.choice(len(t), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    return (1 - r1) * p0[k] + r1 * (1 - r2) * p1[k] + r1 * r2 * p2[k]


def chamfer(a: cq.Shape, b: cq.Shape, n: int = 4096, seed: int = 0) -> float:
    pa, pb = surface_samples(a, n, seed), surface_samples(b, n, seed + 1)
    d_ab = cKDTree(pb).query(pa)[0]
    d_ba = cKDTree(pa).query(pb)[0]
    lo, hi = _bounds(b)
    return float((d_ab.mean() + d_ba.mean()) / 2 / np.linalg.norm(hi - lo))
