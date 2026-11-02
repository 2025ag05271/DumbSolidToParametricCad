"""Rigid + scale augmentation of a part's B-Rep graph features (pose invariance).

All vector quantities are transformed consistently: grid points, normals and tangents, and the
analytic face / edge attributes (axes, plane normals, locations, radii, lengths, areas). Applied
to raw (mm) arrays before the dataset's coordinate scaling.
"""

from __future__ import annotations

import numpy as np

AXIAL_TYPES = (1, 2, 4)  # cylinder, cone, torus: axis direction sign is canonicalized


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _canonical(d: np.ndarray) -> np.ndarray:
    idx = np.abs(d).argmax(-1)
    sign = np.sign(np.take_along_axis(d, idx[:, None], 1))
    sign[sign == 0] = 1
    return d * sign


def transform(item: dict, R: np.ndarray, t: np.ndarray, s: float) -> dict:
    """Apply x → s·R·x + t to a raw-mm item (keys node_x, edge_x, optional face_attr, edge_attr)."""
    out = dict(item)
    R = R.astype(np.float32)
    nx = item["node_x"].copy()
    nx[..., :3] = (nx[..., :3] @ R.T) * s + t
    nx[..., 3:6] = nx[..., 3:6] @ R.T
    out["node_x"] = nx
    ex = item["edge_x"].copy()
    ex[..., :3] = (ex[..., :3] @ R.T) * s + t
    ex[..., 3:6] = ex[..., 3:6] @ R.T
    out["edge_x"] = ex
    if "face_attr" in item:
        fa = item["face_attr"].copy()
        kind = fa[:, :6].argmax(1)
        axis = fa[:, 6:9] @ R.T
        loc = (fa[:, 9:12] @ R.T) * s + t
        ax_m = np.isin(kind, AXIAL_TYPES)
        axis[ax_m] = _canonical(axis[ax_m])
        plane = kind == 0  # location = closest point of the plane to the origin
        loc[plane] = axis[plane] * (axis[plane] * loc[plane]).sum(1, keepdims=True)
        cyl = kind == 1  # location = closest axis point to the origin
        loc[cyl] = loc[cyl] - axis[cyl] * (axis[cyl] * loc[cyl]).sum(1, keepdims=True)
        fa[:, 6:9], fa[:, 9:12] = axis, loc
        fa[:, 12:14] *= s
        fa[:, 16] = np.log1p(np.expm1(fa[:, 16]) * s * s)
        out["face_attr"] = fa
    if "edge_attr" in item:
        ea = item["edge_attr"].copy()
        ea[:, 7:9] *= s
        out["edge_attr"] = ea
    return out


def random_pose(item: dict, rng: np.random.Generator, translate: float = 100.0,
                scale: tuple[float, float] = (0.5, 2.0)) -> dict:
    s = float(np.exp(rng.uniform(np.log(scale[0]), np.log(scale[1]))))
    t = rng.uniform(-translate, translate, size=3).astype(np.float32)
    return transform(item, random_rotation(rng), t, s)
