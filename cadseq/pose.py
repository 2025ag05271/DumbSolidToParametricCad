"""Rigid pose transforms of a design history (for pose augmentation and placement handling).

    R = SIGNED_PERMUTATIONS[k]                 # one of the 48 axis-aligned rotations / reflections
    seq2 = transform_sequence(seq, R, t)       # history of the transformed solid x -> R x + t

Only transforms that keep sketch planes axis-aligned are supported (signed permutations), so the
transformed history is again in the tokenizer's canonical form. With ``t`` on the 0.05 mm grid all
values stay exact. Sketches are re-expressed in the canonical frame of their new plane (origin on the
axis, canonical x_dir), so 2-D coordinates are recomputed rather than copied.
"""

from __future__ import annotations

import itertools

import numpy as np

from .schema import CADSequence
from .tokenizer import Q_PER_MM, TokenizeError, canonical_x_dir, decode, encode

SIGNED_PERMUTATIONS = [
    np.array([[s[i] if j == p[i] else 0 for j in range(3)] for i in range(3)], dtype=float)
    for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)
]


def _snap(x) -> float:
    return round(round(float(x) * Q_PER_MM) / Q_PER_MM, 6) + 0.0


def _snap3(v) -> tuple:
    return tuple(_snap(c) for c in v)


def _frame(normal, x_dir):
    n, x = np.asarray(normal, float), np.asarray(x_dir, float)
    return n, x, np.cross(n, x)


def transform_sequence(seq: CADSequence, R: np.ndarray, t, canonical: bool = True) -> CADSequence:
    """History whose solid is the input solid transformed by x -> R x + t (R a signed permutation).

    With ``canonical`` the result is re-tokenized into canonical form, which raises TokenizeError if
    a value leaves the tokenizer range; ``canonical=False`` returns the (buildable) history as is."""
    t = np.asarray(t, dtype=float)
    out = []
    for f in seq.model_dump()["features"]:
        ty = f["type"]
        if ty == "sketch":
            n, x, y = _frame(f["plane"]["normal"], f["plane"]["x_dir"])
            o = np.asarray(f["plane"]["origin"], float)
            n2 = R @ n
            x2 = np.asarray(canonical_x_dir(tuple(int(round(c)) for c in n2)), float)
            y2 = np.cross(n2, x2)
            o2 = n2 * float(n2 @ (R @ o + t))

            def to2(p, o=o, x=x, y=y, x2=x2, y2=y2, o2=o2):
                w = R @ (o + p[0] * x + p[1] * y) + t
                return (_snap((w - o2) @ x2), _snap((w - o2) @ y2))

            for prof in f["profiles"]:
                for loop in [prof["outer"]] + prof["inner"]:
                    for c in loop:
                        for key in ("start", "end", "mid", "center"):
                            if key in c:
                                c[key] = to2(c[key])
            f["plane"] = {"origin": _snap3(o2), "normal": tuple(float(c) + 0.0 for c in n2),
                          "x_dir": tuple(float(c) + 0.0 for c in x2)}
        elif ty == "hole":
            f["position"] = _snap3(R @ np.asarray(f["position"], float) + t)
            f["direction"] = tuple(float(c) + 0.0 for c in R @ np.asarray(f["direction"], float))
        elif ty in ("fillet", "chamfer"):
            for e in f["edges"]:
                e["point"] = _snap3(R @ np.asarray(e["point"], float) + t)
        elif ty == "mirror":
            n2 = R @ np.asarray(f["plane"]["normal"], float)
            n2 = n2 * np.sign(n2.sum())  # tokenizer: positive axis
            o2 = R @ np.asarray(f["plane"]["origin"], float) + t
            f["plane"] = {"origin": _snap3(n2 * float(n2 @ o2)), "normal": tuple(float(c) + 0.0 for c in n2)}
        out.append(f)
    seq2 = CADSequence.model_validate({"features": out})
    return decode(encode(seq2)) if canonical else seq2


def random_pose(rng: np.random.Generator, translate: float = 100.0):
    """Random signed permutation and grid-aligned translation in [-translate, translate]^3."""
    R = SIGNED_PERMUTATIONS[rng.integers(len(SIGNED_PERMUTATIONS))]
    t = np.round(rng.uniform(-translate, translate, size=3) * Q_PER_MM) / Q_PER_MM
    return R, t


def try_transform(seq: CADSequence, R, t):
    try:
        return transform_sequence(seq, R, t)
    except (TokenizeError, ValueError):
        return None
