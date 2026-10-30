"""Render a contact sheet of generated samples.

Usage:
    python -m datagen.preview <dataset_dir> --n 24 --out preview.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def _render(ax, shape: cq.Shape, title: str):
    verts, tris = shape.tessellate(0.2, 0.3)
    v = np.array([p.toTuple() for p in verts])
    polys = v[np.array(tris)]
    n = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.65 * np.clip(n @ np.array([0.4, -0.5, 0.75]), 0, 1)
    ax.add_collection3d(Poly3DCollection(polys, facecolors=plt.cm.Blues(0.25 + 0.6 * shade), linewidths=0))
    lo, hi = v.min(0), v.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=-55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("preview.png"))
    args = ap.parse_args()

    index = [json.loads(l) for l in (args.dataset / "index.jsonl").read_text().splitlines() if l.strip()]
    index = [m for m in index if not m.get("failed")]
    picks = random.Random(args.seed).sample(index, min(args.n, len(index)))
    rows = (len(picks) + args.cols - 1) // args.cols
    fig = plt.figure(figsize=(args.cols * 2.6, rows * 2.6), dpi=110)
    for i, m in enumerate(picks):
        ax = fig.add_subplot(rows, args.cols, i + 1, projection="3d")
        step = args.dataset / "step" / (m["id"][:-3] or "0") / f"{m['id']}.step"
        shape = cq.importers.importStep(str(step)).val()
        _render(ax, shape, f"{m['id']} {m['family']}\n{m['n_features']} feats, {m['faces']} faces")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
