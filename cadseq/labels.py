"""Per-face provenance labels + exact analytic attributes for a generated dataset.

    python -m cadseq.labels data/synth_v2 --workers 16

For every sample the stored history is replayed with ``cadseq.provenance.TrackedBuilder`` (same
solid as the generator built) and each face of the *input* solid (the STEP the model sees, in graph
order) gets the label of the feature that created it. STEP export / import can reorder faces, so
faces are matched geometrically: sample points of stored face i must lie on the replayed face.

Writes ``labels/<shard>/<id>.npz`` (per sample) and packs, in ``tokens.npz`` order:
    graphs/face_cls.npy      (ΣF,) int8   FACE_CLASSES index (semantic segmentation target)
    graphs/face_role.npy     (ΣF,) int8   FACE_ROLES index (wall, bottom, cb_floor, ...)
    graphs/face_instance.npy (ΣF,) int16  feature instance id within the part (instance segmentation)
    graphs/face_feature.npy  (ΣF,) int16  index of the creating feature in the history (pointer target)
    graphs/face_mirror.npy   (ΣF,) int8   1 if the face is a mirror image
    graphs/face_attr.npy     (ΣF, FACE_ATTR_DIM) float32, graphs/edge_attr.npy (ΣE, EDGE_ATTR_DIM)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .prepare import sample_path

LABEL_KEYS = ["face_cls", "face_role", "face_instance", "face_feature", "face_mirror"]
LABEL_DTYPES = {"face_cls": np.int8, "face_role": np.int8, "face_instance": np.int16,
                "face_feature": np.int16, "face_mirror": np.int8}
MATCH_TOL = 1e-3


def _dist(shape_a, shape_b) -> float:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    d = BRepExtrema_DistShapeShape(shape_a, shape_b)
    return d.Value() if d.IsDone() else float("inf")


def _vertices(points):
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    comp, bb = TopoDS_Compound(), BRep_Builder()
    bb.MakeCompound(comp)
    for p in points:
        bb.Add(comp, BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex())
    return comp


def match_faces(node_x: np.ndarray, faces: list) -> tuple[list[int], int]:
    """perm[i] = index of the replayed face matching stored face i. Returns (perm, fallbacks)."""
    from scipy.optimize import linear_sum_assignment

    n = len(faces)
    on_face = []  # up to 3 points inside each stored face (None if its grid has no inside sample)
    for g in node_x:
        m = g[..., 6] > 0.5
        if m.any():
            pts = g[..., :3][m]
            on_face.append(pts[np.linspace(0, len(pts) - 1, 3).astype(int)])
        else:
            on_face.append(None)

    def lies_on(i, j):
        return on_face[i] is not None and all(_dist(_vertices([p]), faces[j]) <= MATCH_TOL for p in on_face[i])

    perm, used = [-1] * n, set()
    for i in range(n):  # most faces keep their index
        if lies_on(i, i):
            perm[i] = i
            used.add(i)
    for i in range(n):
        if perm[i] < 0 and on_face[i] is not None:
            for j in range(n):
                if j not in used and lies_on(i, j):
                    perm[i] = j
                    used.add(j)
                    break
    rest_i = [i for i in range(n) if perm[i] < 0]
    rest_j = [j for j in range(n) if j not in used]
    if rest_i:  # faces without inside samples: closest grid → face distance, optimal assignment
        cost = np.array([[_dist(_vertices(node_x[i][..., :3].reshape(-1, 3)), faces[j]) for j in rest_j]
                         for i in rest_i])
        ri, cj = linear_sum_assignment(cost)
        for a, b in zip(ri, cj):
            perm[rest_i[a]] = rest_j[b]
    return perm, len(rest_i)


def _degenerate_to_simple(labels: list, j: int):
    """A counterbore whose head is deeper than the wall leaves no hole wall: the remaining
    geometry is a plain hole, so label it as one (cb_wall → wall, cb_floor → bottom)."""
    from cadseq.provenance import FACE_CLASSES, FACE_ROLES, FaceLabel

    lab = labels[j]
    if FACE_CLASSES[lab.cls] != "hole_counterbore":
        return lab
    same = [l for l in labels if l.feature == lab.feature and l.mirrored == lab.mirrored]
    if any(FACE_ROLES[l.role] == "wall" for l in same):
        return lab
    role = {"cb_wall": "wall", "cb_floor": "bottom"}.get(FACE_ROLES[lab.role], FACE_ROLES[lab.role])
    return FaceLabel(lab.feature, FACE_CLASSES.index("hole_simple"), FACE_ROLES.index(role), lab.mirrored)


def label_sample(task) -> dict:
    """Worker entry: ``task = (dataset root, sample id)``; writes labels/<shard>/<id>.npz."""
    import warnings

    warnings.filterwarnings("ignore")
    import cadquery as cq

    from cadseq.build import QUANTIZED_EDGE_TOL
    from cadseq.provenance import faces_of, label_faces
    from cadseq.schema import load_sequence
    from step_to_graph import brep_to_graph_arrays

    root, sid = Path(task[0]), task[1]
    body, labels = label_faces(load_sequence(sample_path(root, "json", sid)), edge_tol=QUANTIZED_EDGE_TOL)
    shape = cq.importers.importStep(str(sample_path(root, "step", sid))).val()
    with np.load(sample_path(root, "graph", sid)) as z:
        stored = {k: z[k] for k in z.files}
    arrays = brep_to_graph_arrays(shape.wrapped, stored["node_x"].shape[1], with_attributes=True)
    if not (np.array_equal(arrays["edge_index"], stored["edge_index"])
            and np.allclose(arrays["node_x"], stored["node_x"], atol=1e-4)):
        raise RuntimeError("recomputed graph differs from the stored graph")
    faces = faces_of(body)
    if len(faces) != len(stored["node_x"]):
        raise RuntimeError(f"face count {len(faces)} vs stored {len(stored['node_x'])}")
    perm, fallbacks = match_faces(stored["node_x"], faces)
    lab = [_degenerate_to_simple(labels, j) for j in perm]
    inst: dict = {}
    out = {
        "face_cls": [l.cls for l in lab],
        "face_role": [l.role for l in lab],
        "face_instance": [inst.setdefault((l.feature, l.mirrored), len(inst)) for l in lab],
        "face_feature": [l.feature for l in lab],
        "face_mirror": [l.mirrored for l in lab],
    }
    out = {k: np.asarray(v, dtype=LABEL_DTYPES[k]) for k, v in out.items()}
    path = sample_path(root, "graph", sid)
    path = root / "labels" / path.parent.name / f"{sid}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out, face_attr=arrays["face_attr"], edge_attr=arrays["edge_attr"])
    return {"id": sid, "faces": len(lab), "fallbacks": fallbacks,
            "identity": sum(i == j for i, j in enumerate(perm)), "classes": np.bincount(out["face_cls"]).tolist()}


def pack_labels(root: Path) -> dict:
    """Concatenate per-sample label files into graphs/face_*.npy in tokens.npz order."""
    from step_to_graph import EDGE_ATTR_DIM, FACE_ATTR_DIM

    ids = [str(s) for s in np.load(root / "tokens.npz")["ids"]]
    node_off = np.load(root / "graphs" / "node_off.npy")
    edge_off = np.load(root / "graphs" / "edge_off.npy")
    out = {k: np.zeros(node_off[-1], LABEL_DTYPES[k]) for k in LABEL_KEYS}
    face_attr = np.lib.format.open_memmap(root / "graphs" / "face_attr.npy", "w+", np.float32,
                                          (int(node_off[-1]), FACE_ATTR_DIM))
    edge_attr = np.lib.format.open_memmap(root / "graphs" / "edge_attr.npy", "w+", np.float32,
                                          (int(edge_off[-1]), EDGE_ATTR_DIM))
    for k, sid in enumerate(ids):
        p = root / "labels" / sample_path(root, "graph", sid).parent.name / f"{sid}.npz"
        with np.load(p) as z:
            a, b, c, d = node_off[k], node_off[k + 1], edge_off[k], edge_off[k + 1]
            if len(z["face_cls"]) != b - a or len(z["edge_attr"]) != d - c:
                raise RuntimeError(f"{sid}: label sizes do not match the packed graph")
            for key in LABEL_KEYS:
                out[key][a:b] = z[key]
            face_attr[a:b] = z["face_attr"]
            edge_attr[c:d] = z["edge_attr"]
    face_attr.flush()
    edge_attr.flush()
    for key, arr in out.items():
        np.save(root / "graphs" / f"{key}.npy", arr)
    return {"faces": int(node_off[-1]), "class_counts": np.bincount(out["face_cls"]).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from cadseq.provenance import FACE_CLASSES
    from cadseq.workers import SupervisedPool
    from s2pnet.train import keep_awake

    keep_awake()
    root = args.dataset.resolve()
    ids = [str(s) for s in np.load(root / "tokens.npz")["ids"]][: args.limit]
    todo = [sid for sid in ids
            if not (root / "labels" / sample_path(root, "graph", sid).parent.name / f"{sid}.npz").exists()]
    print(f"{len(ids) - len(todo)} already labelled, labelling {len(todo)} with {args.workers} workers", flush=True)

    t0, stats, failed = time.time(), Counter(), []
    pool = SupervisedPool(label_sample, args.workers, args.timeout)
    for k, (i, status, res) in enumerate(pool.run([(str(root), sid) for sid in todo]), 1):
        if status == "ok":
            stats["faces"] += res["faces"]
            stats["identity"] += res["identity"]
            stats["fallbacks"] += res["fallbacks"]
        else:
            failed.append((todo[i], f"{status}: {res}"))
        if k % 500 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] failed={len(failed)} {k / el:.1f}/s eta {(len(todo) - k) * el / k / 60:.1f} min",
                  flush=True)
    pool.close()
    if failed:
        print(f"{len(failed)} samples failed, e.g. {failed[:3]}")
        (root / "labels_failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
        raise SystemExit("labelling incomplete; fix failures before packing")
    if args.limit is not None:
        print(f"labelled {len(todo)} samples (--limit: packing skipped)")
        return

    packed = pack_labels(root)
    summary = {
        "samples": len(ids), "faces": packed["faces"],
        "faces_kept_index": stats["identity"], "faces_matched_by_fallback": stats["fallbacks"],
        "class_counts": dict(zip(FACE_CLASSES, packed["class_counts"])),
        "seconds": round(time.time() - t0, 1),
    }
    (root / "labels_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
