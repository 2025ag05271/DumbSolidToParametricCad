"""Rebuild a solid from a CADSequence with CadQuery/OpenCascade.

Usage:
    python -m cadseq.build examples/bracket.json --step out/bracket.step
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cadquery as cq

from .schema import (
    Arc,
    CADSequence,
    Chamfer,
    Circle,
    EdgeRef,
    Fillet,
    Hole,
    Line,
    Mirror,
    Pad,
    Plane,
    Pocket,
    Profile,
    Sketch,
    load_sequence,
)

EDGE_MATCH_TOL = 1e-3
QUANTIZED_EDGE_TOL = 0.3  # tolerance for quantized (0.05 mm) / predicted edge-ref points
TOOL_OVERSHOOT = 0.01


class BuildError(RuntimeError):
    pass


# ------------------------------------------------------------ sketch → faces --

def _cq_plane(p: Plane) -> cq.Plane:
    return cq.Plane(origin=p.origin, xDir=p.x_dir, normal=p.normal)


def _loop_to_wire(loop, plane: cq.Plane) -> cq.Wire:
    w = lambda uv: plane.toWorldCoords(tuple(uv))  # noqa: E731
    if isinstance(loop[0], Circle):
        c = loop[0]
        return cq.Wire.makeCircle(c.radius, w(c.center), plane.zDir)
    edges = []
    for c in loop:
        if isinstance(c, Line):
            edges.append(cq.Edge.makeLine(w(c.start), w(c.end)))
        elif isinstance(c, Arc):
            edges.append(cq.Edge.makeThreePointArc(w(c.start), w(c.mid), w(c.end)))
    return cq.Wire.assembleEdges(edges)


def _profile_face(profile: Profile, plane: cq.Plane) -> cq.Face:
    outer = _loop_to_wire(profile.outer, plane)
    inner = [_loop_to_wire(l, plane) for l in profile.inner]
    face = cq.Face.makeFromWires(outer, inner)
    if not face.isValid():
        raise BuildError("invalid sketch profile face")
    return face


# ----------------------------------------------------------- solid helpers --

def _as_solid(shape: cq.Shape) -> cq.Shape:
    shape = shape.clean()
    solids = shape.Solids()
    if len(solids) != 1:
        raise BuildError(f"operation produced {len(solids)} solids (expected 1)")
    return solids[0]


def _extrude(faces, plane: cq.Plane, length: float, direction: int, symmetric: bool) -> cq.Shape:
    n = plane.zDir
    tools = []
    for face in faces:
        if symmetric:
            face = face.translate(n * (-length / 2))
            vec = n * length
        else:
            vec = n * (length * direction)
        tools.append(cq.Solid.extrudeLinear(face, vec))
    tool = tools[0]
    for t in tools[1:]:
        tool = tool.fuse(t)
    return tool


def _through_length(body: cq.Shape) -> float:
    return 2.0 * body.BoundingBox().DiagonalLength + 1.0


def _hole_tool(h: Hole, body: cq.Shape) -> cq.Shape:
    d = cq.Vector(h.direction).normalized()
    p = cq.Vector(h.position)
    start = p - d * TOOL_OVERSHOOT
    depth = _through_length(body) if h.through_all else h.depth
    tool = cq.Solid.makeCylinder(h.diameter / 2, depth + TOOL_OVERSHOOT, start, d)
    if h.kind == "counterbore":
        cb = cq.Solid.makeCylinder(h.cb_diameter / 2, h.cb_depth + TOOL_OVERSHOOT, start, d)
        tool = tool.fuse(cb)
    elif h.kind == "countersink":
        r = h.cs_diameter / 2
        cone_h = r / math.tan(math.radians(h.cs_angle / 2))
        # Cone tip lies on the axis; extend its base slightly above the entry face.
        cone_start = p - d * (TOOL_OVERSHOOT)
        r_top = r + TOOL_OVERSHOOT * math.tan(math.radians(h.cs_angle / 2))
        cone = cq.Solid.makeCone(r_top, 0.0, cone_h + TOOL_OVERSHOOT, cone_start, d)
        tool = tool.fuse(cone)
    return tool


def curve_type(edge: cq.Edge) -> str:
    t = edge.geomType().lower()
    return t if t in {"line", "circle", "ellipse", "bspline"} else "other"


def edge_ref(edge: cq.Edge) -> EdgeRef:
    """Build an EdgeRef from a concrete edge (used by the dataset generator)."""
    p = edge.positionAt(0.5)
    return EdgeRef(point=(p.x, p.y, p.z), curve=curve_type(edge))


def resolve_edge(body: cq.Shape, ref: EdgeRef, tol: float = EDGE_MATCH_TOL) -> cq.Edge:
    probe = cq.Vertex.makeVertex(*ref.point)
    best, best_d = None, math.inf
    for e in body.Edges():
        if curve_type(e) != ref.curve:
            continue
        d = e.distance(probe)
        if d < best_d:
            best, best_d = e, d
    if best is None or best_d > tol:
        raise BuildError(f"no {ref.curve} edge within {tol} of {ref.point} (closest {best_d:.4g})")
    return best


def _resolve_edges(body: cq.Shape, refs: list[EdgeRef], tol: float = EDGE_MATCH_TOL) -> list[cq.Edge]:
    edges = [resolve_edge(body, r, tol) for r in refs]
    if len({e.hashCode() for e in edges}) != len(edges):
        raise BuildError("multiple edge refs resolved to the same edge")
    return edges


def _resolve_edges_lenient(body: cq.Shape, refs: list[EdgeRef], tol: float) -> list[cq.Edge]:
    """Nearest matching edge per ref within ``tol``; unresolved refs and duplicates are dropped."""
    out, seen = [], set()
    for r in refs:
        try:
            e = resolve_edge(body, r, tol)
        except BuildError:
            continue
        if e.hashCode() not in seen:
            seen.add(e.hashCode())
            out.append(e)
    if not out:
        raise BuildError("no edge refs could be resolved")
    return out


# ------------------------------------------------------------------ build --

class Builder:
    """Applies features one at a time; supports snapshot/restore for trial edits.

    With ``check_each=True`` every intermediate body must be a valid single solid
    (used by the dataset generator so failures are caught at the offending feature).
    ``edge_tol`` is the max distance between an EdgeRef point and the matched edge; raise it
    for quantized / predicted sequences (see ``QUANTIZED_EDGE_TOL``). With ``lenient_edges=True``
    unresolvable / duplicate edge refs are dropped instead of failing the feature.
    """

    def __init__(self, check_each: bool = False, edge_tol: float = EDGE_MATCH_TOL,
                 lenient_edges: bool = False):
        self.sketches: dict[str, tuple[list[cq.Face], cq.Plane]] = {}
        self.body: cq.Shape | None = None
        self.check_each = check_each
        self.edge_tol = edge_tol
        self.lenient_edges = lenient_edges

    def _edges(self, refs):
        if self.lenient_edges:
            return _resolve_edges_lenient(self.body, refs, self.edge_tol)
        return _resolve_edges(self.body, refs, self.edge_tol)

    def snapshot(self):
        return dict(self.sketches), self.body

    def restore(self, snap) -> None:
        self.sketches, self.body = dict(snap[0]), snap[1]

    def _set_body(self, shape: cq.Shape) -> None:
        body = _as_solid(shape)
        if self.check_each and not body.isValid():
            raise BuildError("result is not a valid solid")
        self.body = body

    def apply(self, f) -> None:
        try:
            self._apply(f)
        except BuildError as e:
            raise BuildError(f"{f.id} ({f.type}): {e}") from e
        except Exception as e:  # OCC kernel failures surface as generic exceptions
            raise BuildError(f"{f.id} ({f.type}): kernel error: {e}") from e

    def _apply(self, f) -> None:
        body = self.body
        if isinstance(f, Sketch):
            plane = _cq_plane(f.plane)
            self.sketches[f.id] = ([_profile_face(p, plane) for p in f.profiles], plane)
            return
        if body is None and not isinstance(f, Pad):
            raise BuildError("no body yet")
        if isinstance(f, Pad):
            faces, plane = self.sketches[f.sketch]
            tool = _extrude(faces, plane, f.length, f.direction, f.symmetric)
            self._set_body(tool if body is None else body.fuse(tool))
        elif isinstance(f, Pocket):
            faces, plane = self.sketches[f.sketch]
            length = _through_length(body) if f.through_all else f.depth
            tool = _extrude(faces, plane, length, f.direction, False)
            self._set_body(body.cut(tool))
        elif isinstance(f, Hole):
            self._set_body(body.cut(_hole_tool(f, body)))
        elif isinstance(f, Fillet):
            self._set_body(body.fillet(f.radius, self._edges(f.edges)))
        elif isinstance(f, Chamfer):
            self._set_body(body.chamfer(f.distance, None, self._edges(f.edges)))
        elif isinstance(f, Mirror):
            mirrored = body.mirror(cq.Vector(f.plane.normal), cq.Vector(f.plane.origin))
            self._set_body(body.fuse(mirrored))


def build(seq: CADSequence, edge_tol: float = EDGE_MATCH_TOL) -> cq.Shape:
    b = Builder(edge_tol=edge_tol)
    for f in seq.features:
        b.apply(f)
    if b.body is None or not b.body.isValid():
        raise BuildError("final body is missing or invalid")
    return b.body


LENIENT_SNAP = 0.03  # edge-snap tolerance as a fraction of the body's bbox diagonal


def build_lenient(seq: CADSequence, snap: float = LENIENT_SNAP) -> tuple[cq.Shape, list[str]]:
    """Best-effort rebuild for model predictions.

    Edge refs snap to the nearest matching edge within ``snap`` × bbox diagonal (duplicates and
    unresolvable refs dropped); any feature that still fails is skipped (a skipped sketch also skips
    its pad/pocket). Returns the body and the ids of skipped features.
    """
    b = Builder(lenient_edges=True)
    skipped: list[str] = []
    for f in seq.features:
        if b.body is not None:
            b.edge_tol = max(QUANTIZED_EDGE_TOL, snap * b.body.BoundingBox().DiagonalLength)
        try:
            b.apply(f)
        except BuildError:
            skipped.append(f.id)
    if b.body is None or not b.body.isValid():
        raise BuildError("no valid body could be built")
    return b.body, skipped


def summarize(body: cq.Shape) -> dict:
    bb = body.BoundingBox()
    return {
        "valid": body.isValid(),
        "faces": len(body.Faces()),
        "edges": len(body.Edges()),
        "vertices": len(body.Vertices()),
        "volume": round(body.Volume(), 3),
        "bbox": [round(v, 3) for v in (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", type=Path)
    ap.add_argument("--step", type=Path, help="write the rebuilt solid to this STEP file")
    args = ap.parse_args()

    body = build(load_sequence(args.json))
    print(summarize(body))
    if args.step:
        args.step.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(cq.Workplane().add(body), str(args.step), cq.exporters.ExportTypes.STEP)
        print(f"wrote {args.step}")


if __name__ == "__main__":
    main()
