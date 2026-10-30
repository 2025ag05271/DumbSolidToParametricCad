"""Geometric queries on the in-progress body used to place features sensibly."""

from __future__ import annotations

import cadquery as cq
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from cadseq.build import _profile_face, _through_length, curve_type
from cadseq.schema import Plane, Profile

from .profiles import snap

AXES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def axis_of(v: cq.Vector, tol: float = 1e-7):
    for a in AXES:
        if v.dot(cq.Vector(a)) > 1 - tol:
            return a
    return None


def axis_plane(axis, offset: float) -> Plane:
    """Canonical sketch plane: origin = projection of the world origin, x_dir = X (or Y for ±X)."""
    origin = tuple(snap(c * offset * sum(axis)) if c else 0.0 for c in axis)
    x_dir = (0, 1, 0) if axis[0] else (1, 0, 0)
    return Plane(origin=origin, normal=axis, x_dir=x_dir)


def cq_plane(p: Plane) -> cq.Plane:
    return cq.Plane(origin=p.origin, xDir=p.x_dir, normal=p.normal)


class HostFace:
    """An axis-aligned planar face of the body, with its canonical sketch plane."""

    def __init__(self, face: cq.Face, axis):
        self.face = face
        self.axis = axis
        offset = face.Center().dot(cq.Vector(axis)) * sum(axis)  # signed coordinate along the axis
        self.plane = axis_plane(axis, offset)
        self.cqp = cq_plane(self.plane)
        bb = face.BoundingBox()
        corners = [cq.Vector(x, y, z) for x in (bb.xmin, bb.xmax) for y in (bb.ymin, bb.ymax)
                   for z in (bb.zmin, bb.zmax)]
        loc = [self.cqp.toLocalCoords(c) for c in corners]
        self.u0, self.u1 = min(p.x for p in loc), max(p.x for p in loc)
        self.v0, self.v1 = min(p.y for p in loc), max(p.y for p in loc)

    @property
    def width(self):
        return self.u1 - self.u0

    @property
    def height(self):
        return self.v1 - self.v0

    def world(self, u, v) -> cq.Vector:
        return self.cqp.toWorldCoords((u, v))

    def profile_face(self, prof: Profile) -> cq.Face:
        return _profile_face(prof, self.cqp)


def host_faces(body: cq.Shape, min_area: float = 100.0, min_dim: float = 8.0) -> list[HostFace]:
    out = []
    for f in body.Faces():
        if f.geomType() != "PLANE" or f.Area() < min_area:
            continue
        axis = axis_of(f.normalAt())
        if axis is None:
            continue
        h = HostFace(f, axis)
        if min(h.width, h.height) >= min_dim:
            out.append(h)
    return out


def fits(region: cq.Face, host: cq.Face, margin: float) -> bool:
    """True if ``region`` grown by ``margin`` lies fully inside ``host`` (coplanar faces)."""
    test = region
    if margin > 0:
        wires = region.outerWire().offset2D(margin, "arc")
        if len(wires) != 1:
            return False
        test = cq.Face.makeFromWires(wires[0])
    try:
        common = host.intersect(test)
    except Exception:
        return False
    a = test.Area()
    return abs(common.Area() - a) <= 1e-6 * max(1.0, a)


def thickness(body: cq.Shape, point: cq.Vector, direction: cq.Vector) -> float:
    """Material length along ``direction`` starting at a surface ``point`` (0 if none)."""
    d = direction.normalized()
    start = point - d * 0.01
    probe = cq.Edge.makeLine(start, point + d * _through_length(body))
    try:
        inside = body.intersect(probe)
    except Exception:
        return 0.0
    pv = cq.Vertex.makeVertex(point.x, point.y, point.z)
    for seg in inside.Edges():
        if seg.distance(pv) < 0.05:
            return max(0.0, seg.Length() - 0.01)
    return 0.0


def dressable_edges(body: cq.Shape, min_length: float = 2.0) -> list[cq.Edge]:
    """Line/circle edges shared by two distinct faces (excludes seams)."""
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(body.wrapped, TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, m.Extent() + 1):
        faces = []
        for s in m.FindFromIndex(i):
            if not any(s.IsSame(o) for o in faces):
                faces.append(s)
        if len(faces) != 2:
            continue
        e = cq.Edge(TopoDS.Edge_s(m.FindKey(i)))
        if curve_type(e) in ("line", "circle") and e.Length() >= min_length:
            out.append(e)
    return out
