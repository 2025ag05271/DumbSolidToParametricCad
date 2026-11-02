"""Face provenance: which feature of a design history created each face of the final solid.

``TrackedBuilder`` replays a ``CADSequence`` with the *same* OpenCascade operations and settings
as ``cadseq.build.Builder`` (so the solid is identical), but keeps each operation's history
(``Modified`` / ``IsDeleted`` / ``Generated``) and propagates a label to every face:

    body, labels = label_faces(seq)          # labels[i] ↔ i-th face of body (TopExp order)
    labels[i] = FaceLabel(feature=7, cls=FACE_CLASSES.index("hole_counterbore"), role=..., mirrored=0)

Label rules
- Faces of a pad / pocket tool: ``*_end`` if planar and perpendicular to the sketch normal, else
  ``*_side``. Hole tool faces are classed by hole kind, with a geometric role (wall, bottom, ...).
- Fillet / chamfer: every face the operation creates.
- Mirror: images inherit the label of their source face with ``mirrored = 1``.
- A face that keeps existing (possibly split or trimmed) keeps its label. When faces with different
  labels merge (coplanar faces unified by ``clean``), the label of the largest source face wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import (
    TopTools_DataMapOfShapeInteger,
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListOfShape,
)

from .build import Builder, BuildError, EDGE_MATCH_TOL, _extrude, _hole_tool, _through_length
from .schema import CADSequence, Chamfer, Fillet, Hole, Mirror, Pad, Pocket, Sketch

FACE_CLASSES = ["extrude_side", "extrude_end", "cut_side", "cut_end",
                "hole_simple", "hole_counterbore", "hole_countersink", "fillet", "chamfer"]
FACE_ROLES = ["side", "end", "wall", "bottom", "cb_wall", "cb_floor", "cs_cone", "blend"]
_CLS = {c: i for i, c in enumerate(FACE_CLASSES)}
_ROLE = {r: i for i, r in enumerate(FACE_ROLES)}
_HOLE_CLS = {"simple": "hole_simple", "counterbore": "hole_counterbore", "countersink": "hole_countersink"}


@dataclass(frozen=True)
class FaceLabel:
    feature: int  # index into seq.features of the solid feature that created the face
    cls: int      # index into FACE_CLASSES
    role: int     # index into FACE_ROLES
    mirrored: int = 0  # 1 if the face is the image of a Mirror feature

    @property
    def cls_name(self) -> str:
        return FACE_CLASSES[self.cls]

    @property
    def role_name(self) -> str:
        return FACE_ROLES[self.role]


def faces_of(shape) -> list:
    exp = TopExp_Explorer(shape.wrapped if hasattr(shape, "wrapped") else shape, TopAbs_FACE)
    out = []
    while exp.More():
        out.append(TopoDS.Face_s(exp.Current()))
        exp.Next()
    return out


def _area(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


class _LabelMap:
    """Face → FaceLabel with IsSame semantics; conflicts resolved by the larger source face."""

    def __init__(self):
        self._map = TopTools_DataMapOfShapeInteger()
        self._items: list[list] = []  # [label, source face, source area | None]

    def put(self, face, label: FaceLabel, source) -> None:
        if not self._map.IsBound(face):
            self._map.Bind(face, len(self._items))
            self._items.append([label, source, None])
            return
        item = self._items[self._map.Find(face)]
        if item[0] == label:
            return
        if item[2] is None:
            item[2] = _area(item[1])
        a = _area(source)
        if a > item[2]:
            item[0], item[1], item[2] = label, source, a

    def get(self, face) -> FaceLabel | None:
        return self._items[self._map.Find(face)][0] if self._map.IsBound(face) else None


def _propagate(sources, modified, removed, out: _LabelMap) -> None:
    """Carry (face, label) pairs through an operation's history."""
    for face, label in sources:
        if removed(face):
            continue
        mods = modified(face)
        if mods.Size() == 0:
            out.put(face, label, face)  # unchanged: the face itself is in the result
        else:
            for m in mods:
                out.put(m, label, face)


def _unit(v):
    n = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
    return cq.Vector(v.x / n, v.y / n, v.z / n)


def _extrude_tool_labels(tool, feature: int, cls_prefix: str, normal: cq.Vector):
    out = []
    for face in faces_of(tool):
        f = cq.Face(face)
        end = f.geomType() == "PLANE" and abs(_unit(f.normalAt()).dot(_unit(normal))) > 1 - 1e-6
        role = "end" if end else "side"
        out.append((face, FaceLabel(feature, _CLS[f"{cls_prefix}_{role}"], _ROLE[role])))
    return out


def _hole_tool_labels(tool, h: Hole, feature: int):
    cls = _CLS[_HOLE_CLS[h.kind]]
    axis_p, axis_d = cq.Vector(h.position), _unit(cq.Vector(h.direction))
    out = []
    for face in faces_of(tool):
        f = cq.Face(face)
        t = f.geomType()
        if t == "CONE":
            role = "cs_cone"
        elif t == "CYLINDER":
            r = f._geomAdaptor().Cylinder().Radius()
            role = "cb_wall" if h.kind == "counterbore" and abs(r - h.cb_diameter / 2) < 1e-6 else "wall"
        else:  # planes perpendicular to the axis
            depth = (f.Center() - axis_p).dot(axis_d)
            role = "cb_floor" if h.kind == "counterbore" and abs(depth - h.cb_depth) < 1e-6 else "bottom"
        out.append((face, FaceLabel(feature, cls, _ROLE[role])))
    return out


class TrackedBuilder(Builder):
    """``Builder`` that also tracks a FaceLabel for every face of the current body."""

    def __init__(self, edge_tol: float = EDGE_MATCH_TOL):
        super().__init__(edge_tol=edge_tol)
        self.labels = _LabelMap()
        self._k = -1

    def apply_indexed(self, f, index: int) -> None:
        self._k = index
        self.apply(f)

    def body_labels(self) -> list[FaceLabel]:
        return [self.labels.get(face) for face in faces_of(self.body)]

    # -- helpers -------------------------------------------------------------
    def _body_sources(self):
        return [(face, self.labels.get(face)) for face in faces_of(self.body)]

    def _finish(self, shape: cq.Shape, lm: _LabelMap) -> None:
        """Same as ``build._as_solid`` (clean + single solid), carrying labels through ``clean``."""
        up = ShapeUpgrade_UnifySameDomain(shape.wrapped, True, True, True)
        up.AllowInternalEdges(False)
        up.Build()
        hist = up.History()
        cleaned = cq.Shape.cast(up.Shape())
        solids = cleaned.Solids()
        if len(solids) != 1:
            raise BuildError(f"operation produced {len(solids)} solids (expected 1)")
        src = [(face, lm.get(face)) for face in faces_of(shape) if lm.get(face) is not None]
        out = _LabelMap()
        _propagate(src, hist.Modified, hist.IsRemoved, out)
        self.body, self.labels = solids[0], out

    def _bop(self, op, tool: cq.Shape, tool_labels) -> None:
        args, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
        args.Append(self.body.wrapped)
        tools.Append(tool.wrapped)
        op.SetArguments(args)
        op.SetTools(tools)
        op.SetRunParallel(True)  # as cadquery.Shape._bool_op
        op.Build()
        result = cq.Shape.cast(op.Shape())
        lm = _LabelMap()
        _propagate(self._body_sources(), op.Modified, op.IsDeleted, lm)
        _propagate(tool_labels, op.Modified, op.IsDeleted, lm)
        self._finish(result, lm)

    def _dressup(self, mk, cls: str) -> None:
        mk.Build()
        result = cq.Shape.cast(mk.Shape())
        lm = _LabelMap()
        _propagate(self._body_sources(), mk.Modified, mk.IsDeleted, lm)
        new = FaceLabel(self._k, _CLS[cls], _ROLE["blend"])
        for face in faces_of(result):
            if lm.get(face) is None:
                lm.put(face, new, face)
        self._finish(result, lm)

    # -- operations (mirror Builder._apply exactly) ----------------------------
    def _apply(self, f) -> None:
        body, k = self.body, self._k
        if isinstance(f, Sketch):
            return super()._apply(f)
        if body is None and not isinstance(f, Pad):
            raise BuildError("no body yet")
        if isinstance(f, Pad):
            faces, plane = self.sketches[f.sketch]
            tool = _extrude(faces, plane, f.length, f.direction, f.symmetric)
            tl = _extrude_tool_labels(tool, k, "extrude", plane.zDir)
            if body is None:
                lm = _LabelMap()
                for face, lab in tl:
                    lm.put(face, lab, face)
                self._finish(tool, lm)
            else:
                self._bop(BRepAlgoAPI_Fuse(), tool, tl)
        elif isinstance(f, Pocket):
            faces, plane = self.sketches[f.sketch]
            length = _through_length(body) if f.through_all else f.depth
            tool = _extrude(faces, plane, length, f.direction, False)
            self._bop(BRepAlgoAPI_Cut(), tool, _extrude_tool_labels(tool, k, "cut", plane.zDir))
        elif isinstance(f, Hole):
            tool = _hole_tool(f, body)
            self._bop(BRepAlgoAPI_Cut(), tool, _hole_tool_labels(tool, f, k))
        elif isinstance(f, Fillet):
            mk = BRepFilletAPI_MakeFillet(body.wrapped)
            for e in self._edges(f.edges):
                mk.Add(f.radius, e.wrapped)
            self._dressup(mk, "fillet")
        elif isinstance(f, Chamfer):
            emap = TopTools_IndexedDataMapOfShapeListOfShape()
            TopExp.MapShapesAndAncestors_s(body.wrapped, TopAbs_EDGE, TopAbs_FACE, emap)
            mk = BRepFilletAPI_MakeChamfer(body.wrapped)
            for e in self._edges(f.edges):  # as cadquery Mixin3D.chamfer (symmetric)
                face = emap.FindFromKey(e.wrapped).First()
                mk.Add(f.distance, f.distance, e.wrapped, TopoDS.Face_s(face))
            self._dressup(mk, "chamfer")
        elif isinstance(f, Mirror):
            trsf = gp_Trsf()
            trsf.SetMirror(gp_Ax2(gp_Pnt(*f.plane.origin), gp_Dir(*f.plane.normal)))
            tr = BRepBuilderAPI_Transform(body.wrapped, trsf, True)
            image = cq.Shape.cast(tr.Shape())
            tl = []
            for face, lab in self._body_sources():
                img = TopoDS.Face_s(tr.ModifiedShape(face))
                tl.append((img, FaceLabel(lab.feature, lab.cls, lab.role, 1)))
            self._bop(BRepAlgoAPI_Fuse(), image, tl)


def label_faces(seq: CADSequence, edge_tol: float = EDGE_MATCH_TOL) -> tuple[cq.Shape, list[FaceLabel]]:
    """Replay ``seq``; return the final solid and one FaceLabel per face (TopExp order)."""
    b = TrackedBuilder(edge_tol=edge_tol)
    for i, f in enumerate(seq.features):
        b.apply_indexed(f, i)
    if b.body is None or not b.body.isValid():
        raise BuildError("final body is missing or invalid")
    labels = b.body_labels()
    missing = sum(lab is None for lab in labels)
    if missing:
        raise BuildError(f"{missing} face(s) without provenance")
    return b.body, labels
