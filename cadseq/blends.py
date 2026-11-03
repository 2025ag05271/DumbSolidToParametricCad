"""Sharp edges under fillet / chamfer faces ("virtual edges").

A fillet or chamfer replaces the edge it was applied to, so that edge — which the design history
must reference — no longer exists in the final (dumb) solid. It is recovered analytically: the
blend face's two **support faces** are intersected (``GeomAPI_IntSS`` on the underlying surfaces)
and a point of the blend face is projected onto the intersection curve. Support pairs: fillet → the
two neighbours it is tangent to; chamfer → the two neighbours it meets at an angle (neither tangent
nor perpendicular), else the two with the longest shared boundary; unknown kind (label-free use) →
every distinct pair from those rules, so one face may give several candidates. Any point of the
sharp edge identifies it (``cadseq.build.resolve_edge`` matches by distance), so the projected point
can stand in for the history's edge-midpoint reference.

``sharp_edges(shape, blend_faces)`` → ``VirtualEdge``s for the blend faces that could be resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VirtualEdge:
    face: int            # index of the blend face (graph / TopExp order)
    point: np.ndarray    # (3,) point on the sharp edge
    curve: str           # "line" | "circle" | "ellipse" | "bspline" | "other" (cadseq.build.curve_type)
    supports: tuple      # indices of the two support faces
    geom: object = None  # Geom_Curve of the sharp edge (infinite / full)


def _curve_name(curve) -> str:
    from OCP.GeomAbs import GeomAbs_BSplineCurve, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Line
    from OCP.GeomAdaptor import GeomAdaptor_Curve

    t = GeomAdaptor_Curve(curve).GetType()
    return {GeomAbs_Line: "line", GeomAbs_Circle: "circle", GeomAbs_Ellipse: "ellipse",
            GeomAbs_BSplineCurve: "bspline"}.get(t, "other")


def _face_point(face):
    """A point on the face (surface at the middle of its UV bounds, falling back to the centroid of
    its vertices) — unlike the centre of mass it lies on the blend even for full rings."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepTools import BRepTools

    umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
    p = BRep_Tool.Surface_s(face).Value(0.5 * (umin + umax), 0.5 * (vmin + vmax))
    return p


def _normal(face, edge):
    """Outward normal of ``face`` at the middle of ``edge``."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.TopAbs import TopAbs_REVERSED

    c = BRepAdaptor_Curve(edge)
    p = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
    surf = BRep_Tool.Surface_s(face)
    proj = GeomAPI_ProjectPointOnSurf(p, surf)
    if proj.NbPoints() == 0:
        return None
    u, v = proj.LowerDistanceParameters()
    props = GeomLProp_SLProps(surf, u, v, 1, 1e-7)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    n = np.array([n.X(), n.Y(), n.Z()])
    return -n if face.Orientation() == TopAbs_REVERSED else n


def _edge_length(edge) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    g = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, g)
    return g.Mass()


def sharp_edges(shape, blend_faces) -> list[VirtualEdge]:
    """Virtual sharp edges for the faces ``blend_faces`` (indices, or {index: "fillet" | "chamfer"})
    of ``shape`` (OCP TopoDS_Shape or cadquery Shape). Faces whose supports cannot be found or whose
    supports do not intersect are skipped."""
    from OCP.BRep import BRep_Tool
    from OCP.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

    s = shape.wrapped if hasattr(shape, "wrapped") else shape
    kinds = blend_faces if isinstance(blend_faces, dict) else {i: None for i in blend_faces}
    fmap = TopTools_IndexedMapOfShape()
    exp = TopExp_Explorer(s, TopAbs_FACE)
    while exp.More():
        fmap.Add(exp.Current())
        exp.Next()
    e2f = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(s, TopAbs_EDGE, TopAbs_FACE, e2f)

    out = []
    for fi, kind in kinds.items():
        face = TopoDS.Face_s(fmap.FindKey(fi + 1))
        nb = {}  # neighbour index -> [shared length, tangent?, |cos| of the normals]
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            edge = TopoDS.Edge_s(exp.Current())
            exp.Next()
            if BRep_Tool.Degenerated_s(edge) or not e2f.Contains(edge):
                continue
            for other in e2f.FindFromKey(edge):
                j = fmap.FindIndex(other) - 1
                if j < 0 or j == fi:
                    continue
                n1, n2 = _normal(face, edge), _normal(TopoDS.Face_s(other), edge)
                dot = float(n1 @ n2) if n1 is not None and n2 is not None else 0.0
                rec = nb.setdefault(j, [0.0, False, 0.0])
                rec[0] += _edge_length(edge)
                rec[1] |= dot > 0.999
                rec[2] = max(rec[2], abs(dot))
        cand = sorted(nb.items(), key=lambda kv: -kv[1][0])
        tangent = [j for j, r in cand if r[1]]
        angled = [j for j, r in cand if not r[1] and 0.05 < r[2] < 0.999]
        longest = [j for j, _ in cand]
        if kind == "fillet":
            rules = [tangent]
        elif kind == "chamfer":
            rules = [angled, longest]
        else:
            rules = [tangent, angled, longest]
        pairs = []
        for r in rules:
            if len(r) >= 2 and tuple(sorted(r[:2])) not in pairs:
                pairs.append(tuple(sorted(r[:2])))
            if kind is not None and pairs:
                break
        p = _face_point(face)
        found = []
        for sup in pairs:
            try:
                s1 = BRep_Tool.Surface_s(TopoDS.Face_s(fmap.FindKey(sup[0] + 1)))
                s2 = BRep_Tool.Surface_s(TopoDS.Face_s(fmap.FindKey(sup[1] + 1)))
                inter = GeomAPI_IntSS(s1, s2, 1e-7)
                if not inter.IsDone() or inter.NbLines() == 0:
                    continue
                best = None
                for k in range(1, inter.NbLines() + 1):
                    crv = inter.Line(k)
                    proj = GeomAPI_ProjectPointOnCurve(p, crv)
                    if proj.NbPoints() == 0:
                        continue
                    d = proj.LowerDistance()
                    if best is None or d < best[0]:
                        q = proj.NearestPoint()
                        best = (d, np.array([q.X(), q.Y(), q.Z()]), crv)
            except Exception:  # OCC numeric failures on degenerate surface pairs
                continue
            if best is not None and not any(np.allclose(best[1], v.point, atol=1e-6) for v in found):
                found.append(VirtualEdge(fi, best[1], _curve_name(best[2]), sup, best[2]))
        out += found
    return out
