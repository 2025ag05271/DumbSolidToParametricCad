"""STEP → UV-Net face-adjacency graph (script version of STEP_to_UVNet_Graph.ipynb).

Same logic as the notebook, as importable functions plus a command-line walkthrough:

    load_step            STEP file → TopoDS_Shape
    tessellate_shape     triangles per face (visualisation only)
    render_shape         3-D plot of the tessellated shape
    count_topo           number of faces / edges / vertices / solids
    check_brep           BRepCheck validity + non-manifold edge count
    sample_face_uv_grid  (N, N, 7) face grid  [x, y, z, nx, ny, nz, trim_mask]
    sample_edge_uv_grid  (N, 6)    edge grid  [x, y, z, tx, ty, tz]
    face_attributes      exact surface type / axis / radius / convexity of a face (optional extra)
    edge_attributes      curve type, convex / concave / smooth, length of an edge (optional extra)
    brep_to_graph_arrays face-adjacency graph as numpy arrays (no DGL needed)
    brep_to_dgl_graph    face-adjacency graph as a DGLGraph (ndata['x'] (F, N*N*7), edata['x'] (E, N*6))
    graph_checks         the notebook's sanity checks
    plot_graph           3-D centroid overlay + 2-D spring layout

This module is the single implementation of the UV-grid sampling used by the whole pipeline
(dataset generation, training data and prediction all call it through ``cadseq.graph``).

Usage (from solid2param/, with an environment that has cadquery-ocp; DGL/networkx optional):
    python step_to_graph.py examples/SimpleRectangle.stp
    python step_to_graph.py path/to/folder --uv 10 --plots out/ --save-npz out/
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_CLProps, BRepLProp_SLProps
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCP.GeomAbs import (
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.gp import gp_Dir, gp_Pnt, gp_Pnt2d, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_IN,
    TopAbs_ON,
    TopAbs_REVERSED,
    TopAbs_SOLID,
    TopAbs_VERTEX,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

SEED = 42
UV_RESOLUTION = 10
MAX_RANGE = 1000.0  # clamp for unbounded parametric domains (e.g. infinite planes)


# ------------------------------------------------------------------ 1. load --

def load_step(path: str | Path) -> TopoDS_Shape:
    """Read a STEP file and return the TopoDS_Shape."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"Null shape: {path}")
    return shape


def find_step_files(path: str | Path) -> list[Path]:
    """A single STEP file, or all *.step / *.stp files in a folder (sorted)."""
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(p.glob("*.step")) + sorted(p.glob("*.stp"))


# ------------------------------------------------------- 2. visualisation --

def tessellate_shape(shape: TopoDS_Shape, deflection: float = 0.1):
    """Return list-of-lists of triangles, one list per face."""
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)
    face_tris = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        trsf = loc.Transformation()
        tris = []
        if tri is not None:
            for i in range(1, tri.NbTriangles() + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                pts = [tri.Node(n).Transformed(trsf) for n in (n1, n2, n3)]
                tris.append([[p.X(), p.Y(), p.Z()] for p in pts])
        face_tris.append(tris)
        exp.Next()
    return face_tris


def render_shape(shape, title="", ax=None):
    """Render the tessellated shape with randomly colored faces."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    face_tris = tessellate_shape(shape)
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
    rng_c = random.Random(0)
    all_pts = []
    for tris in face_tris:
        color = [rng_c.random() * 0.6 + 0.3 for _ in range(3)] + [0.6]
        if tris:
            ax.add_collection3d(
                Poly3DCollection(tris, alpha=0.6, facecolor=color, edgecolor="gray", linewidth=0.15)
            )
            for t in tris:
                all_pts.extend(t)
    if all_pts:
        pts = np.array(all_pts)
        margin = 0.1 * (pts.max() - pts.min())
        for i, lab in enumerate("XYZ"):
            getattr(ax, f"set_{lab.lower()}lim")(pts[:, i].min() - margin, pts[:, i].max() + margin)
            getattr(ax, f"set_{lab.lower()}label")(lab)
    ax.set_title(title, fontsize=10)
    return ax


# ------------------------------------------------------------- 3. topology --

def count_topo(shape: TopoDS_Shape) -> dict:
    """Count B-Rep sub-shape elements.

    Same as the notebook: TopExp_Explorer visits a shared sub-shape once per parent, so edges and
    vertices are counted once per face that uses them (a box reports 24 edges / 48 vertices, i.e.
    12 unique edges × 2 faces and 8 unique vertices × 6 face-uses).
    """
    counts = {}
    for name, flag in [("faces", TopAbs_FACE), ("edges", TopAbs_EDGE),
                       ("vertices", TopAbs_VERTEX), ("solids", TopAbs_SOLID)]:
        exp, n = TopExp_Explorer(shape, flag), 0
        while exp.More():
            n += 1
            exp.Next()
        counts[name] = n
    return counts


# ------------------------------------------------------------- 4. validity --

def check_brep(shape: TopoDS_Shape) -> dict:
    """Basic B-Rep validity checks."""
    valid = BRepCheck_Analyzer(shape).IsValid()

    # Non-manifold edges (shared by >2 faces)
    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, emap)
    non_manifold = sum(1 for i in range(1, emap.Extent() + 1) if emap.FindFromIndex(i).Extent() > 2)
    return {"brep_valid": valid, "non_manifold_edges": non_manifold}


# ------------------------------------------------------- 5. UV-grid sampling --

def sample_face_uv_grid(face, N: int = UV_RESOLUTION) -> np.ndarray:
    """
    Sample an NxN UV-grid on a B-Rep face.
    Returns array of shape (N, N, 7): [x,y,z, nx,ny,nz, trim_mask].
    """
    adaptor = BRepAdaptor_Surface(face, True)
    u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()

    # Clamp large parametric domains (e.g. infinite planes)
    u0, u1 = max(u0, -MAX_RANGE), min(u1, MAX_RANGE)
    v0, v1 = max(v0, -MAX_RANGE), min(v1, MAX_RANGE)

    us = np.linspace(u0, u1, N)
    vs = np.linspace(v0, v1, N)
    classifier = BRepTopAdaptor_FClass2d(face, 1e-6)

    grid = np.zeros((N, N, 7), dtype=np.float32)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            pnt = adaptor.Value(u, v)
            grid[i, j, 0:3] = [pnt.X(), pnt.Y(), pnt.Z()]

            props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
            if props.IsNormalDefined():
                n = props.Normal()
                grid[i, j, 3:6] = [n.X(), n.Y(), n.Z()]

            state = classifier.Perform(gp_Pnt2d(u, v))
            grid[i, j, 6] = 1.0 if state in (TopAbs_IN, TopAbs_ON) else 0.0

    return grid


def sample_edge_uv_grid(edge, N: int = UV_RESOLUTION) -> np.ndarray:
    """
    Sample N points along a B-Rep edge curve.
    Returns array of shape (N, 6): [x,y,z, tx,ty,tz].
    """
    adaptor = BRepAdaptor_Curve(edge)
    t0, t1 = adaptor.FirstParameter(), adaptor.LastParameter()
    t0, t1 = max(t0, -MAX_RANGE), min(t1, MAX_RANGE)

    ts = np.linspace(t0, t1, N)
    grid = np.zeros((N, 6), dtype=np.float32)
    for i, t in enumerate(ts):
        pnt = adaptor.Value(t)
        grid[i, 0:3] = [pnt.X(), pnt.Y(), pnt.Z()]
        props = BRepLProp_CLProps(adaptor, t, 1, 1e-6)
        if props.IsTangentDefined():
            tangent = gp_Dir()
            props.Tangent(tangent)
            grid[i, 3:6] = [tangent.X(), tangent.Y(), tangent.Z()]
    return grid


# ------------------------------------- 5b. exact analytic attributes (optional) --
# Not part of the notebook. The B-Rep stores exact surface/curve definitions; these vectors expose
# them to learning models next to the sampled grids (e.g. a hole's exact radius and axis).

FACE_TYPES = ["plane", "cylinder", "cone", "sphere", "torus", "other"]
CURVE_TYPES = ["line", "circle", "ellipse", "other"]
EDGE_CONVEXITY = ["convex", "concave", "smooth"]
# face_attr: type one-hot (6) | axis or outward plane normal (3) | location (3) | r1 | r2 |
#            cone half-angle (rad) | convexity (+1 convex, -1 concave, 0 plane/other) | log1p(area)
FACE_ATTR_DIM = 17
FACE_ATTR_LENGTHS = slice(9, 14)  # mm-valued entries: location, r1, r2
# edge_attr: curve type one-hot (4) | convexity one-hot (3) | length | radius (circle)
EDGE_ATTR_DIM = 9
EDGE_ATTR_LENGTHS = slice(7, 9)
SMOOTH_COS = math.cos(math.radians(10.0))


def _vec(v) -> np.ndarray:
    return np.array([v.X(), v.Y(), v.Z()], dtype=np.float64)


def _canonical_dir(d: np.ndarray) -> np.ndarray:
    """Axis directions have no inherent sign: make the largest component positive."""
    return d if d[np.argmax(np.abs(d))] > 0 else -d


def _outward_normal(face, adaptor, u: float, v: float):
    props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    n = _vec(props.Normal())
    return -n if face.Orientation() == TopAbs_REVERSED else n


def _face_area(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def face_attributes(face) -> np.ndarray:
    """(FACE_ATTR_DIM,) exact surface description of a face (lengths in model units, mm)."""
    a = BRepAdaptor_Surface(face, True)
    out = np.zeros(FACE_ATTR_DIM, dtype=np.float32)
    u0, u1 = max(a.FirstUParameter(), -MAX_RANGE), min(a.LastUParameter(), MAX_RANGE)
    v0, v1 = max(a.FirstVParameter(), -MAX_RANGE), min(a.LastVParameter(), MAX_RANGE)
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    p = _vec(a.Value(um, vm))
    n = _outward_normal(face, a, um, vm)
    t = a.GetType()
    kind = {GeomAbs_Plane: 0, GeomAbs_Cylinder: 1, GeomAbs_Cone: 2, GeomAbs_Sphere: 3,
            GeomAbs_Torus: 4}.get(t, 5)
    out[kind] = 1.0

    def convexity(radial):
        if n is None or np.linalg.norm(radial) < 1e-9:
            return 0.0
        return 1.0 if np.dot(n, radial) > 0 else -1.0

    if kind == 0:
        nn = n if n is not None else _vec(a.Plane().Axis().Direction())
        out[6:9] = nn
        out[9:12] = nn * np.dot(nn, _vec(a.Plane().Location()))  # closest point to the origin
    elif kind in (1, 2, 4):
        g = a.Cylinder() if kind == 1 else a.Cone() if kind == 2 else a.Torus()
        ax = g.Axis()
        d = _canonical_dir(_vec(ax.Direction()))
        loc = _vec(ax.Location())
        out[6:9] = d
        if kind == 1:
            out[9:12] = loc - np.dot(loc, d) * d  # axis point closest to the origin
            out[12] = g.Radius()
        elif kind == 2:
            out[9:12] = _vec(g.Apex())
            out[12] = g.RefRadius()
            out[14] = g.SemiAngle()
        else:
            out[9:12] = loc
            out[12], out[13] = g.MajorRadius(), g.MinorRadius()
        foot = loc + np.dot(p - loc, d) * d
        radial = p - foot
        if kind == 4:  # distance from the torus' major circle
            radial = radial - (radial / max(np.linalg.norm(radial), 1e-12)) * g.MajorRadius()
        out[15] = convexity(radial)
    elif kind == 3:
        s = a.Sphere()
        c = _vec(s.Location())
        out[9:12] = c
        out[12] = s.Radius()
        out[15] = convexity(p - c)
    out[16] = math.log1p(_face_area(face))
    return out


def edge_attributes(edge, face1, face2) -> np.ndarray:
    """(EDGE_ATTR_DIM,) curve type, convexity between the two faces, length and radius."""
    out = np.zeros(EDGE_ATTR_DIM, dtype=np.float32)
    c = BRepAdaptor_Curve(edge)
    t = c.GetType()
    out[{GeomAbs_Line: 0, GeomAbs_Circle: 1, GeomAbs_Ellipse: 2}.get(t, 3)] = 1.0
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    out[7] = props.Mass()
    if t == GeomAbs_Circle:
        out[8] = c.Circle().Radius()

    tm = 0.5 * (c.FirstParameter() + c.LastParameter())
    pnt, d1 = gp_Pnt(), gp_Vec()
    c.D1(tm, pnt, d1)
    tangent = _vec(d1)
    # orient the tangent as the co-edge of face1 (material on the left of the co-edge)
    exp = TopExp_Explorer(face1, TopAbs_EDGE)
    while exp.More():
        if exp.Current().IsSame(edge):
            if exp.Current().Orientation() == TopAbs_REVERSED:
                tangent = -tangent
            break
        exp.Next()
    normals = []
    for face in (face1, face2):
        uv = ShapeAnalysis_Surface(BRep_Tool.Surface_s(face)).ValueOfUV(pnt, 1e-6)
        normals.append(_outward_normal(face, BRepAdaptor_Surface(face, True), uv.X(), uv.Y()))
    n1, n2 = normals
    if n1 is None or n2 is None or np.dot(n1, n2) > SMOOTH_COS:
        out[6] = 1.0  # smooth (tangent-continuous, e.g. fillet boundaries)
    elif np.dot(np.cross(n1, n2), tangent) > 0:
        out[4] = 1.0  # convex
    else:
        out[5] = 1.0  # concave
    return out


# --------------------------------------------- 6. face-adjacency graph build --

def brep_to_graph_arrays(shape: TopoDS_Shape, N: int = UV_RESOLUTION, with_centroids: bool = False,
                         with_attributes: bool = False) -> dict:
    """
    Face-adjacency graph (UV-Net style) as numpy arrays:
        node_x     (F, N, N, 7)  face UV-grids
        edge_index (2, E)        directed graph edges (both directions per shared B-Rep edge)
        edge_x     (E, N, 6)     edge U-grids
        centroids  (F, 3)        trimmed-face centroids (only if ``with_centroids``)
        face_attr  (F, FACE_ATTR_DIM), edge_attr (E, EDGE_ATTR_DIM)
                                 exact analytic attributes (only if ``with_attributes``)
    """
    # 1. Index faces
    face_map = TopTools_IndexedMapOfShape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face_map.Add(exp.Current())
        exp.Next()
    num_faces = face_map.Extent()

    # 2. Face UV-grids (node features) + centroids
    node_feats, centroids, face_attrs = [], [], []
    for i in range(1, num_faces + 1):
        face = TopoDS.Face_s(face_map.FindKey(i))
        grid = sample_face_uv_grid(face, N)  # (N, N, 7)
        node_feats.append(grid)
        if with_attributes:
            face_attrs.append(face_attributes(face))
        mask = grid[:, :, 6] > 0.5
        if mask.any():
            centroids.append(grid[:, :, 0:3][mask].mean(axis=0))
        else:
            centroids.append(grid[:, :, 0:3].mean(axis=(0, 1)))

    # 3. Build edges via shared B-Rep edges
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    src, dst, edge_feats, edge_attrs = [], [], [], []
    for i in range(1, edge_face_map.Extent() + 1):
        fi_list = []
        for adjacent_shape in edge_face_map.FindFromIndex(i):
            fi = face_map.FindIndex(adjacent_shape)
            if fi > 0:  # only valid face indices
                fi_list.append(fi - 1)
        if len(fi_list) < 2:
            continue  # boundary edge

        brep_edge = TopoDS.Edge_s(edge_face_map.FindKey(i))
        egrid = sample_edge_uv_grid(brep_edge, N)  # (N, 6)

        # Add graph edges (bidirectional)
        for a in range(len(fi_list)):
            for b in range(a + 1, len(fi_list)):
                src += [fi_list[a], fi_list[b]]
                dst += [fi_list[b], fi_list[a]]
                edge_feats += [egrid, egrid]
                if with_attributes:
                    fa = TopoDS.Face_s(face_map.FindKey(fi_list[a] + 1))
                    fb = TopoDS.Face_s(face_map.FindKey(fi_list[b] + 1))
                    edge_attrs += [edge_attributes(brep_edge, fa, fb), edge_attributes(brep_edge, fb, fa)]

    out = {
        "node_x": np.stack(node_feats) if node_feats else np.zeros((0, N, N, 7), dtype=np.float32),
        "edge_index": np.array([src, dst], dtype=np.int64).reshape(2, -1),
        "edge_x": np.stack(edge_feats) if edge_feats else np.zeros((0, N, 6), dtype=np.float32),
    }
    if with_centroids:
        out["centroids"] = np.array(centroids, dtype=np.float32).reshape(-1, 3)
    if with_attributes:
        out["face_attr"] = np.stack(face_attrs) if face_attrs else np.zeros((0, FACE_ATTR_DIM), np.float32)
        out["edge_attr"] = np.stack(edge_attrs) if edge_attrs else np.zeros((0, EDGE_ATTR_DIM), np.float32)
    return out


def brep_to_dgl_graph(shape: TopoDS_Shape, N: int = UV_RESOLUTION):
    """
    Convert a B-Rep shape into a DGL face-adjacency graph (UV-Net style).

    Returns:
        g        : DGL graph with ndata['x'] (F, N*N*7) and edata['x'] (E, N*6)
        centroids: (num_faces, 3) face centroids for visualization
    """
    import dgl
    import torch

    a = brep_to_graph_arrays(shape, N, with_centroids=True)
    g = dgl.graph((torch.from_numpy(a["edge_index"][0]), torch.from_numpy(a["edge_index"][1])),
                  num_nodes=a["node_x"].shape[0])
    g.ndata["x"] = torch.tensor(a["node_x"].reshape(a["node_x"].shape[0], -1), dtype=torch.float32)
    g.edata["x"] = torch.tensor(a["edge_x"].reshape(a["edge_x"].shape[0], -1), dtype=torch.float32)
    return g, a["centroids"]


# -------------------------------------------------------- 7. sanity checks --

def is_connected(num_nodes: int, edge_index: np.ndarray) -> bool:
    """Union-find connectivity test (no networkx needed)."""
    parent = list(range(num_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edge_index.T:
        parent[find(int(a))] = find(int(b))
    return num_nodes > 0 and len({find(i) for i in range(num_nodes)}) == 1


def graph_checks(arrays: dict, topo: dict, N: int = UV_RESOLUTION) -> dict:
    """The notebook's sanity checks, on the array form of the graph."""
    node_x, edge_index, edge_x = arrays["node_x"], arrays["edge_index"], arrays["edge_x"]
    num_nodes, num_edges = node_x.shape[0], edge_index.shape[1]
    return {
        "num_graph_nodes == num_brep_faces": num_nodes == topo["faces"],
        "graph has edges": num_edges > 0,
        "no self-loops": not bool((edge_index[0] == edge_index[1]).any()),
        "node feat dim = N*N*7": node_x[0].size == N * N * 7 if num_nodes else False,
        "edge feat dim = N*6": edge_x[0].size == N * 6 if num_edges else True,
        "no NaN in node features": not bool(np.isnan(node_x).any()),
        "no NaN in edge features": not bool(np.isnan(edge_x).any()) if num_edges else True,
        "graph is connected": is_connected(num_nodes, edge_index),
    }


# ---------------------------------------------------- 8. graph visualisation --

def plot_graph(shape: TopoDS_Shape, arrays: dict, title: str = "", seed: int = SEED):
    """3-D overlay of face centroids + adjacency on the part, and a 2-D spring layout."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    centroids, edge_index = arrays["centroids"], arrays["edge_index"]
    fig = plt.figure(figsize=(16, 7))

    # Left: 3-D overlay
    ax1 = fig.add_subplot(121, projection="3d")
    for tris in tessellate_shape(shape):
        if tris:
            ax1.add_collection3d(Poly3DCollection(tris, alpha=0.12, facecolor="lightgray",
                                                  edgecolor="silver", linewidth=0.1))
    ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="red", s=50, zorder=5, depthshade=False)
    drawn = set()
    for s, d in zip(edge_index[0], edge_index[1]):
        key = (min(s, d), max(s, d))
        if key in drawn:
            continue
        drawn.add(key)
        ax1.plot(*[[centroids[s, k], centroids[d, k]] for k in range(3)], "b-", alpha=0.6, linewidth=0.9)
    margin = 0.15 * (centroids.max() - centroids.min() + 1e-6)
    for i, lab in enumerate("XYZ"):
        getattr(ax1, f"set_{lab.lower()}lim")(centroids[:, i].min() - margin, centroids[:, i].max() + margin)
    ax1.set_title(f"3-D Graph Overlay\n{title}", fontsize=10)

    # Right: 2-D spring layout
    ax2 = fig.add_subplot(122)
    try:
        import networkx as nx

        nxg = nx.Graph()
        nxg.add_nodes_from(range(len(centroids)))
        nxg.add_edges_from(drawn)
        pos = nx.spring_layout(nxg, seed=seed)
        nx.draw_networkx(nxg, pos, ax=ax2, node_size=120, node_color="tomato",
                         edge_color="steelblue", width=0.8, font_size=7, with_labels=True)
    except ImportError:
        ax2.text(0.5, 0.5, "install networkx for the 2-D layout", ha="center", va="center")
    ax2.set_title(f"2-D Spring Layout\n{len(centroids)} nodes, {len(drawn)} edges", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- CLI --

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="STEP file or folder of STEP files")
    ap.add_argument("--uv", type=int, default=UV_RESOLUTION, help="UV-grid resolution N")
    ap.add_argument("--plots", type=Path, default=None, help="folder to save shape/graph PNGs")
    ap.add_argument("--save-npz", type=Path, default=None, help="folder to save graph arrays (.npz)")
    ap.add_argument("--dgl", action="store_true", help="also build the DGLGraph (requires dgl + torch)")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    files = find_step_files(args.path)
    if not files:
        raise SystemExit(f"No .step/.stp files found in {args.path}")

    for f in files:
        print(f"\n=== {f.name} ===")
        shape = load_step(f)

        topo = count_topo(shape)
        print(f"Topology : {topo}")

        ck = check_brep(shape)
        status = "OK" if ck["brep_valid"] and ck["non_manifold_edges"] == 0 else "WARN"
        print(f"Validity : [{status}] valid={ck['brep_valid']} non-manifold={ck['non_manifold_edges']}")

        arrays = brep_to_graph_arrays(shape, args.uv, with_centroids=True)
        n_nodes, n_edges = arrays["node_x"].shape[0], arrays["edge_index"].shape[1]
        print(f"Node grids: {arrays['node_x'].shape}  Edge grids: {arrays['edge_x'].shape}")

        for name, passed in graph_checks(arrays, topo, args.uv).items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        degree = np.bincount(arrays["edge_index"][1], minlength=n_nodes) if n_nodes else np.zeros(1)
        print(f"B-Rep: {topo['faces']} faces, {topo['edges']} edges")
        print(f"Graph: {n_nodes} nodes, {n_edges // 2} undirected edges, avg degree {degree.mean():.1f}")

        if args.dgl:
            g, _ = brep_to_dgl_graph(shape, args.uv)
            print(g)

        if args.save_npz:
            args.save_npz.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_npz / f"{f.stem}.npz",
                                **{k: v for k, v in arrays.items() if k != "centroids"})
        if args.plots:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            args.plots.mkdir(parents=True, exist_ok=True)
            fig = plt.figure(figsize=(8, 8))
            render_shape(shape, title=f.name, ax=fig.add_subplot(111, projection="3d"))
            fig.savefig(args.plots / f"{f.stem}_shape.png", dpi=110)
            plot_graph(shape, arrays, title=f.name).savefig(args.plots / f"{f.stem}_graph.png", dpi=110)
            plt.close("all")
            print(f"plots -> {args.plots}")


if __name__ == "__main__":
    main()
