"""STEP → UV-Net face-adjacency graph (script version of STEP_to_UVNet_Graph.ipynb).

Same logic as the notebook, as importable functions plus a command-line walkthrough:

    load_step            STEP file → TopoDS_Shape
    tessellate_shape     triangles per face (visualisation only)
    render_shape         3-D plot of the tessellated shape
    count_topo           number of faces / edges / vertices / solids
    check_brep           BRepCheck validity + non-manifold edge count
    sample_face_uv_grid  (N, N, 7) face grid  [x, y, z, nx, ny, nz, trim_mask]
    sample_edge_uv_grid  (N, 6)    edge grid  [x, y, z, tx, ty, tz]
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
import random
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLProp import BRepLProp_CLProps, BRepLProp_SLProps
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCP.gp import gp_Dir, gp_Pnt2d
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_SOLID, TopAbs_VERTEX
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


# --------------------------------------------- 6. face-adjacency graph build --

def brep_to_graph_arrays(shape: TopoDS_Shape, N: int = UV_RESOLUTION, with_centroids: bool = False) -> dict:
    """
    Face-adjacency graph (UV-Net style) as numpy arrays:
        node_x     (F, N, N, 7)  face UV-grids
        edge_index (2, E)        directed graph edges (both directions per shared B-Rep edge)
        edge_x     (E, N, 6)     edge U-grids
        centroids  (F, 3)        trimmed-face centroids (only if ``with_centroids``)
    """
    # 1. Index faces
    face_map = TopTools_IndexedMapOfShape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face_map.Add(exp.Current())
        exp.Next()
    num_faces = face_map.Extent()

    # 2. Face UV-grids (node features) + centroids
    node_feats, centroids = [], []
    for i in range(1, num_faces + 1):
        grid = sample_face_uv_grid(TopoDS.Face_s(face_map.FindKey(i)), N)  # (N, N, 7)
        node_feats.append(grid)
        mask = grid[:, :, 6] > 0.5
        if mask.any():
            centroids.append(grid[:, :, 0:3][mask].mean(axis=0))
        else:
            centroids.append(grid[:, :, 0:3].mean(axis=(0, 1)))

    # 3. Build edges via shared B-Rep edges
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    src, dst, edge_feats = [], [], []
    for i in range(1, edge_face_map.Extent() + 1):
        fi_list = []
        for adjacent_shape in edge_face_map.FindFromIndex(i):
            fi = face_map.FindIndex(adjacent_shape)
            if fi > 0:  # only valid face indices
                fi_list.append(fi - 1)
        if len(fi_list) < 2:
            continue  # boundary edge

        egrid = sample_edge_uv_grid(TopoDS.Edge_s(edge_face_map.FindKey(i)), N)  # (N, 6)

        # Add graph edges (bidirectional)
        for a in range(len(fi_list)):
            for b in range(a + 1, len(fi_list)):
                src += [fi_list[a], fi_list[b]]
                dst += [fi_list[b], fi_list[a]]
                edge_feats += [egrid, egrid]

    out = {
        "node_x": np.stack(node_feats) if node_feats else np.zeros((0, N, N, 7), dtype=np.float32),
        "edge_index": np.array([src, dst], dtype=np.int64).reshape(2, -1),
        "edge_x": np.stack(edge_feats) if edge_feats else np.zeros((0, N, 6), dtype=np.float32),
    }
    if with_centroids:
        out["centroids"] = np.array(centroids, dtype=np.float32).reshape(-1, 3)
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
