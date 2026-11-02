"""Feature identification for a STEP solid: which faces form which machining / modelling feature.

    .venv\\Scripts\\python.exe identify.py part.stp --ckpt runs/seg/best.pt --out out/part

Outputs (``--out`` prefix):
    <out>_features.json   per-face class + region, feature regions, and a feature list with
                          parameters read from the exact B-Rep (hole diameter / axis / depth / through,
                          counterbore / countersink size, fillet radius, extrude direction / height)
    <out>_features.png    render with faces coloured by feature class and numbered by region
    <out>_features.step   the input solid with per-face colours (open in CATIA / any STEP viewer)

Face classes (UV-Net / Fusion 360 segmentation style): extrude side / end, cut side / end,
simple / counterbore / countersink hole, fillet, chamfer. A *region* is a connected set of faces of
one feature; fillet / chamfer regions with the same size are grouped into one feature, as a CAD
tree would.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cadseq.dataset import collate, scale_features, to_device
from cadseq.provenance import FACE_CLASSES
from s2pnet.segment import CLASS_TO_TYPE, FEATURE_TYPES, FaceSegmenter, predict
from step_to_graph import FACE_TYPES, brep_to_graph_arrays, load_step, tessellate_shape

CLASS_COLORS = {
    "extrude_side": (0.62, 0.70, 0.80), "extrude_end": (0.42, 0.50, 0.62),
    "cut_side": (0.98, 0.62, 0.25), "cut_end": (0.85, 0.40, 0.10),
    "hole_simple": (0.90, 0.20, 0.20), "hole_counterbore": (0.55, 0.25, 0.75),
    "hole_countersink": (0.90, 0.30, 0.65), "fillet": (0.25, 0.70, 0.35), "chamfer": (0.15, 0.65, 0.70),
}
PRETTY = {
    "extrude_side": "Extrude (side)", "extrude_end": "Extrude (end)", "cut_side": "Cut / pocket (side)",
    "cut_end": "Cut / pocket (floor)", "hole_simple": "Hole", "hole_counterbore": "Counterbore hole",
    "hole_countersink": "Countersink hole", "fillet": "Fillet", "chamfer": "Chamfer",
}
TYPE_PRETTY = {"extrude": "Extrude / boss", "cut": "Pocket / cut", "hole_simple": "Hole",
               "hole_counterbore": "Counterbore hole", "hole_countersink": "Countersink hole",
               "fillet": "Fillet", "chamfer": "Chamfer"}


def _masked_points(grid: np.ndarray) -> np.ndarray:
    m = grid[..., 6] > 0.5
    return grid[..., :3][m] if m.any() else grid[..., :3].reshape(-1, 3)


def describe(ftype: str, faces: list[int], cls_names: list[str], node_x: np.ndarray, face_attr: np.ndarray) -> dict:
    """Engineering parameters of one feature region, from the exact face attributes (mm)."""
    types = [FACE_TYPES[int(face_attr[f, :6].argmax())] for f in faces]
    pts = np.concatenate([_masked_points(node_x[f]) for f in faces])
    info: dict = {}
    if ftype.startswith("hole"):
        cyl = [f for f, t in zip(faces, types) if t == "cylinder"]
        if cyl:
            radii = sorted({round(float(face_attr[f, 12]), 4) for f in cyl})
            main = min(cyl, key=lambda f: face_attr[f, 12])
            axis = face_attr[main, 6:9].astype(float)
            on_axis = face_attr[main, 9:12].astype(float)
            tvals = (pts - on_axis) @ axis
            info.update({"diameter": round(2 * radii[0], 3), "axis": np.round(axis, 4).tolist(),
                         "depth": round(float(np.ptp(_masked_points(node_x[main]) @ axis)), 3),
                         "axis_point": np.round(on_axis + axis * tvals.mean(), 3).tolist()})
            if ftype == "hole_counterbore" and len(radii) > 1:
                info["counterbore_diameter"] = round(2 * radii[-1], 3)
            floors = [f for f, t in zip(faces, types) if t == "plane" and abs(abs(face_attr[f, 6:9] @ axis) - 1) < 1e-3]
            # a hole bottom lies entirely within the hole radius (a counterbore floor is an annulus)
            radial = lambda f: np.linalg.norm(np.cross(_masked_points(node_x[f]) - on_axis, axis), axis=1)  # noqa: E731
            info["through"] = not any(radial(f).max() < radii[0] + 1e-3 for f in floors)
        cones = [f for f, t in zip(faces, types) if t == "cone"]
        if cones:
            info["countersink_angle_deg"] = round(math.degrees(2 * float(face_attr[cones[0], 14])), 2)
    elif ftype in ("fillet", "chamfer"):
        r = [float(face_attr[f, 12]) for f, t in zip(faces, types) if t == "cylinder"]
        r += [float(face_attr[f, 13]) for f, t in zip(faces, types) if t == "torus"]
        if ftype == "fillet" and r:
            info["radius"] = round(min(r), 3)
        conv = [face_attr[f, 15] for f, t in zip(faces, types) if t in ("cylinder", "torus", "cone")]
        if conv:
            info["edge"] = "convex" if np.mean(conv) > 0 else "concave"
    else:  # extrude / cut: direction from an end (or floor) face, height along it
        ends = [f for f in faces if cls_names[f].endswith("_end") and FACE_TYPES[int(face_attr[f, :6].argmax())] == "plane"]
        if ends:
            n = face_attr[ends[0], 6:9].astype(float)
            info["direction"] = np.round(n, 4).tolist()
            info["height"] = round(float(np.ptp(pts @ n)), 3)
    lo, hi = pts.min(0), pts.max(0)
    info["bbox"] = [np.round(lo, 3).tolist(), np.round(hi, 3).tolist()]
    info["surface_types"] = sorted(set(types))
    return info


def group_features(regions: list[dict]) -> list[dict]:
    """History-style grouping: fillet / chamfer regions with equal size and edge convexity form one
    feature (as in a CAD tree, where one fillet feature rounds several edges)."""
    groups: dict = {}
    for r in regions:
        if r["type"] in ("fillet", "chamfer"):
            key = (r["type"], r.get("radius"), r.get("edge"))
        else:
            key = ("region", r["region"])
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        g = {"type": rs[0]["type"], "label": TYPE_PRETTY[rs[0]["type"]], "regions": [r["region"] for r in rs],
             "faces": sorted(f for r in rs for f in r["faces"])}
        for k in ("radius", "edge", "diameter", "counterbore_diameter", "countersink_angle_deg", "depth",
                  "through", "axis", "axis_point", "direction", "height"):
            if k in rs[0]:
                g[k] = rs[0][k]
        out.append(g)
    out.sort(key=lambda g: (FEATURE_TYPES.index(g["type"]), g["faces"][0]))
    return out


def write_colored_step(shape, colors: list[tuple], path: Path) -> None:
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    XCAFApp_Application.GetApplication_s().NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    label = st.AddShape(shape, False)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    for rgb in colors:
        sub = st.AddSubShape(label, exp.Current())
        ct.SetColor(sub, Quantity_Color(*rgb, Quantity_TOC_RGB), XCAFDoc_ColorType.XCAFDoc_ColorSurf)
        exp.Next()
    writer = STEPCAFControl_Writer()
    writer.SetColorMode(True)
    writer.Transfer(doc, STEPControl_AsIs)
    writer.Write(str(path))


def _vtk_render(shape, cls_names, size=(1100, 850), elev_deg=28.0, azim_deg=-55.0):
    """Offscreen VTK render (proper depth buffer): faces coloured by class + black B-Rep edges.
    Returns (RGB image, world→pixel projection function)."""
    import vtk
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from vtk.util.numpy_support import vtk_to_numpy

    pts, polys = vtk.vtkPoints(), vtk.vtkCellArray()
    cols = vtk.vtkUnsignedCharArray()
    cols.SetNumberOfComponents(3)
    allp = []
    for tris, c in zip(tessellate_shape(shape, 0.05), cls_names):
        rgb = [int(255 * v) for v in CLASS_COLORS[c]]
        for t in tris:
            polys.InsertNextCell(3, [pts.InsertNextPoint(*p) for p in t])
            cols.InsertNextTuple3(*rgb)
            allp.extend(t)
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(pts)
    mesh.SetPolys(polys)
    mesh.GetCellData().SetScalars(cols)
    m1 = vtk.vtkPolyDataMapper()
    m1.SetInputData(mesh)
    m1.SetResolveCoincidentTopologyToPolygonOffset()
    faces = vtk.vtkActor()
    faces.SetMapper(m1)
    faces.GetProperty().SetInterpolationToFlat()
    faces.GetProperty().SetAmbient(0.35)
    faces.GetProperty().SetDiffuse(0.7)

    lpts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        c = BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))
        ts = np.linspace(c.FirstParameter(), c.LastParameter(), 128)
        ids = [lpts.InsertNextPoint(*(lambda p: (p.X(), p.Y(), p.Z()))(c.Value(t))) for t in ts]
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(ids))
        for k, i in enumerate(ids):
            line.GetPointIds().SetId(k, i)
        lines.InsertNextCell(line)
        exp.Next()
    wire = vtk.vtkPolyData()
    wire.SetPoints(lpts)
    wire.SetLines(lines)
    m2 = vtk.vtkPolyDataMapper()
    m2.SetInputData(wire)
    edges = vtk.vtkActor()
    edges.SetMapper(m2)
    edges.GetProperty().SetColor(0.1, 0.1, 0.1)
    edges.GetProperty().SetLineWidth(1.5)

    ren = vtk.vtkRenderer()
    ren.AddActor(faces)
    ren.AddActor(edges)
    ren.SetBackground(1, 1, 1)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(ren)
    win.SetSize(*size)
    allp = np.asarray(allp)
    ctr = (allp.min(0) + allp.max(0)) / 2
    el, az = math.radians(elev_deg), math.radians(azim_deg)
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(*ctr)
    cam.SetPosition(*(ctr + np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])))
    cam.SetViewUp(0, 0, 1)
    ren.ResetCamera()
    cam.Zoom(1.15)
    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.Update()
    out = grab.GetOutput()
    w, h, _ = out.GetDimensions()
    img = vtk_to_numpy(out.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3]

    vtk_m = cam.GetCompositeProjectionTransformMatrix(w / h, -1, 1)
    M = np.array([[vtk_m.GetElement(i, j) for j in range(4)] for i in range(4)])
    zbuf = vtk.vtkFloatArray()
    win.GetZbufferData(0, 0, w - 1, h - 1, zbuf)
    zbuf = vtk_to_numpy(zbuf).reshape(h, w)[::-1]

    def project(p):
        """World point → (x, y) pixel and whether it is visible (not occluded) in the render."""
        x, y, z, wq = M @ np.array([*map(float, p), 1.0])
        px, py = (x / wq + 1) / 2 * w, (1 - (y / wq + 1) / 2) * h
        ix, iy = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
        win_z = (z / wq + 1) / 2
        nb = zbuf[max(iy - 2, 0):iy + 3, max(ix - 2, 0):ix + 3]
        return px, py, bool(win_z <= nb.max() + 2e-4)

    return img, project


def render(shape, cls_names: list[str], instance: np.ndarray, anchors: list, title: str, path: Path):
    """``anchors[f]``: candidate label points on face f (first visible one is used per region)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    img, project = _vtk_render(shape, cls_names)
    fig, ax = plt.subplots(figsize=(img.shape[1] / 110, img.shape[0] / 110), dpi=110)
    ax.imshow(img)
    for inst in np.unique(instance):
        faces = np.nonzero(instance == inst)[0]
        if cls_names[faces[0]].startswith("extrude"):
            continue
        for f in faces:
            hit = next(((x, y) for x, y, vis in (project(p) for p in anchors[f]) if vis), None)
            if hit:
                ax.text(*hit, str(inst), fontsize=7, color="black", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
                break
    present = [c for c in FACE_CLASSES if c in cls_names]
    ax.legend(handles=[Patch(color=CLASS_COLORS[c], label=PRETTY[c]) for c in present],
              loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def identify(step_path: Path, ckpt: Path, uv: int = 10, device: str | None = None) -> tuple:
    shape = load_step(step_path)
    raw = brep_to_graph_arrays(shape, uv, with_centroids=True, with_attributes=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FaceSegmenter.from_checkpoint(torch.load(ckpt, map_location="cpu", weights_only=False)["model"]).to(dev)
    item = scale_features({k: raw[k].copy() for k in ("node_x", "edge_x", "face_attr", "edge_attr")})
    item = {k: torch.from_numpy(v) for k, v in item.items()}
    item.update({"id": step_path.stem, "edge_index": torch.from_numpy(raw["edge_index"]),
                 "tokens": torch.zeros(1, dtype=torch.long)})
    batch = to_device(collate([item]), dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        pred = predict(model, batch)[0]
    return shape, raw, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", type=Path)
    ap.add_argument("--ckpt", type=Path, default=Path("runs/seg/best.pt"))
    ap.add_argument("--out", type=Path, default=None, help="output prefix (default: next to the STEP)")
    ap.add_argument("--uv", type=int, default=10)
    args = ap.parse_args()

    out = args.out or args.step.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    shape, raw, pred = identify(args.step, args.ckpt, args.uv)
    cls_names = [FACE_CLASSES[c] for c in pred["cls"]]
    regions = []
    for inst in np.unique(pred["instance"]):
        faces = [int(f) for f in np.nonzero(pred["instance"] == inst)[0]]
        ftype = FEATURE_TYPES[np.bincount(CLASS_TO_TYPE[pred["cls"][faces]]).argmax()]
        regions.append({
            "region": int(inst), "type": ftype, "faces": faces,
            "face_classes": sorted({cls_names[f] for f in faces}),
            "confidence": round(float(pred["prob"][faces, pred["cls"][faces]].mean()), 3),
            **describe(ftype, faces, cls_names, raw["node_x"], raw["face_attr"]),
        })
    features = group_features(regions)
    result = {"step": str(args.step), "faces": len(cls_names), "face_class": cls_names,
              "face_region": pred["instance"].tolist(), "features": features, "regions": regions}
    (out.parent / f"{out.name}_features.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    counts: dict = {}
    for f in features:
        counts[f["label"]] = counts.get(f["label"], 0) + 1
    title = f"{args.step.name}: " + ", ".join(f"{v}× {k}" for k, v in counts.items())
    # label anchors: the face sample closest to the face's mean point, then a spread of other samples
    anchors = []
    for g in raw["node_x"]:
        p = _masked_points(g)
        order = np.argsort(np.linalg.norm(p - p.mean(0), axis=1))
        anchors.append(p[np.concatenate([order[:1], order[1:][:: max(1, len(order) // 8)]])])
    render(shape, cls_names, pred["instance"], anchors, title, out.parent / f"{out.name}_features.png")
    write_colored_step(shape, [CLASS_COLORS[c] for c in cls_names], out.parent / f"{out.name}_features.step")

    print(f"{args.step.name}: {len(cls_names)} faces, {len(regions)} feature regions, {len(features)} features")
    for i, f in enumerate(features, 1):
        params = {k: v for k, v in f.items() if k not in ("type", "label", "faces", "regions")}
        where = f"regions {f['regions']}" if len(f["regions"]) > 1 else f"region {f['regions'][0]}"
        print(f"  {i:>2}. {f['label']:<18} {where:<16} faces {len(f['faces']):<3} {params}")
    print(f"wrote {out}_features.json / .png / .step")


if __name__ == "__main__":
    main()
