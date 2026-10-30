"""Feature stages that decorate a base body. Each stage picks real faces/edges of the
current body, checks placement, and adds features atomically (rolled back on failure)."""

from __future__ import annotations

import math

import cadquery as cq

from cadseq.build import QUANTIZED_EDGE_TOL, edge_ref, resolve_edge
from cadseq.schema import Chamfer, Fillet, Hole, Mirror, MirrorPlane, Pad, Pocket, Sketch

from .context import Gen
from .geom import axis_of, dressable_edges, fits, host_faces, thickness
from .profiles import circle_profile, q, rect, regular_polygon, rounded_rect, slot, snap

# Metric clearance holes: (d, counterbore D, counterbore depth, countersink D) for M3..M16
ISO_HOLES = [
    (3.4, 6.5, 3.3, 6.3), (4.5, 8.0, 4.4, 9.4), (5.5, 10.0, 5.4, 10.4), (6.6, 11.0, 6.5, 12.6),
    (9.0, 15.0, 8.6, 17.3), (11.0, 18.0, 10.8, 20.0), (13.5, 20.0, 13.0, 24.0), (17.5, 26.0, 17.5, 32.0),
]
PLAIN_DIAMETERS = [2, 2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25]
CS_ANGLES = [82, 90, 90, 90, 100, 120]
FILLET_RADII = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]
CHAMFER_DISTS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]


def _pick_host(g: Gen, min_area=150.0, min_dim=10.0, prefer=None):
    hosts = host_faces(g.body, min_area, min_dim)
    if not hosts:
        return None
    if prefer:
        pref = [h for h in hosts if prefer(h)]
        if pref and g.rng.random() < 0.8:
            hosts = pref
    # area-weighted choice favours the main faces of the part
    return g.rng.choices(hosts, [h.face.Area() ** 0.5 for h in hosts])[0]


def _random_profile(g: Gen, host, lo=0.12, hi=0.5, margin=2.0):
    r = g.rng
    W, H = host.width, host.height
    shape = r.choices(["rect", "rounded", "circle", "slot", "polygon"], [0.25, 0.25, 0.2, 0.2, 0.1])[0]
    if shape == "circle":
        rad = max(1.5, q(r.uniform(lo, hi) * min(W, H) / 2))
        sw = sh = 2 * rad
    elif shape in ("rect", "rounded"):
        sw, sh = max(3.0, q(r.uniform(lo, hi) * W)), max(3.0, q(r.uniform(lo, hi) * H))
    elif shape == "slot":
        vertical = r.random() < 0.5
        along = H if vertical else W
        length = max(6.0, q(r.uniform(0.2, 0.6) * along))
        width = max(3.0, q(r.uniform(0.25, 0.6) * length))
        if width >= length - 1:
            return None
        sw, sh = (width, length) if vertical else (length, width)
    else:
        rad = max(3.0, q(r.uniform(lo, hi) * min(W, H) / 2))
        sw = sh = 2 * rad
    lo_u, hi_u = host.u0 + sw / 2 + margin, host.u1 - sw / 2 - margin
    lo_v, hi_v = host.v0 + sh / 2 + margin, host.v1 - sh / 2 - margin
    if lo_u > hi_u or lo_v > hi_v:
        return None
    cx, cy = q(r.uniform(lo_u, hi_u)), q(r.uniform(lo_v, hi_v))
    if shape == "circle":
        return circle_profile(cx, cy, sw / 2), (cx, cy)
    if shape == "rect":
        return rect(cx, cy, sw, sh), (cx, cy)
    if shape == "rounded":
        rr = q(r.uniform(0.5, min(sw, sh) / 2 - 0.5))
        return (rounded_rect(cx, cy, sw, sh, rr) if rr >= 0.5 else rect(cx, cy, sw, sh)), (cx, cy)
    if shape == "slot":
        return slot(cx, cy, max(sw, sh), min(sw, sh), vertical=sh > sw), (cx, cy)
    return regular_polygon(cx, cy, sw / 2, r.randint(3, 8), r.choice([0, 15, 30, 45])), (cx, cy)


def _placed_profile(g: Gen, host, tries=8, **kw):
    for _ in range(tries):
        res = _random_profile(g, host, **kw)
        if res is None:
            continue
        prof, center = res
        try:
            if fits(host.profile_face(prof), host.face, margin=1.5):
                return prof, center
        except Exception:
            continue
    return None


# ----------------------------------------------------------------- stages --

def add_boss(g: Gen) -> bool:
    host = _pick_host(g, min_area=300, min_dim=15)
    if host is None:
        return False
    res = _placed_profile(g, host, lo=0.15, hi=0.45)
    if res is None:
        return False
    s = Sketch(id=g.tid("S"), type="sketch", plane=host.plane, profiles=[res[0]])
    p = Pad(id=g.tid("P"), type="pad", sketch=s.id, length=q(g.rng.uniform(3, 30)), direction=1)
    return g.try_add([s, p])


def add_pocket(g: Gen) -> bool:
    host = _pick_host(g, min_area=200, min_dim=12)
    if host is None:
        return False
    res = _placed_profile(g, host, lo=0.15, hi=0.55)
    if res is None:
        return False
    prof, (cx, cy) = res
    t = thickness(g.body, host.world(cx, cy), -cq.Vector(host.axis))
    if t < 2:
        return False
    s = Sketch(id=g.tid("S"), type="sketch", plane=host.plane, profiles=[prof])
    if g.rng.random() < 0.25:
        k = Pocket(id=g.tid("K"), type="pocket", sketch=s.id, through_all=True, direction=-1)
    else:
        depth = max(1.0, q(g.rng.uniform(0.2, 0.7) * t))
        k = Pocket(id=g.tid("K"), type="pocket", sketch=s.id, depth=depth, direction=-1)
    return g.try_add([s, k])


def _hole_spec(g: Gen, max_outer: float):
    r = g.rng
    kind = r.choices(["simple", "counterbore", "countersink"], [0.5, 0.25, 0.25])[0]
    if kind == "simple":
        ds = [d for d in PLAIN_DIAMETERS if d <= max_outer]
        return (kind, r.choice(ds), {}) if ds else None
    opts = [h for h in ISO_HOLES if (h[1] if kind == "counterbore" else h[3]) <= max_outer]
    if not opts:
        return None
    d, cbd, cbt, csd = r.choice(opts)
    if kind == "counterbore":
        return kind, d, {"cb_diameter": cbd, "cb_depth": cbt}
    return kind, d, {"cs_diameter": csd, "cs_angle": r.choice(CS_ANGLES)}


def _outer_d(kind, d, extra):
    return extra.get("cb_diameter") or extra.get("cs_diameter") or d


def _pattern_centers(g: Gen, host, od: float, circular_bias: float):
    r = g.rng
    m = od / 2 + 2.0
    u0, u1, v0, v1 = host.u0 + m, host.u1 - m, host.v0 + m, host.v1 - m
    if u0 > u1 or v0 > v1:
        return None
    kind = r.choices(["single", "linear", "grid", "circular"],
                     [0.3, 0.25, 0.15, 0.3 + circular_bias])[0]
    if kind == "single":
        return [(q(r.uniform(u0, u1)), q(r.uniform(v0, v1)))]
    if kind == "linear":
        horizontal = (u1 - u0) >= (v1 - v0) if r.random() < 0.7 else r.random() < 0.5
        span = (u1 - u0) if horizontal else (v1 - v0)
        n = r.randint(2, 5)
        pitch = q(r.uniform(od + 2, max(od + 2, span / max(1, n - 1))))
        if pitch * (n - 1) > span:
            n = int(span // pitch) + 1
        if n < 2:
            return None
        start = q(r.uniform(u0 if horizontal else v0, (u1 if horizontal else v1) - pitch * (n - 1)))
        other = q(r.uniform(v0, v1) if horizontal else r.uniform(u0, u1))
        return [((start + i * pitch, other) if horizontal else (other, start + i * pitch)) for i in range(n)]
    if kind == "grid":
        nu, nv = r.randint(2, 3), r.randint(2, 3)
        pu = q(r.uniform(od + 2, max(od + 2, (u1 - u0) / (nu - 1))))
        pv = q(r.uniform(od + 2, max(od + 2, (v1 - v0) / (nv - 1))))
        if pu * (nu - 1) > u1 - u0 or pv * (nv - 1) > v1 - v0:
            return None
        cu, cv = q((host.u0 + host.u1) / 2), q((host.v0 + host.v1) / 2)
        return [(cu + (i - (nu - 1) / 2) * pu, cv + (j - (nv - 1) / 2) * pv)
                for i in range(nu) for j in range(nv)]
    n = r.randint(3, 8)
    cu, cv = q((host.u0 + host.u1) / 2), q((host.v0 + host.v1) / 2)
    rmin = (od + 2) / (2 * math.sin(math.pi / n))
    rmax = min(host.width, host.height) / 2 - m + 2.0 - 0.5
    if rmin > rmax:
        return None
    rad = q(r.uniform(rmin, rmax))
    a0 = math.radians(r.choice([0, 15, 30, 45, 90]) if r.random() < 0.5 else 0)
    return [(cu + rad * math.cos(a0 + 2 * math.pi * i / n), cv + rad * math.sin(a0 + 2 * math.pi * i / n))
            for i in range(n)]


def add_hole_group(g: Gen) -> bool:
    bolt_z = g.hints.get("bolt_circle_face_z")
    prefer = (lambda h: h.axis == (0, 0, 1) and abs(h.plane.origin[2] - bolt_z) < 1e-6) if bolt_z is not None else None
    host = _pick_host(g, min_area=150, min_dim=10, prefer=prefer)
    if host is None:
        return False
    spec = _hole_spec(g, max_outer=min(host.width, host.height) * 0.4)
    if spec is None:
        return False
    kind, d, extra = spec
    od = _outer_d(kind, d, extra)
    centers = _pattern_centers(g, host, od, circular_bias=0.4 if g.hints.get("turned") else 0.0)
    if not centers:
        return False
    for (u, v) in centers:
        if not fits(host.profile_face(circle_profile(u, v, od / 2)), host.face, margin=1.0):
            return False

    into = -cq.Vector(host.axis)
    t = min(thickness(g.body, host.world(u, v), into) for u, v in centers)
    if t < 1.5:
        return False
    head = extra.get("cb_depth", 0.0)
    if "cs_diameter" in extra:
        head = (extra["cs_diameter"] / 2) / math.tan(math.radians(extra["cs_angle"] / 2))
    blind_depth = q(g.rng.uniform(0.4, 0.85) * t)
    through = g.rng.random() < 0.6 or blind_depth <= head + 1.0

    holes = []
    for (u, v) in centers:
        p = host.world(u, v)
        holes.append(Hole(
            id=g.tid("H"), type="hole", kind=kind, position=(snap(p.x), snap(p.y), snap(p.z)),
            direction=tuple(float(-c) + 0.0 for c in host.axis), diameter=d,
            depth=None if through else blind_depth, through_all=through, **extra,
        ))
    return g.try_add(holes)


def _select_edges(g: Gen):
    r = g.rng
    edges = dressable_edges(g.body)
    if not edges:
        return []
    mode = r.choices(["axis", "circles", "random"], [0.45, 0.3, 0.25])[0]
    if mode == "axis":
        ax = cq.Vector(r.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)]))
        cand = [e for e in edges if e.geomType() == "LINE" and abs(e.tangentAt(0.5).dot(ax)) > 1 - 1e-7]
        kmax = 6
    elif mode == "circles":
        cand = [e for e in edges if e.geomType() == "CIRCLE"]
        kmax = 4
    else:
        cand, kmax = edges, 4
    if not cand:
        cand, kmax = edges, 3
    r.shuffle(cand)
    picked = cand[: r.randint(1, min(kmax, len(cand)))]
    out = []
    for e in picked:
        ref = edge_ref(e)
        ref = ref.model_copy(update={"point": tuple(snap(c) for c in ref.point)})
        try:
            if resolve_edge(g.body, ref, QUANTIZED_EDGE_TOL).isSame(e):
                out.append((e, ref))
        except Exception:
            pass
    return out


def _add_dressup(g: Gen, kind: str) -> bool:
    sel = _select_edges(g)
    if not sel:
        return False
    min_len = min(e.Length() for e, _ in sel)
    sizes = [s for s in (FILLET_RADII if kind == "fillet" else CHAMFER_DISTS) if s <= 0.3 * min_len]
    if not sizes:
        return False
    refs = [ref for _, ref in sel]
    idx = g.rng.randrange(len(sizes))
    for _ in range(3):  # shrink on failure
        s = sizes[idx]
        if kind == "fillet":
            f = Fillet(id=g.tid("F"), type="fillet", radius=s, edges=refs)
        else:
            f = Chamfer(id=g.tid("C"), type="chamfer", distance=s, edges=refs)
        if g.try_add([f]):
            return True
        if idx == 0:
            break
        idx //= 2
    return False


def add_fillet(g: Gen) -> bool:
    return _add_dressup(g, "fillet")


def add_chamfer(g: Gen) -> bool:
    return _add_dressup(g, "chamfer")


def add_mirror(g: Gen) -> bool:
    bb = g.body.BoundingBox()
    lo = {0: bb.xmin, 1: bb.ymin, 2: bb.zmin}
    hi = {0: bb.xmax, 1: bb.ymax, 2: bb.zmax}
    side = {0: (bb.ylen * bb.zlen), 1: (bb.xlen * bb.zlen), 2: (bb.xlen * bb.ylen)}
    cands = []
    for f in g.body.Faces():
        if f.geomType() != "PLANE":
            continue
        axis = axis_of(f.normalAt())
        if axis is None:
            continue
        i = [abs(c) for c in axis].index(1)
        coord = f.Center().toTuple()[i]
        extreme = hi[i] if sum(axis) > 0 else lo[i]
        if abs(coord - extreme) < 1e-3 and f.Area() >= 0.15 * side[i]:
            cands.append((axis, snap(coord)))
    if not cands:
        return False
    axis, coord = g.rng.choice(cands)
    origin = tuple(float(coord) if c else 0.0 for c in axis)
    m = Mirror(id=g.tid("M"), type="mirror",
               plane=MirrorPlane(origin=origin, normal=tuple(float(abs(c)) for c in axis)))
    return g.try_add([m])
