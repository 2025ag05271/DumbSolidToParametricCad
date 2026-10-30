"""Base-body templates for each part family. Families only add sketch+pad features;
family-specific pockets are deferred to the pocket stage to keep canonical ordering."""

from __future__ import annotations

from cadseq.schema import Pad, Pocket, Profile, Sketch

from .context import Gen
from .geom import axis_plane
from .profiles import (
    arc,
    chamfered_rect,
    circle_profile,
    line,
    polyline,
    q,
    rect,
    regular_polygon,
    rounded_rect,
)

Z, NZ, NY = (0, 0, 1), (0, 0, -1), (0, -1, 0)


def _sketch_pad(g: Gen, plane, profile: Profile, length, direction=1, symmetric=False) -> bool:
    s = Sketch(id=g.tid("S"), type="sketch", plane=plane, profiles=[profile])
    p = Pad(id=g.tid("P"), type="pad", sketch=s.id, length=length, direction=direction,
            symmetric=symmetric)
    return g.try_add([s, p])


def _defer_pocket(g: Gen, plane, profile: Profile, depth=None, through_all=False):
    def run(g: Gen) -> bool:
        s = Sketch(id=g.tid("S"), type="sketch", plane=plane, profiles=[profile])
        k = Pocket(id=g.tid("K"), type="pocket", sketch=s.id, depth=depth,
                   through_all=through_all, direction=-1)
        return g.try_add([s, k])
    g.deferred_pockets.append(run)


# ------------------------------------------------------------- prismatic --

def plate(g: Gen) -> bool:
    r = g.rng
    w, h, t = q(r.uniform(40, 220)), q(r.uniform(30, 160)), q(r.uniform(4, 25))
    kind = r.choices(["rect", "rounded", "chamfered", "polygon"], [0.35, 0.35, 0.15, 0.15])[0]
    if kind == "rect":
        prof = rect(0, 0, w, h)
    elif kind == "rounded":
        prof = rounded_rect(0, 0, w, h, q(r.uniform(2, min(w, h) * 0.3)))
    elif kind == "chamfered":
        prof = chamfered_rect(0, 0, w, h, q(r.uniform(2, min(w, h) * 0.25)))
    else:
        prof = regular_polygon(0, 0, q(r.uniform(25, 100)), r.choice([5, 6, 8]))
    return _sketch_pad(g, axis_plane(Z, 0), prof, t)


def block(g: Gen) -> bool:
    r = g.rng
    w, d, h = q(r.uniform(30, 140)), q(r.uniform(30, 140)), q(r.uniform(20, 90))
    return _sketch_pad(g, axis_plane(Z, 0), rect(0, 0, w, d), h)


def _xz_extrusion(g: Gen, pts_or_curves, width) -> bool:
    """Profile drawn in the XZ plane (sketch normal -Y => local Y = +Z), padded along +Y."""
    outer = polyline(pts_or_curves) if isinstance(pts_or_curves[0], tuple) else pts_or_curves
    return _sketch_pad(g, axis_plane(NY, 0), Profile(outer=outer), width, direction=-1)


def l_bracket(g: Gen, gussets: bool = False) -> bool:
    r = g.rng
    a, b = q(r.uniform(40, 160)), q(r.uniform(40, 140))
    t1, t2 = q(r.uniform(5, 20)), q(r.uniform(5, 20))
    w = q(r.uniform(30, 140))
    if r.random() < 0.35:  # rounded top on the upright
        rad = t2 / 2
        curves = [
            line((0, 0), (a, 0)), line((a, 0), (a, t1)), line((a, t1), (t2, t1)),
            line((t2, t1), (t2, b)), arc((t2, b), (rad, b + rad), (0, b)), line((0, b), (0, 0)),
        ]
        ok = _xz_extrusion(g, curves, w)
    else:
        ok = _xz_extrusion(g, [(0, 0), (a, 0), (a, t1), (t2, t1), (t2, b), (0, b)], w)
    if not ok or not gussets:
        return ok
    gs = q(r.uniform(0.3, 0.7) * min(a - t2, b - t1))
    gt = q(r.uniform(3, min(12, w / 4)))
    tri = Profile(outer=polyline([(t2, t1), (t2 + gs, t1), (t2, t1 + gs)]))
    positions = [w / 2] if w < 70 or r.random() < 0.5 else [q(w * 0.2), q(w * 0.8)]
    for y in positions:
        _sketch_pad(g, axis_plane(NY, y), tri, gt, symmetric=True)
    return True


def gusset_bracket(g: Gen) -> bool:
    return l_bracket(g, gussets=True)


def u_channel(g: Gen) -> bool:
    r = g.rng
    a, b = q(r.uniform(40, 150)), q(r.uniform(25, 100))
    t = q(r.uniform(4, min(18, a / 5, b / 3)))
    w = q(r.uniform(30, 160))
    return _xz_extrusion(g, [(0, 0), (a, 0), (a, b), (a - t, b), (a - t, t), (t, t), (t, b), (0, b)], w)


def t_block(g: Gen) -> bool:
    r = g.rng
    a, b = q(r.uniform(50, 160)), q(r.uniform(30, 110))
    t1 = q(r.uniform(6, b / 2.5))
    s = q(r.uniform(8, a / 2.5))
    w = q(r.uniform(30, 140))
    x0, x1 = q(a / 2 - s / 2), q(a / 2 + s / 2)
    return _xz_extrusion(g, [(0, 0), (a, 0), (a, t1), (x1, t1), (x1, b), (x0, b), (x0, t1), (0, t1)], w)


# ---------------------------------------------------------------- turned --

def stepped_shaft(g: Gen) -> bool:
    r = g.rng
    n = r.randint(2, 5)
    radii = [q(r.uniform(6, 40))]
    for _ in range(n - 1):
        nxt = q(r.uniform(5, 40))
        while abs(nxt - radii[-1]) < 2:
            nxt = q(nxt + r.choice([-3, 3]))
        radii.append(max(nxt, 4.0))
    z = 0.0
    for rad in radii:
        length = q(r.uniform(8, 60))
        if not _sketch_pad(g, axis_plane(Z, z), circle_profile(0, 0, rad), length):
            return False
        z = q(z + length)
    if r.random() < 0.4:
        bore = q(r.uniform(2, min(radii) - 2.5))
        if bore >= 2:
            _defer_pocket(g, axis_plane(Z, z), circle_profile(0, 0, bore), through_all=True)
    g.hints["turned"] = True
    return True


def flange(g: Gen) -> bool:
    r = g.rng
    R = q(r.uniform(30, 110))
    t = q(r.uniform(6, 22))
    if not _sketch_pad(g, axis_plane(Z, 0), circle_profile(0, 0, R), t):
        return False
    hub = q(r.uniform(0.25, 0.55) * R)
    top = t + q(r.uniform(8, 50))
    if not _sketch_pad(g, axis_plane(Z, t), circle_profile(0, 0, hub), top - t):
        return False
    if r.random() < 0.35:  # spigot on the underside
        spig = q(r.uniform(hub, 0.8 * R))
        _sketch_pad(g, axis_plane(NZ, 0), circle_profile(0, 0, spig), q(r.uniform(3, 12)))
    if r.random() < 0.85:
        bore = q(r.uniform(0.35, 0.8) * hub)
        if hub - bore >= 3:
            _defer_pocket(g, axis_plane(Z, top), circle_profile(0, 0, bore), through_all=True)
    g.hints["turned"] = True
    g.hints["bolt_circle_face_z"] = t
    return True


def hub_disc(g: Gen) -> bool:
    """Wheel/pulley-like disc: thick rim + thin web + hub on both sides."""
    r = g.rng
    R = q(r.uniform(35, 110))
    web = q(r.uniform(5, 15))
    if not _sketch_pad(g, axis_plane(Z, 0), circle_profile(0, 0, R), web, symmetric=True):
        return False
    hub = q(r.uniform(0.2, 0.4) * R)
    hh = q(r.uniform(5, 30))
    _sketch_pad(g, axis_plane(Z, web / 2), circle_profile(0, 0, hub), hh)
    _sketch_pad(g, axis_plane(NZ, -web / 2), circle_profile(0, 0, hub), hh)
    top = web / 2 + hh
    if r.random() < 0.6:  # polygonal/round boss variants
        bh = q(r.uniform(2, 6))
        if _sketch_pad(g, axis_plane(Z, web / 2),
                       regular_polygon(0, 0, q(hub * 1.3), 6) if r.random() < 0.3 else
                       circle_profile(0, 0, q(hub * 1.3)), bh):
            top = max(top, web / 2 + bh)
    if r.random() < 0.8:
        bore = q(r.uniform(0.3, 0.7) * hub)
        _defer_pocket(g, axis_plane(Z, top), circle_profile(0, 0, bore), through_all=True)
    g.hints["turned"] = True
    return True


FAMILIES = {
    "plate": (plate, 0.15),
    "block": (block, 0.08),
    "l_bracket": (l_bracket, 0.12),
    "gusset_bracket": (gusset_bracket, 0.08),
    "u_channel": (u_channel, 0.10),
    "t_block": (t_block, 0.07),
    "stepped_shaft": (stepped_shaft, 0.14),
    "flange": (flange, 0.16),
    "hub_disc": (hub_disc, 0.10),
}
