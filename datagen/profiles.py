"""2-D sketch profile builders (local sketch coordinates) returning schema models."""

from __future__ import annotations

import math

from cadseq.schema import Arc, Circle, Line, Profile
from cadseq.tokenizer import Q_PER_MM


def snap(x: float) -> float:
    """Snap to the tokenizer grid (0.05 mm) so every generated value tokenizes exactly."""
    return round(round(float(x) * Q_PER_MM) / Q_PER_MM, 6) + 0.0  # + 0.0 turns -0.0 into 0.0


def q(x: float, step: float = 0.5) -> float:
    """Quantize a dimension to a 'designer-friendly' grid."""
    return snap(round(x / step) * step)


def _p(pt) -> tuple[float, float]:
    return (snap(pt[0]), snap(pt[1]))


def line(a, b) -> Line:
    return Line(type="line", start=_p(a), end=_p(b))


def arc(a, m, b) -> Arc:
    return Arc(type="arc", start=_p(a), mid=_p(m), end=_p(b))


def circle(c, r) -> Circle:
    return Circle(type="circle", center=_p(c), radius=snap(r))


def polyline(pts) -> list[Line]:
    return [line(a, b) for a, b in zip(pts, pts[1:] + pts[:1])]


# --------------------------------------------------------------- profiles --

def rect(cx, cy, w, h) -> Profile:
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return Profile(outer=polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))


def rounded_rect(cx, cy, w, h, r) -> Profile:
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    k = r * (1 - 1 / math.sqrt(2))
    return Profile(outer=[
        line((x0 + r, y0), (x1 - r, y0)),
        arc((x1 - r, y0), (x1 - k, y0 + k), (x1, y0 + r)),
        line((x1, y0 + r), (x1, y1 - r)),
        arc((x1, y1 - r), (x1 - k, y1 - k), (x1 - r, y1)),
        line((x1 - r, y1), (x0 + r, y1)),
        arc((x0 + r, y1), (x0 + k, y1 - k), (x0, y1 - r)),
        line((x0, y1 - r), (x0, y0 + r)),
        arc((x0, y0 + r), (x0 + k, y0 + k), (x0 + r, y0)),
    ])


def chamfered_rect(cx, cy, w, h, c) -> Profile:
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return Profile(outer=polyline([
        (x0 + c, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1 - c),
        (x1 - c, y1), (x0 + c, y1), (x0, y1 - c), (x0, y0 + c),
    ]))


def circle_profile(cx, cy, r) -> Profile:
    return Profile(outer=[circle((cx, cy), r)])


def slot(cx, cy, length, width, vertical=False) -> Profile:
    """Obround slot; ``length`` is the overall length (> width)."""
    s, r = (length - width) / 2, width / 2

    def t(x, y):  # rotate 90° about the centre when vertical
        return (cx - (y - cy), cy + (x - cx)) if vertical else (x, y)

    return Profile(outer=[
        line(t(cx - s, cy - r), t(cx + s, cy - r)),
        arc(t(cx + s, cy - r), t(cx + s + r, cy), t(cx + s, cy + r)),
        line(t(cx + s, cy + r), t(cx - s, cy + r)),
        arc(t(cx - s, cy + r), t(cx - s - r, cy), t(cx - s, cy - r)),
    ])


def regular_polygon(cx, cy, radius, n, rot_deg=0.0) -> Profile:
    a0 = math.radians(rot_deg)
    pts = [(cx + radius * math.cos(a0 + 2 * math.pi * i / n),
            cy + radius * math.sin(a0 + 2 * math.pi * i / n)) for i in range(n)]
    return Profile(outer=polyline(pts))
