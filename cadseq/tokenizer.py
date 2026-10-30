"""Compact token encoding of a CADSequence for sequence models.

Every number is quantized to ``Q`` = 0.05 mm (or 0.05°) and written as two tokens: a coarse
10 mm bucket ``#k`` and a fine 0.05 mm offset ``.k``. Range: [-510.0, 510.0).
The dataset generator snaps all geometry to this grid, so dataset sequences are exact.

Grammar (one line per feature; ``n`` = number = 2 tokens, ``axis`` in +X -X +Y -Y +Z -Z):

    <bos>
    SKETCH axis n_offset ( PROFILE LOOP curve+ ( LOOP curve+ )* )+
        curve := LINE n_x n_y | ARC n_x n_y n_mx n_my | CIRCLE n_cx n_cy n_r
    PAD     ( EXT+ | EXT- | EXT_SYM ) n_length                 # consumes the preceding SKETCH
    POCKET  ( EXT+ | EXT- ) ( THROUGH | n_depth )              # consumes the preceding SKETCH
    HOLE    ( H_SIMPLE | H_CBORE n_D n_t | H_CSINK n_D n_angle ) axis n_x n_y n_z n_d ( THROUGH | n_depth )
    FILLET  n_radius   ( E_LINE | E_CIRCLE | ... ) n_x n_y n_z  ( ... )*
    CHAMFER n_distance ( E_LINE | E_CIRCLE | ... ) n_x n_y n_z  ( ... )*
    MIRROR  axis n_offset
    <eos>

Canonicalization (so equal geometry → equal tokens as far as cheaply possible):
- Sketch planes are axis-aligned with origin on the axis (``datagen.geom.axis_plane`` rule:
  x_dir = +Y for ±X normals, +X otherwise); a sketch is immediately consumed by one PAD/POCKET.
- Lines/arcs store only their start point (end = next curve's start; loops are closed).
- Loops are re-oriented counter-clockwise and rotated to start at the lowest (x, y) vertex;
  inner loops and profiles are sorted; fillet/chamfer edges are sorted by point.
- Symmetric pads drop the direction; mirror normals use the positive axis.
Feature ids are regenerated canonically (S1, P1, K1, H1, ...) on decode.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .schema import (
    ID_PREFIX,
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
    Pocket,
    Profile,
    Sketch,
)

Q_PER_MM = 20  # quantization: 0.05 mm
NUM_OFFSET = 10200  # integer steps below zero → range [-510.0, 510.0)
NUM_FINE = 200  # fine bucket count (0.00 .. 9.95 mm in 0.05 steps)
NUM_COARSE = 2 * NUM_OFFSET // NUM_FINE  # 102 coarse buckets of 10 mm

AXES = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0),
        "+Z": (0, 0, 1), "-Z": (0, 0, -1)}
EDGE_TYPES = {"line": "E_LINE", "circle": "E_CIRCLE", "ellipse": "E_ELLIPSE",
              "bspline": "E_BSPLINE", "other": "E_OTHER"}
HOLE_KINDS = {"simple": "H_SIMPLE", "counterbore": "H_CBORE", "countersink": "H_CSINK"}

SPECIAL = ["<pad>", "<bos>", "<eos>"]
FEATURE_TOKENS = ["SKETCH", "PAD", "POCKET", "HOLE", "FILLET", "CHAMFER", "MIRROR"]
STRUCT = (FEATURE_TOKENS + ["PROFILE", "LOOP", "LINE", "ARC", "CIRCLE"] + list(AXES)
          + ["EXT+", "EXT-", "EXT_SYM", "THROUGH"] + list(HOLE_KINDS.values()) + list(EDGE_TYPES.values()))
NUMBER_TOKENS = [f"#{i}" for i in range(NUM_COARSE)] + [f".{i}" for i in range(NUM_FINE)]

VOCAB: list[str] = SPECIAL + STRUCT + NUMBER_TOKENS
TOKEN_ID: dict[str, int] = {t: i for i, t in enumerate(VOCAB)}
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
COARSE_BASE = TOKEN_ID["#0"]
FINE_BASE = TOKEN_ID[".0"]


class TokenizeError(ValueError):
    """The sequence cannot be represented (non-canonical plane, out-of-range number, ...)."""


class DecodeError(ValueError):
    """The token sequence is not grammatical / does not form a valid CADSequence."""


# ----------------------------------------------------------------- numbers --

def quantize(v: float) -> int:
    n = round(v * Q_PER_MM)
    if not -NUM_OFFSET <= n < NUM_OFFSET:
        raise TokenizeError(f"value {v} outside [-{NUM_OFFSET / Q_PER_MM}, {NUM_OFFSET / Q_PER_MM})")
    return n


def dequantize(n: int) -> float:
    return round(n / Q_PER_MM, 6) + 0.0


def _ntok(n: int) -> list[int]:
    """Already-quantized integer → [coarse, fine] token ids."""
    hi, lo = divmod(n + NUM_OFFSET, NUM_FINE)
    return [COARSE_BASE + hi, FINE_BASE + lo]


def _num_tokens(v: float) -> list[int]:
    return _ntok(quantize(v))


def _axis_name(v, positive_only=False) -> str:
    t = tuple(round(c, 9) + 0.0 for c in v)
    for name, a in AXES.items():
        if t == a:
            if positive_only and name[0] == "-":
                return "+" + name[1]
            return name
    raise TokenizeError(f"direction {v} is not axis-aligned")


def canonical_x_dir(axis) -> tuple[int, int, int]:
    return (0, 1, 0) if axis[0] else (1, 0, 0)


# ------------------------------------------------------------ loop canon. --

def _q2(p) -> tuple[int, int]:
    return quantize(p[0]), quantize(p[1])


def _canonical_loop(loop) -> list[tuple]:
    """Quantized loop as [('LINE', s) | ('ARC', s, m) | ('CIRCLE', c, r)], CCW from the min vertex."""
    if isinstance(loop[0], Circle):
        c = loop[0]
        return [("CIRCLE", _q2(c.center), quantize(c.radius))]
    curves = [("LINE", _q2(c.start), _q2(c.end)) if isinstance(c, Line)
              else ("ARC", _q2(c.start), _q2(c.end), _q2(c.mid)) for c in loop]
    pts = []
    for c in curves:
        pts.append(c[1])
        if c[0] == "ARC":
            pts.append(c[3])
    area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
    if area2 < 0:  # reverse orientation: swap start/end, keep arc mid
        curves = [(c[0], c[2], c[1], *c[3:]) for c in reversed(curves)]
    k = min(range(len(curves)), key=lambda i: curves[i][1])
    curves = curves[k:] + curves[:k]
    return [("LINE", c[1]) if c[0] == "LINE" else ("ARC", c[1], c[3]) for c in curves]


def _loop_key(loop) -> tuple:
    c = loop[0]
    return (c[1], c[0]) if c[0] != "CIRCLE" else (c[1], "CIRCLE", c[2])


# ------------------------------------------------------------------ encode --

def encode(seq: CADSequence, add_special: bool = True) -> list[int]:
    out: list[int] = [BOS_ID] if add_special else []
    T = TOKEN_ID
    feats = seq.features
    i = 0
    while i < len(feats):
        f = feats[i]
        if isinstance(f, Sketch):
            nxt = feats[i + 1] if i + 1 < len(feats) else None
            if not isinstance(nxt, (Pad, Pocket)) or nxt.sketch != f.id:
                raise TokenizeError(f"{f.id}: sketch must be immediately consumed by a pad/pocket")
            ax = _axis_name(f.plane.normal)
            if tuple(f.plane.x_dir) != canonical_x_dir(AXES[ax]):
                raise TokenizeError(f"{f.id}: non-canonical sketch x_dir {f.plane.x_dir}")
            a = AXES[ax]
            idx = [abs(c) for c in a].index(1)
            if any(abs(f.plane.origin[j]) > 1e-9 for j in range(3) if j != idx):
                raise TokenizeError(f"{f.id}: sketch origin must lie on the {ax[1]} axis")
            out += [T["SKETCH"], T[ax], *_num_tokens(f.plane.origin[idx])]
            profiles = []
            for p in f.profiles:
                outer = _canonical_loop(p.outer)
                inner = sorted((_canonical_loop(l) for l in p.inner), key=_loop_key)
                profiles.append([outer] + inner)
            profiles.sort(key=lambda p: _loop_key(p[0]))
            for loops in profiles:
                out.append(T["PROFILE"])
                for loop in loops:
                    out.append(T["LOOP"])
                    for c in loop:
                        out.append(T[c[0]])
                        if c[0] == "CIRCLE":
                            vals = [*c[1], c[2]]
                        else:
                            vals = [*c[1]] + ([*c[2]] if c[0] == "ARC" else [])
                        for n in vals:
                            out += _ntok(n)
            f = nxt
            i += 1
            if isinstance(f, Pad):
                ext = "EXT_SYM" if f.symmetric else ("EXT+" if f.direction == 1 else "EXT-")
                out += [T["PAD"], T[ext], *_num_tokens(f.length)]
            else:
                out += [T["POCKET"], T["EXT+" if f.direction == 1 else "EXT-"]]
                out += [T["THROUGH"]] if f.through_all else _num_tokens(f.depth)
        elif isinstance(f, (Pad, Pocket)):
            raise TokenizeError(f"{f.id}: pad/pocket without a directly preceding sketch")
        elif isinstance(f, Hole):
            out += [T["HOLE"], T[HOLE_KINDS[f.kind]]]
            if f.kind == "counterbore":
                out += _num_tokens(f.cb_diameter) + _num_tokens(f.cb_depth)
            elif f.kind == "countersink":
                out += _num_tokens(f.cs_diameter) + _num_tokens(f.cs_angle)
            out.append(T[_axis_name(f.direction)])
            for v in (*f.position, f.diameter):
                out += _num_tokens(v)
            out += [T["THROUGH"]] if f.through_all else _num_tokens(f.depth)
        elif isinstance(f, (Fillet, Chamfer)):
            out += [T[f.type.upper()], *_num_tokens(f.radius if isinstance(f, Fillet) else f.distance)]
            refs = sorted(((tuple(quantize(c) for c in e.point), e.curve) for e in f.edges))
            for pt, curve in refs:
                out.append(T[EDGE_TYPES[curve]])
                for n in pt:
                    out += _ntok(n)
        elif isinstance(f, Mirror):
            ax = _axis_name(f.plane.normal, positive_only=True)
            idx = "XYZ".index(ax[1])
            if any(abs(f.plane.origin[j]) > 1e-9 for j in range(3) if j != idx):
                raise TokenizeError(f"{f.id}: mirror origin must lie on the {ax[1]} axis")
            out += [T["MIRROR"], T[ax], *_num_tokens(f.plane.origin[idx])]
        i += 1
    if add_special:
        out.append(EOS_ID)
    return out


# ------------------------------------------------------------------ decode --

@dataclass
class _Reader:
    ids: list[int]
    pos: int = 0

    def peek(self) -> str | None:
        return VOCAB[self.ids[self.pos]] if self.pos < len(self.ids) else None

    def take(self, *allowed: str) -> str:
        t = self.peek()
        if t is None or (allowed and t not in allowed):
            raise DecodeError(f"pos {self.pos}: expected {allowed or 'token'}, got {t}")
        self.pos += 1
        return t

    def num(self) -> float:
        if self.pos + 1 >= len(self.ids):
            raise DecodeError(f"pos {self.pos}: truncated number")
        hi, lo = self.ids[self.pos] - COARSE_BASE, self.ids[self.pos + 1] - FINE_BASE
        if not (0 <= hi < NUM_COARSE and 0 <= lo < NUM_FINE):
            raise DecodeError(f"pos {self.pos}: expected number, got {VOCAB[self.ids[self.pos]]} "
                              f"{VOCAB[self.ids[self.pos + 1]]}")
        self.pos += 2
        return dequantize(hi * NUM_FINE + lo - NUM_OFFSET)

    def pt(self, k: int) -> tuple[float, ...]:
        return tuple(self.num() for _ in range(k))


def _decode_loop(r: _Reader):
    if r.peek() == "CIRCLE":
        r.take()
        c = r.pt(2)
        return [Circle(type="circle", center=c, radius=r.num())]
    raw = []
    while r.peek() in ("LINE", "ARC"):
        kind = r.take()
        raw.append((kind, r.pt(2), r.pt(2) if kind == "ARC" else None))
    if not raw:
        raise DecodeError(f"pos {r.pos}: empty loop")
    curves = []
    for k, (kind, start, mid) in enumerate(raw):
        end = raw[(k + 1) % len(raw)][1]
        curves.append(Line(type="line", start=start, end=end) if kind == "LINE"
                      else Arc(type="arc", start=start, mid=mid, end=end))
    return curves


def decode(ids, strict_special: bool = True) -> CADSequence:
    """Tokens → CADSequence (canonical ids). Raises DecodeError on any grammar/validation error."""
    ids = list(ids)
    if strict_special:
        if not ids or ids[0] != BOS_ID:
            raise DecodeError("missing <bos>")
        if EOS_ID not in ids:
            raise DecodeError("missing <eos>")
        ids = ids[1:ids.index(EOS_ID)]
    r = _Reader(ids)
    counters: Counter = Counter()

    def new_id(kind: str) -> str:
        counters[kind] += 1
        return f"{ID_PREFIX[kind]}{counters[kind]}"

    feats: list[dict] = []
    while r.peek() is not None:
        t = r.take(*FEATURE_TOKENS)
        if t == "SKETCH":
            ax = r.take(*AXES)
            a = AXES[ax]
            off = r.num()
            origin = tuple(off if c else 0.0 for c in a)
            profiles = []
            while r.peek() == "PROFILE":
                r.take()
                loops = []
                while r.peek() == "LOOP":
                    r.take()
                    loops.append(_decode_loop(r))
                if not loops:
                    raise DecodeError(f"pos {r.pos}: profile without loops")
                profiles.append(Profile(outer=loops[0], inner=loops[1:]))
            if not profiles:
                raise DecodeError(f"pos {r.pos}: sketch without profiles")
            sid = new_id("sketch")
            feats.append({"id": sid, "type": "sketch", "profiles": profiles,
                          "plane": {"origin": origin, "normal": a, "x_dir": canonical_x_dir(a)}})
            op = r.take("PAD", "POCKET")
            if op == "PAD":
                ext = r.take("EXT+", "EXT-", "EXT_SYM")
                feats.append({"id": new_id("pad"), "type": "pad", "sketch": sid, "length": r.num(),
                              "direction": -1 if ext == "EXT-" else 1, "symmetric": ext == "EXT_SYM"})
            else:
                ext = r.take("EXT+", "EXT-")
                f = {"id": new_id("pocket"), "type": "pocket", "sketch": sid,
                     "direction": 1 if ext == "EXT+" else -1}
                if r.peek() == "THROUGH":
                    r.take()
                    f["through_all"] = True
                else:
                    f["depth"] = r.num()
                feats.append(f)
        elif t in ("PAD", "POCKET"):
            raise DecodeError(f"pos {r.pos}: {t} without a preceding SKETCH")
        elif t == "HOLE":
            kind_tok = r.take(*HOLE_KINDS.values())
            kind = {v: k for k, v in HOLE_KINDS.items()}[kind_tok]
            f = {"id": new_id("hole"), "type": "hole", "kind": kind}
            if kind == "counterbore":
                f["cb_diameter"], f["cb_depth"] = r.num(), r.num()
            elif kind == "countersink":
                f["cs_diameter"], f["cs_angle"] = r.num(), r.num()
            f["direction"] = AXES[r.take(*AXES)]
            f["position"] = r.pt(3)
            f["diameter"] = r.num()
            if r.peek() == "THROUGH":
                r.take()
                f["through_all"] = True
            else:
                f["depth"] = r.num()
            feats.append(f)
        elif t in ("FILLET", "CHAMFER"):
            size = r.num()
            edges = []
            inv = {v: k for k, v in EDGE_TYPES.items()}
            while r.peek() in inv:
                curve = inv[r.take()]
                edges.append(EdgeRef(point=r.pt(3), curve=curve))
            kind = t.lower()
            feats.append({"id": new_id(kind), "type": kind, "edges": edges,
                          ("radius" if kind == "fillet" else "distance"): size})
        elif t == "MIRROR":
            a = AXES[r.take("+X", "+Y", "+Z")]
            off = r.num()
            feats.append({"id": new_id("mirror"), "type": "mirror",
                          "plane": {"origin": tuple(off if c else 0.0 for c in a), "normal": a}})
    try:
        return CADSequence.model_validate({"features": feats})
    except Exception as e:  # pydantic ValidationError: degenerate values, missing pad, ...
        raise DecodeError(f"invalid sequence: {e}") from e


# ------------------------------------------------------------------- utils --

def canonicalize(seq: CADSequence) -> CADSequence:
    """Quantized, canonical form of ``seq`` (what a perfect model would output)."""
    return decode(encode(seq))


def to_text(ids) -> str:
    """Human-readable tokens; numbers are shown as their value, one feature per line."""
    out, i, ids = [], 0, list(ids)
    while i < len(ids):
        t = VOCAB[ids[i]]
        if t.startswith("#") and i + 1 < len(ids) and VOCAB[ids[i + 1]].startswith("."):
            hi, lo = ids[i] - COARSE_BASE, ids[i + 1] - FINE_BASE
            out.append(f"{dequantize(hi * NUM_FINE + lo - NUM_OFFSET):g}")
            i += 2
            continue
        out.append(("\n" if t in FEATURE_TOKENS or t == "<eos>" else "") + t)
        i += 1
    return " ".join(out).replace(" \n", "\n")
