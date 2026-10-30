"""Pydantic models for the JSON design-history format (schema v1.0).

All lengths are in millimetres, angles in degrees. 2-D sketch coordinates are
expressed in the sketch plane's local frame: local X = ``x_dir``, local Y =
``normal x x_dir``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
CLOSURE_TOL = 1e-6
# Canonical feature-id prefixes: ids are <prefix><n>, numbered per prefix in sequence order.
ID_PREFIX = {"sketch": "S", "pad": "P", "pocket": "K", "hole": "H",
             "fillet": "F", "chamfer": "C", "mirror": "M"}

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Direction = Literal[1, -1]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _norm(v: Vec3) -> float:
    return math.sqrt(sum(c * c for c in v))


# ---------------------------------------------------------------- geometry --

class Plane(_Model):
    origin: Vec3
    normal: Vec3
    x_dir: Vec3

    @model_validator(mode="after")
    def _check(self) -> "Plane":
        if _norm(self.normal) < 1e-9 or _norm(self.x_dir) < 1e-9:
            raise ValueError("plane normal / x_dir must be non-zero")
        dot = sum(a * b for a, b in zip(self.normal, self.x_dir))
        if abs(dot) / (_norm(self.normal) * _norm(self.x_dir)) > 1e-6:
            raise ValueError("plane x_dir must be perpendicular to normal")
        return self


class MirrorPlane(_Model):
    origin: Vec3
    normal: Vec3


class Line(_Model):
    type: Literal["line"]
    start: Vec2
    end: Vec2


class Arc(_Model):
    """Three-point arc: start -> mid -> end."""

    type: Literal["arc"]
    start: Vec2
    mid: Vec2
    end: Vec2


class Circle(_Model):
    type: Literal["circle"]
    center: Vec2
    radius: float = Field(gt=0)


Curve = Annotated[Union[Line, Arc, Circle], Field(discriminator="type")]


def _check_loop(loop: list[Curve]) -> None:
    if not loop:
        raise ValueError("empty loop")
    if any(isinstance(c, Circle) for c in loop):
        if len(loop) != 1:
            raise ValueError("a circle must be the only curve in its loop")
        return
    for a, b in zip(loop, loop[1:] + loop[:1]):
        if math.dist(a.end, b.start) > CLOSURE_TOL:
            raise ValueError(f"loop not closed/continuous between {a.end} and {b.start}")


class Profile(_Model):
    """A closed region: one outer loop and optional inner loops (islands removed)."""

    outer: list[Curve]
    inner: list[list[Curve]] = []

    @model_validator(mode="after")
    def _check(self) -> "Profile":
        _check_loop(self.outer)
        for loop in self.inner:
            _check_loop(loop)
        return self


class EdgeRef(_Model):
    """Topology-independent edge reference: a point on the edge + its curve type.

    Rebuilders resolve it to the nearest edge of matching type in the current body.
    """

    point: Vec3
    curve: Literal["line", "circle", "ellipse", "bspline", "other"]


# ---------------------------------------------------------------- features --

class Sketch(_Model):
    id: str
    type: Literal["sketch"]
    plane: Plane
    profiles: list[Profile] = Field(min_length=1)


class Pad(_Model):
    """Extrude a sketch and add material. ``direction`` is relative to the sketch normal."""

    id: str
    type: Literal["pad"]
    sketch: str
    length: float = Field(gt=0)
    direction: Direction = 1
    symmetric: bool = False


class Pocket(_Model):
    """Extrude a sketch and remove material. ``direction`` is relative to the sketch normal."""

    id: str
    type: Literal["pocket"]
    sketch: str
    depth: Optional[float] = Field(default=None, gt=0)
    through_all: bool = False
    direction: Direction = -1

    @model_validator(mode="after")
    def _check(self) -> "Pocket":
        if self.through_all == (self.depth is not None):
            raise ValueError("pocket needs exactly one of depth / through_all")
        return self


class Hole(_Model):
    """Drilled hole. ``position`` lies on the entry face, ``direction`` points into material."""

    id: str
    type: Literal["hole"]
    kind: Literal["simple", "counterbore", "countersink"] = "simple"
    position: Vec3
    direction: Vec3
    diameter: float = Field(gt=0)
    depth: Optional[float] = Field(default=None, gt=0)
    through_all: bool = False
    cb_diameter: Optional[float] = Field(default=None, gt=0)
    cb_depth: Optional[float] = Field(default=None, gt=0)
    cs_diameter: Optional[float] = Field(default=None, gt=0)
    cs_angle: Optional[float] = Field(default=None, gt=0, lt=180)

    @model_validator(mode="after")
    def _check(self) -> "Hole":
        if self.through_all == (self.depth is not None):
            raise ValueError("hole needs exactly one of depth / through_all")
        if _norm(self.direction) < 1e-9:
            raise ValueError("hole direction must be non-zero")
        if self.kind == "counterbore":
            if self.cb_diameter is None or self.cb_depth is None:
                raise ValueError("counterbore needs cb_diameter and cb_depth")
            if self.cb_diameter <= self.diameter:
                raise ValueError("cb_diameter must exceed diameter")
        if self.kind == "countersink":
            if self.cs_diameter is None or self.cs_angle is None:
                raise ValueError("countersink needs cs_diameter and cs_angle")
            if self.cs_diameter <= self.diameter:
                raise ValueError("cs_diameter must exceed diameter")
        return self


class Fillet(_Model):
    id: str
    type: Literal["fillet"]
    radius: float = Field(gt=0)
    edges: list[EdgeRef] = Field(min_length=1)


class Chamfer(_Model):
    """Equal-distance chamfer."""

    id: str
    type: Literal["chamfer"]
    distance: float = Field(gt=0)
    edges: list[EdgeRef] = Field(min_length=1)


class Mirror(_Model):
    """Mirror the whole current body about a plane and fuse with the original."""

    id: str
    type: Literal["mirror"]
    plane: MirrorPlane


Feature = Annotated[
    Union[Sketch, Pad, Pocket, Hole, Fillet, Chamfer, Mirror],
    Field(discriminator="type"),
]


class CADSequence(_Model):
    version: Literal["1.0"] = SCHEMA_VERSION
    units: Literal["mm"] = "mm"
    features: list[Feature] = Field(min_length=2)

    @model_validator(mode="after")
    def _check(self) -> "CADSequence":
        seen: dict[str, str] = {}
        has_body = False
        for f in self.features:
            if f.id in seen:
                raise ValueError(f"duplicate feature id {f.id!r}")
            if isinstance(f, (Pad, Pocket)):
                if seen.get(f.sketch) != "sketch":
                    raise ValueError(f"{f.id}: sketch {f.sketch!r} must be an earlier sketch")
            if isinstance(f, Pad):
                has_body = True
            elif not isinstance(f, Sketch) and not has_body:
                raise ValueError(f"{f.id}: first solid feature must be a pad")
            seen[f.id] = f.type
        if not has_body:
            raise ValueError("sequence contains no pad")
        return self


def load_sequence(path: str | Path) -> CADSequence:
    return CADSequence.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
