"""Generation context: incremental build with atomic try/rollback per feature group."""

from __future__ import annotations

import random
from collections import Counter

from cadseq.build import QUANTIZED_EDGE_TOL, Builder, BuildError
from cadseq.schema import ID_PREFIX, CADSequence, Pad, Pocket

MAX_FACES = 200
MAX_FEATURES = 48
MIN_EDGE_LENGTH = 0.2  # mm; shorter edges indicate sliver geometry (near-coincident features)


class Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.b = Builder(check_each=True, edge_tol=QUANTIZED_EDGE_TOL)
        self.features: list = []
        self.stats: Counter = Counter()
        self.hints: dict = {}
        self.deferred_pockets: list = []  # family-specific pockets run at the pocket stage
        self._n = 0

    @property
    def body(self):
        return self.b.body

    def tid(self, prefix: str) -> str:
        self._n += 1
        return f"_{prefix}{self._n}"

    def full(self) -> bool:
        return len(self.features) >= MAX_FEATURES

    def try_add(self, feats: list) -> bool:
        """Apply a group atomically. Rejects failures and no-op / runaway results."""
        if len(self.features) + len(feats) > MAX_FEATURES:
            return False
        kind = feats[-1].type
        snap = self.b.snapshot()
        vol0 = self.body.Volume() if self.body is not None else 0.0
        try:
            for f in feats:
                self.b.apply(f)
            vol1 = self.body.Volume()
            dv = vol1 - vol0
            tol = 1e-3
            if kind in ("pad", "mirror") and dv <= tol:
                raise BuildError("did not add material")
            if kind in ("pocket", "hole") and dv >= -tol:
                raise BuildError("did not remove material")
            if kind in ("fillet", "chamfer") and abs(dv) <= 1e-6:
                raise BuildError("no effect")
            if len(self.body.Faces()) > MAX_FACES:
                raise BuildError("too many faces")
            if min(e.Length() for e in self.body.Edges()) < MIN_EDGE_LENGTH:
                raise BuildError("sliver geometry")
        except BuildError:
            self.b.restore(snap)
            self.stats[f"reject_{kind}"] += 1
            return False
        self.features.extend(feats)
        self.stats[f"ok_{kind}"] += 1
        return True

    def finalize(self) -> CADSequence:
        """Renumber ids canonically (S1, P1, K1, H1, ...) and validate."""
        counters: Counter = Counter()
        mapping: dict[str, str] = {}
        out = []
        for f in self.features:
            p = ID_PREFIX[f.type]
            counters[p] += 1
            mapping[f.id] = f"{p}{counters[p]}"
            upd = {"id": mapping[f.id]}
            if isinstance(f, (Pad, Pocket)):
                upd["sketch"] = mapping[f.sketch]
            out.append(f.model_copy(update=upd))
        return CADSequence.model_validate({"features": [f.model_dump() for f in out]})
