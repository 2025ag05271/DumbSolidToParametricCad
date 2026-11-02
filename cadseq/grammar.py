"""Incremental grammar for the token format: which tokens may come next.

Used for constrained decoding (invalid tokens are masked out of the logits) and to validate
token sequences. Implemented as a coroutine that mirrors ``tokenizer.decode``:

    g = Grammar()
    g.allowed          # frozenset of token ids permitted next
    g.feed(tok)        # advance (raises GrammarError if tok is not allowed)
    g.done             # True after <eos>

Value constraints enforced on the fly: lengths, depths, radii, diameters and angles must be > 0;
non-circle loops need >= 2 curves. With ``canonical_order=True`` (default) features must follow the
dataset's canonical stage order: pads → pockets → holes → fillets → chamfers → [mirror] → <eos>.
"""

from __future__ import annotations

from functools import lru_cache

from .tokenizer import (
    AXES,
    BOS_ID,
    COARSE_BASE,
    EDGE_TYPES,
    EOS_ID,
    FINE_BASE,
    HOLE_KINDS,
    NUM_COARSE,
    NUM_FINE,
    NUM_OFFSET,
    TOKEN_ID,
)

T = TOKEN_ID
_ZERO_COARSE, _ZERO_FINE = divmod(NUM_OFFSET, NUM_FINE)

COARSE = frozenset(range(COARSE_BASE, COARSE_BASE + NUM_COARSE))
POS_COARSE = frozenset(range(COARSE_BASE + _ZERO_COARSE, COARSE_BASE + NUM_COARSE))
FINE = frozenset(range(FINE_BASE, FINE_BASE + NUM_FINE))
POS_FINE_AT_ZERO = frozenset(range(FINE_BASE + _ZERO_FINE + 1, FINE_BASE + NUM_FINE))
AXIS6 = frozenset(T[a] for a in AXES)
AXIS_POS = frozenset(T[a] for a in ("+X", "+Y", "+Z"))
CURVE_FIRST = frozenset({T["LINE"], T["ARC"], T["CIRCLE"]})
CURVE_NEXT = frozenset({T["LINE"], T["ARC"]})
EDGE_TOKS = frozenset(T[e] for e in EDGE_TYPES.values())
HOLE_TOKS = frozenset(T[h] for h in HOLE_KINDS.values())
_INV_EDGE = {T[e] for e in EDGE_TYPES.values()}

# stage of each solid feature in the canonical order
_STAGE = {"PAD": 0, "POCKET": 1, "HOLE": 2, "FILLET": 3, "CHAMFER": 4, "MIRROR": 5}


class GrammarError(ValueError):
    pass


ANY = (-NUM_OFFSET, NUM_OFFSET - 1)  # quantized number range (inclusive)
POS = (1, NUM_OFFSET - 1)


@lru_cache(maxsize=None)
def _coarse_set(lo: int, hi: int) -> frozenset:
    k0, k1 = (lo + NUM_OFFSET) // NUM_FINE, (hi + NUM_OFFSET) // NUM_FINE
    return frozenset(range(COARSE_BASE + k0, COARSE_BASE + k1 + 1))


@lru_cache(maxsize=None)
def _fine_set(k: int, lo: int, hi: int) -> frozenset:
    base = k * NUM_FINE - NUM_OFFSET
    f0, f1 = max(0, lo - base), min(NUM_FINE - 1, hi - base)
    return frozenset(range(FINE_BASE + f0, FINE_BASE + f1 + 1))


def _number(state: dict, lo: int, hi: int, extra: frozenset = frozenset()):
    """Yield the allowed sets of one number with quantized value in [lo, hi]; return its value
    (None if a token of ``extra``, e.g. THROUGH, was chosen instead). While the number is about to
    start, ``state["range"]`` exposes (lo, hi) to the copy head."""
    if hi < lo:  # impossible bound (degenerate earlier value): do not dead-end the sequence
        lo, hi = ANY
    state["range"] = (lo, hi)
    c = yield _coarse_set(lo, hi) | extra
    state["range"] = None
    if c in extra:
        return None
    k = c - COARSE_BASE
    f = yield _fine_set(k, lo, hi)
    return k * NUM_FINE + (f - FINE_BASE) - NUM_OFFSET


def _program(canonical_order: bool, state: dict):
    yield frozenset({BOS_ID})
    stage, has_body, mirrored = 0, False, False
    tok = None
    while True:
        if tok is None:  # need the start of the next feature
            starts = set()
            if not mirrored or not canonical_order:
                if not canonical_order or stage <= 1:
                    starts.add(T["SKETCH"])
                if has_body:
                    for name in ("HOLE", "FILLET", "CHAMFER", "MIRROR"):
                        if not canonical_order or stage <= _STAGE[name]:
                            starts.add(T[name])
            if has_body:
                starts.add(EOS_ID)
            tok = yield frozenset(starts)
        if tok == EOS_ID:
            return

        if tok == T["SKETCH"]:
            yield AXIS6
            yield from _number(state, *ANY)
            yield frozenset({T["PROFILE"]})
            # profiles → loops → curves; exits on PAD / POCKET
            ops = set()
            if not has_body:
                ops.add(T["PAD"])
            else:
                if not canonical_order or stage == 0:
                    ops.add(T["PAD"])
                ops.add(T["POCKET"])
            ops = frozenset(ops)
            t = yield frozenset({T["LOOP"]})  # first loop of first profile
            while True:  # t is LOOP here
                c = yield CURVE_FIRST
                n = 0
                while True:
                    if c == T["CIRCLE"]:
                        yield from _number(state, *ANY)
                        yield from _number(state, *ANY)
                        yield from _number(state, *POS)
                        nxt = yield frozenset({T["LOOP"], T["PROFILE"]}) | ops
                        break
                    yield from _number(state, *ANY)
                    yield from _number(state, *ANY)
                    if c == T["ARC"]:
                        yield from _number(state, *ANY)
                        yield from _number(state, *ANY)
                    n += 1
                    allowed = set(CURVE_NEXT)
                    if n >= 2:
                        allowed |= {T["LOOP"], T["PROFILE"]} | ops
                    c = yield frozenset(allowed)
                    if c not in CURVE_NEXT:
                        nxt = c
                        break
                if nxt == T["PROFILE"]:
                    t = yield frozenset({T["LOOP"]})
                    continue
                if nxt == T["LOOP"]:
                    continue
                op = nxt
                break
            if op == T["PAD"]:
                yield frozenset({T["EXT+"], T["EXT-"], T["EXT_SYM"]})
                yield from _number(state, *POS)
                has_body = True
                stage = max(stage, 0)
            else:
                yield frozenset({T["EXT+"], T["EXT-"]})
                yield from _number(state, *POS, extra=frozenset({T["THROUGH"]}))
                stage = max(stage, 1)
            tok = None

        elif tok == T["HOLE"]:
            kind = yield HOLE_TOKS
            head = None
            if kind in (T["H_CBORE"], T["H_CSINK"]):
                head = yield from _number(state, *POS)  # counterbore / countersink diameter
                yield from _number(state, *POS)
            yield AXIS6
            for _ in range(3):
                yield from _number(state, *ANY)
            # hole diameter must be smaller than the head diameter (schema rule)
            yield from _number(state, 1, head - 1 if head is not None else POS[1])
            yield from _number(state, *POS, extra=frozenset({T["THROUGH"]}))
            stage = max(stage, 2)
            tok = None

        elif tok in (T["FILLET"], T["CHAMFER"]):
            yield from _number(state, *POS)
            yield EDGE_TOKS
            stage = max(stage, _STAGE["FILLET" if tok == T["FILLET"] else "CHAMFER"])
            while True:
                for _ in range(3):
                    yield from _number(state, *ANY)
                # another edge, or the start of the next feature
                starts = set(EDGE_TOKS)
                if not canonical_order or stage <= 1:
                    starts.add(T["SKETCH"])
                for name in ("HOLE", "FILLET", "CHAMFER", "MIRROR"):
                    if not canonical_order or stage <= _STAGE[name]:
                        starts.add(T[name])
                starts.add(EOS_ID)
                t = yield frozenset(starts)
                if t not in _INV_EDGE:
                    tok = t
                    break

        elif tok == T["MIRROR"]:
            yield AXIS_POS
            yield from _number(state, *ANY)
            mirrored = True
            stage = 5
            tok = None
        else:  # pragma: no cover - guarded by the allowed sets
            raise GrammarError(f"unexpected token {tok}")


class Grammar:
    def __init__(self, canonical_order: bool = True):
        self.state: dict = {"range": None}
        self._gen = _program(canonical_order, self.state)
        self.allowed: frozenset = next(self._gen)
        self.done = False

    @property
    def number_range(self):
        """(lo, hi) quantized value bounds if a number may start next, else None."""
        return self.state["range"]

    def feed(self, tok: int) -> None:
        if self.done:
            raise GrammarError("sequence already finished")
        if tok not in self.allowed:
            raise GrammarError(f"token {tok} not allowed here")
        try:
            self.allowed = self._gen.send(tok)
        except StopIteration:
            self.done = True
            self.allowed = frozenset()


def check(ids, canonical_order: bool = True) -> int:
    """Return the index of the first disallowed token, or -1 if the whole sequence is valid."""
    g = Grammar(canonical_order)
    for i, t in enumerate(ids):
        if t not in g.allowed:
            return i
        g.feed(t)
    return -1 if g.done else len(ids)
