"""Generate the synthetic dataset in parallel (resumable).

Usage:
    python -m datagen.run --out ../data/synth_v1 --count 20000 --workers 8

Layout of <out>:
    step/<shard>/<id>.step    dumb solid (input)
    json/<shard>/<id>.json    design history (target)
    graph/<shard>/<id>.npz    UV-Net arrays: node_x (F,N,N,7), edge_index (2,E), edge_x (E,N,6)
    index.jsonl               one metadata record per sample
    summary.json              dataset statistics
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pipe, Process
from multiprocessing.connection import wait
from pathlib import Path

TASKS_PER_WORKER = 150  # recycle workers to bound OCC memory growth
MAX_REQUEUES = 2  # after a crash/timeout, retry the slot with a fresh seed range


def _worker_loop(conn, out_dir: Path, base_seed: int, uv_n: int) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    from datagen.sample import make_sample

    while True:
        task = conn.recv()
        if task is None:
            return
        index, offset = task
        conn.send(make_sample(index, out_dir, base_seed, uv_n, attempt_offset=offset))


class _Slot:
    """One worker process; killed and replaced on crash, timeout or recycling."""

    def __init__(self, args):
        self.args = args
        self.proc = None
        self.conn = None
        self.task = None
        self.started = 0.0
        self.count = 0
        self.spawn()

    def spawn(self):
        parent, child = Pipe()
        self.proc = Process(target=_worker_loop, args=(child, *self.args), daemon=True)
        self.proc.start()
        child.close()
        self.conn, self.count = parent, 0

    def kill(self):
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(5)
        self.conn.close()

    def submit(self, task):
        if self.count >= TASKS_PER_WORKER:
            self.conn.send(None)
            self.proc.join(10)
            self.kill()
            self.spawn()
        self.task, self.started = task, time.time()
        self.count += 1
        self.conn.send(task)


def _done_ids(out_dir: Path) -> set[str]:
    idx = out_dir / "index.jsonl"
    if not idx.exists():
        return set()
    done = set()
    for line in idx.read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            if not m.get("failed"):
                done.add(m["id"])
    return done


def _remove_orphans(out_dir: Path) -> None:
    """Delete STEP/graph files left by crashed or rejected attempts (no matching JSON)."""
    done = {p.stem for p in (out_dir / "json").rglob("*.json")}
    for sub, ext in (("step", "*.step"), ("graph", "*.npz")):
        for p in (out_dir / sub).rglob(ext):
            if p.stem not in done:
                p.unlink()


def summarize_dataset(out_dir: Path) -> dict:
    recs = [json.loads(l) for l in (out_dir / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = {r["id"]: r for r in recs if not r.get("failed")}.values()
    ok = list(ok)
    ok_ids = {r["id"] for r in ok}
    failed = len({r["id"] for r in recs if r.get("failed")} - ok_ids)

    def pct(vals, p):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(p / 100 * len(vals)))] if vals else None

    ops, holes = Counter(), Counter()
    for r in ok:
        ops.update(r["ops"])
        holes.update(r.get("hole_kinds", {}))
    nf = [r["n_features"] for r in ok]
    fc = [r["faces"] for r in ok]
    return {
        "samples": len(ok),
        "failed_slots": failed,
        "families": dict(Counter(r["family"] for r in ok).most_common()),
        "ops_total": dict(ops.most_common()),
        "samples_with_op": {k: sum(1 for r in ok if r["ops"].get(k)) for k in ops},
        "hole_kinds": dict(holes),
        "n_features": {"min": min(nf), "p50": pct(nf, 50), "p90": pct(nf, 90), "max": max(nf)} if nf else {},
        "faces": {"min": min(fc), "p50": pct(fc, 50), "p90": pct(fc, 90), "max": max(fc)} if fc else {},
        "empty_mask_faces_total": sum(r["empty_mask_faces"] for r in ok),
        "mean_seconds": round(sum(r["seconds"] for r in ok) / max(1, len(ok)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0, help="base seed (change for a disjoint dataset)")
    ap.add_argument("--uv", type=int, default=10, help="UV-grid resolution")
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds per sample before the worker is killed")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    done = _done_ids(args.out)
    queue = [(i, 0) for i in range(args.start, args.start + args.count) if f"{i:08d}" not in done]
    total = len(queue)
    queue.reverse()  # pop() from the end → ascending ids
    print(f"{len(done)} already done, generating {total} with {args.workers} workers", flush=True)

    t0, n_ok, n_fail, n_crash, k = time.time(), 0, 0, 0, 0
    slots = [_Slot((args.out.resolve(), args.seed, args.uv)) for _ in range(min(args.workers, total))]
    with open(args.out / "index.jsonl", "a", encoding="utf-8") as idx:

        def record(rec):
            nonlocal n_ok, n_fail, k
            idx.write(json.dumps(rec) + "\n")
            k += 1
            n_fail += bool(rec.get("failed"))
            n_ok += not rec.get("failed")
            if k % 100 == 0 or k == total:
                idx.flush()
                el = time.time() - t0
                print(f"[{k}/{total}] ok={n_ok} failed={n_fail} crashes/timeouts={n_crash} "
                      f"{k / el:.1f}/s eta {el / k * (total - k) / 60:.1f} min", flush=True)

        for s in slots:
            if queue:
                s.submit(queue.pop())
        while any(s.task for s in slots):
            busy = [s for s in slots if s.task]
            ready = wait([s.conn for s in busy] + [s.proc.sentinel for s in busy], timeout=2.0)
            for s in busy:
                if s.conn in ready:
                    try:
                        rec = s.conn.recv()
                    except (EOFError, OSError):
                        rec = None
                    if rec is not None:
                        record(rec)
                        s.task = None
                if s.task and (not s.proc.is_alive() or time.time() - s.started > args.timeout):
                    n_crash += 1
                    index, offset = s.task
                    reason = "timeout" if s.proc.is_alive() else f"worker crashed (exit {s.proc.exitcode})"
                    s.kill()
                    s.spawn()
                    s.task = None
                    if offset // 6 < MAX_REQUEUES:
                        queue.append((index, offset + 6))  # retry immediately with fresh seeds
                    else:
                        record({"id": f"{index:08d}", "failed": True, "errors": [reason]})
                if s.task is None and queue:
                    s.submit(queue.pop())
        for s in slots:
            try:
                s.conn.send(None)
            except OSError:
                pass
            s.kill()

    _remove_orphans(args.out)
    summary = summarize_dataset(args.out)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
