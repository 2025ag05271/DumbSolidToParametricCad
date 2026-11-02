"""Crash- and hang-tolerant process pool for CAD-kernel work.

``multiprocessing.Pool`` is unsafe for OpenCASCADE jobs: if a worker dies inside the kernel (access
violation) its task is silently lost and ``AsyncResult.get`` only returns after the full timeout;
a hung kernel call blocks a worker forever. Here every worker is supervised individually — a crashed
or timed-out worker is killed and replaced, and its task is reported as ``"crash"`` / ``"timeout"``
immediately.

    pool = SupervisedPool(fn, workers=12, timeout=120)       # fn: picklable top-level function
    for i, status, result in pool.run(tasks):                 # completion order
        ...                                                   # status: ok | error | crash | timeout
    pool.close()

Workers persist across ``run`` calls (no re-spawn cost) and are recycled every
``tasks_per_worker`` tasks to bound kernel memory growth.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing.connection import wait


def _worker_loop(conn, fn) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    while True:
        task = conn.recv()
        if task is None:
            return
        try:
            conn.send(("ok", fn(task)))
        except Exception as e:  # Python-level failure: report, keep the worker
            conn.send(("error", f"{type(e).__name__}: {str(e)[:300]}"))


class _Slot:
    def __init__(self, ctx, fn):
        self.ctx, self.fn = ctx, fn
        self.proc = self.conn = None
        self.task = None  # (index, deadline)
        self.count = 0
        self.spawn()

    def spawn(self):
        parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_worker_loop, args=(child, self.fn), daemon=True)
        self.proc.start()
        child.close()
        self.conn, self.count = parent, 0

    def kill(self):
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(5)
        self.conn.close()

    def submit(self, index, task, timeout, recycle_after):
        if self.count >= recycle_after:
            try:
                self.conn.send(None)
                self.proc.join(10)
            except OSError:
                pass
            self.kill()
            self.spawn()
        self.task = (index, time.time() + timeout)
        self.count += 1
        self.conn.send(task)


class SupervisedPool:
    def __init__(self, fn, workers: int, timeout=120.0, tasks_per_worker: int = 50):
        """``timeout``: seconds per task, or a callable ``task -> seconds``."""
        self.fn, self.workers, self.timeout, self.recycle = fn, workers, timeout, tasks_per_worker
        self.ctx = mp.get_context("spawn")
        self.slots: list[_Slot] = []

    def _timeout(self, task) -> float:
        return self.timeout(task) if callable(self.timeout) else self.timeout

    def run(self, tasks):
        """Yield ``(task index, status, result)`` as tasks finish."""
        queue = list(enumerate(tasks))[::-1]
        while len(self.slots) < min(self.workers, len(queue)):
            self.slots.append(_Slot(self.ctx, self.fn))
        for s in self.slots:
            if queue:
                i, t = queue.pop()
                s.submit(i, t, self._timeout(t), self.recycle)
        while any(s.task for s in self.slots):
            busy = [s for s in self.slots if s.task]
            ready = wait([s.conn for s in busy] + [s.proc.sentinel for s in busy], timeout=1.0)
            for s in busy:
                index = s.task[0]
                out = None
                if s.conn in ready:
                    try:
                        status, result = s.conn.recv()
                        out = (index, status, result)
                    except (EOFError, OSError):
                        pass
                if out is None and (not s.proc.is_alive() or time.time() > s.task[1]):
                    reason = "crash" if not s.proc.is_alive() else "timeout"
                    out = (index, reason, f"worker {reason} (exit code {s.proc.exitcode})")
                    s.kill()
                    s.spawn()
                if out is not None:
                    s.task = None
                    if queue:
                        i, t = queue.pop()
                        s.submit(i, t, self._timeout(t), self.recycle)
                    yield out

    def close(self):
        for s in self.slots:
            try:
                s.conn.send(None)
            except OSError:
                pass
            s.kill()
        self.slots = []
