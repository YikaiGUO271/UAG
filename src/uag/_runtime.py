"""Owned FG values and common accounting for both execution backends."""
import math
import time
import numpy as np


def now():
    return time.perf_counter()


class Meter:
    def __init__(self, fg, max_queries, eps):
        self.fg, self.max_queries, self.eps = fg, max_queries, eps
        self.calls, self.status = 0, 0
        self.started, self.deadline = 0., np.inf
        # Columns: attempted FG index, seconds from solve start, f, ||g||.
        self.history = np.empty((0, 4), dtype=np.float64)
        self.positions = np.empty((0, 0), dtype=np.float64)


class Runtime:
    def __init__(self, scale, eps, max_queries, dimension, meter):
        self.L, self.ell, self.eps = scale, scale, eps
        self.fref, self.curvature, self.legacy_weight = 0., 0., 1.
        self.max_queries, self.calls, self.status = max_queries, 0, 0
        self.has_curvature = False
        self.kind, self.phase = 0, 0
        self.counts = np.zeros(32, dtype=np.int64)
        self.events = np.zeros(128, dtype=np.int64)
        point = (np.zeros(dimension), np.nan, np.full(dimension, np.nan))
        self.last, self.best, self.minimum = point, point, point
        self.meter = meter
        self.scanning, self.scan_found = False, False
        self.scan_lam, self.scan_delta = 0., 0.
        self.scan_target, self.scan_anchor, self.scan_point = point, point, point


def all_finite(x):
    return np.isfinite(x).all()


def query(x, run):
    meter = run.meter
    if meter.status:
        run.status = meter.status
        return run.last
    if run.calls >= run.max_queries or meter.calls >= meter.max_queries:
        run.status = 2
        run.last = run.minimum
        if meter.calls >= meter.max_queries:
            meter.status = 2
        return run.last
    if meter.deadline < np.inf and now() >= meter.deadline:
        run.status, meter.status = 4, 4
        return run.last
    if not all_finite(x):
        run.status, meter.status = 5, 5
        return run.last
    # A callback may return an alias of its input or a reused work buffer.
    # Both the point and its gradient must outlive the callback invocation.
    x = x.copy()
    meter.calls += 1
    run.calls += 1
    run.counts[8*run.phase + run.kind] += 1
    try:
        f, returned_gradient = meter.fg(x)
        g = returned_gradient.copy()
        if g.ndim != 1 or len(g) != len(x):
            f, g = np.nan, np.full(len(x), np.nan)
            run.status, meter.status = 6, 6
    except Exception:
        f, g = np.nan, np.full(len(x), np.nan)
        run.status, meter.status = 6, 6
    norm = math.sqrt(np.dot(g, g))
    row = meter.calls - 1
    if row < meter.positions.shape[0]:
        meter.positions[row, :] = x
    if row < meter.history.shape[0]:
        meter.history[row, 0] = float(meter.calls)
        meter.history[row, 1] = now() - meter.started
        meter.history[row, 2] = f
        meter.history[row, 3] = norm
    point = (x, f, g)
    run.last = point
    if not math.isfinite(f) or not math.isfinite(norm):
        if not meter.status:
            run.status, meter.status = 3, 3
        return point
    if run.calls == 1:
        run.fref, run.best, run.minimum = f, point, point
    else:
        if f < run.best[1]:
            run.best = point
        if norm < math.sqrt(np.dot(run.minimum[2], run.minimum[2])):
            run.minimum = point
    if norm <= run.eps:
        run.status, meter.status = 1, 1
    return point
