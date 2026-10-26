"""Shared native FG boundary with opt-in diagnostics and common budgets.

All solvers share this boundary. No Python callback occurs in a paid FG call.
Compilation and object construction belong outside the timed observation.
"""
import time
import math
import numpy as np
from numba import njit, types, objmode
from numba.experimental import jitclass

ARRAY = types.float64[::1]
POINT_TYPE = types.Tuple((ARRAY, types.float64, ARRAY))
FG_SIGNATURE = types.Tuple((types.float64, ARRAY))(ARRAY)
FG_TYPE = types.FunctionType(FG_SIGNATURE)
@njit(cache=False)
def now():
    # Monotonic cross-platform clock. The Python bridge is charged to every run.
    with objmode(value='float64'):
        value = time.perf_counter()
    return value


@njit(cache=False, inline='always')
def meter_now(meter):
    return now()


@jitclass([("fg", FG_TYPE), ("calls", types.int64), ("status", types.int64),
           ("max_queries", types.int64), ("eps", types.float64),
           ("started", types.float64), ("deadline", types.float64),
           ("history", types.float64[:, ::1]), ("positions", types.float64[:, ::1]),
           ("tick", types.int64[::1])])
class Meter:
    def __init__(self, fg, max_queries, eps):
        self.fg, self.max_queries, self.eps = fg, max_queries, eps
        self.calls, self.status = 0, 0
        self.started, self.deadline = 0., np.inf
        # Columns: attempted FG index, elapsed seconds, f, L2 gradient.
        self.history = np.empty((0, 4), dtype=np.float64)
        self.positions = np.empty((0, 0), dtype=np.float64)
        self.tick = np.zeros(1, dtype=np.int64)


@jitclass(
    [(name, types.float64) for name in ("L", "eps", "fref", "ell", "curvature", "legacy_weight", "scan_lam", "scan_delta")]
    + [(name, types.int64) for name in ("max_queries", "calls", "status", "kind", "phase")]
    + [(name, types.boolean) for name in ("has_curvature", "scanning", "scan_found")]
    + [("counts", types.int64[::1]), ("events", types.int64[::1])]
    + [(name, POINT_TYPE) for name in ("last", "best", "minimum", "scan_target", "scan_anchor", "scan_point")]
    + [("meter", Meter.class_type.instance_type)]
)
class Runtime:
    def __init__(self, scale, eps, max_queries, dimension, meter):
        self.L, self.ell, self.eps = scale, scale, eps
        self.fref, self.curvature, self.legacy_weight = 0., 0., 1.
        self.max_queries, self.calls, self.status = max_queries, 0, 0
        self.has_curvature = False
        self.kind, self.phase = 0, 0
        self.counts = np.zeros(32, dtype=np.int64)
        self.events = np.zeros(32, dtype=np.int64)
        point = (np.zeros(dimension), 0., np.zeros(dimension))
        self.last, self.best, self.minimum = point, point, point
        self.meter = meter
        self.scanning, self.scan_found = False, False
        self.scan_lam, self.scan_delta = 0., 0.
        self.scan_target, self.scan_anchor, self.scan_point = point, point, point


@njit(cache=False)
def all_finite(x):
    # IEEE binary64 exponent bits, reduced without a temporary boolean array.
    bits = x.view(np.uint64)
    bad = np.uint64(0)
    mask = np.uint64(0x7ff0000000000000)
    for j in range(len(x)):
        bad |= np.uint64((bits[j] & mask) == mask)
    return bad == 0


@njit(cache=False)
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
    if meter.deadline < np.inf and meter_now(meter) >= meter.deadline:
        run.status, meter.status = 4, 4
        return run.last
    if not all_finite(x):
        run.status, meter.status = 5, 5
        return run.last
    x = x.copy()
    meter.calls += 1
    run.calls += 1
    run.counts[8*run.phase + run.kind] += 1
    try:
        f, g = meter.fg(x)
    except Exception:
        f, g = np.nan, np.full(len(x), np.nan)
        run.status, meter.status = 6, 6
    norm = math.sqrt(np.dot(g, g))
    row = meter.calls - 1
    if row < meter.positions.shape[0]:
        meter.positions[row, :] = x
    if row < meter.history.shape[0]:
        meter.history[row, 0] = float(meter.calls)
        meter.history[row, 1] = meter_now(meter) - meter.started
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
    if run.scanning and not run.scan_found:
        target, anchor = run.scan_target, run.scan_anchor
        d = target[0] - x
        d2 = np.dot(d, d)
        eta = 2*math.sqrt(128*run.scan_lam*run.scan_delta)/run.scan_lam
        defect = target[1] - f - np.dot(g, d) + run.scan_lam*d2/4
        if (0 < d2 <= (eta/4)**2 and defect < 0
                and target[1] <= anchor[1] + run.scan_delta + 1e-12*max(1., abs(anchor[1]))):
            run.scan_found = True
            run.scan_point = (x.copy(), f, g.copy())
    if norm <= run.eps:
        run.status, meter.status = 1, 1
    return point
