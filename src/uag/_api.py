"""Public, validation-focused interface to the UAG recurrence."""
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Optional
import math
import time
import numpy as np
from . import _core, _runtime, _secant

_KINDS = ("initial", "calibration", "extrapolation", "nag_step",
          "residual_probe", "secant", "proximal_correction", "directional_trial")
_PHASES = ("initialization", "ordinary", "localized", "replay")
# The recurrence uses tol**2. Keeping that square in the normal float64
# range also prevents an underflowed gradient dot product from falsely
# passing an unrealistically small stopping tolerance.
_MIN_TOL = math.sqrt(np.finfo(np.float64).tiny)
_MAX_TOL = math.sqrt(np.finfo(np.float64).max)
_MESSAGES = {
    1: "Gradient norm satisfies the requested tolerance.",
    2: "FG evaluation budget exhausted.",
    3: "The objective returned a nonfinite value or gradient norm.",
    4: "Time limit reached before the next FG evaluation.",
    5: "The algorithm generated a nonfinite trial point.",
    6: "FG callback failed or returned an invalid gradient.",
    7: "Numerical safeguard or internal invariant failed.",
}


def _owned(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WorkStats:
    """Charged FG work, including rejected trials and replay.

    ``counts`` has rows ``phase_names`` and columns ``kind_names``. Its sum
    equals ``nfev`` even when a paid callback fails. ``events`` is an optional
    low-level fixed-size diagnostic counter vector, not an evaluation count.
    """
    counts: np.ndarray
    events: np.ndarray
    phase_names: tuple = _PHASES
    kind_names: tuple = _KINDS

    @property
    def replay_evals(self):
        return int(self.counts[3].sum())

    @property
    def secant_evals(self):
        return int(self.counts[:, 5].sum())


@dataclass(frozen=True)
class OptimizeResult:
    """Owned final values; arrays are read-only snapshots.

    On success, ``x`` is the first queried point meeting ``tol``. Otherwise
    it is the finite queried point with smallest gradient norm, or ``x0``
    with NaN function/gradient if no finite query succeeded. ``nit`` counts
    accepted outer updates; it need not count a terminal successful query.
    History columns are FG index, elapsed seconds, function value, gradient
    norm. Histories include rejected, replayed and invalid paid evaluations.
    """
    x: np.ndarray
    fun: float
    jac: np.ndarray
    nfev: int
    nit: int
    success: bool
    status: int
    message: str
    work: WorkStats
    backend: str
    elapsed_seconds: float
    setup_seconds: float
    history: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None

    @property
    def grad_norm(self):
        return float(np.linalg.norm(self.jac))


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def minimize(fg: Callable, x0, *, tol=1e-6, max_evals=100_000,
             initial_scale=1.0, backend="python", time_limit=None,
             record_history=False, record_points=False, secant=True):
    """Minimize a smooth unconstrained function using the UAG method.

    Parameters
    ----------
    fg : callable
        ``fg(x) -> (f, g)`` for a real one-dimensional float64 point. Return
        a scalar and gradient of the same shape. The function must be
        deterministic for replay and must not mutate ``x``. Reusing an output
        gradient buffer is supported. Python callbacks receive a read-only
        view; the native callback contract is checked by the caller.
    x0 : array_like
        Finite, nonempty one-dimensional initial point, copied on entry.
    tol : float
        Absolute Euclidean gradient-norm tolerance, between sqrt(float64.tiny)
        and sqrt(float64.max), inclusive (approximately 1.49e-154 to 1.34e154).
        The recurrence squares this tolerance; other internal quantities may
        still exceed the usable floating-point range on badly scaled problems.
    max_evals : int
        Nonnegative hard limit on all attempted FG calls (including failures).
    initial_scale : float
        Positive starting guess for calibration; no smoothness bound required.
    backend : {'python', 'numba'}
        Numba requires the accelerated extra and an explicitly njit-compiled
        FG. Type compilation occurs before timing and never evaluates FG.
    time_limit : float or None
        Optional positive solve-time budget, checked before each FG. Cannot
        interrupt an in-progress callback. Native timing/history uses a small
        Python clock bridge; omit both for a fully native per-FG boundary.
    record_history, record_points : bool
        Explicitly allocate O(max_evals) scalar history / O(d*max_evals) points.
        Both default off, preserving O(d) solver working storage.
    secant : bool
        Enable the manuscript's retained one-pair secant proposals (default).
        False is an explicit diagnostic ablation, not the full paper method.

    Invalid arguments raise before any query. Paid callback failures return
    status 6. Numerical failures return status 7 with all work retained.
    """
    started_setup = time.perf_counter()
    if not callable(fg):
        raise TypeError("fg must be callable")
    raw = np.asarray(x0)
    if np.iscomplexobj(raw):
        raise ValueError("x0 must be real")
    x = np.array(raw, dtype=np.float64, order="C", copy=True)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("x0 must be a nonempty finite one-dimensional array")
    tol = _positive(tol, "tol")
    if not _MIN_TOL <= tol <= _MAX_TOL:
        raise ValueError(
            f"tol must be between {_MIN_TOL:.17g} and {_MAX_TOL:.17g} "
            "so the squared tolerance is a normal finite float64 value"
        )
    scale = _positive(initial_scale, "initial_scale")
    if isinstance(max_evals, bool) or not isinstance(max_evals, Integral):
        raise TypeError("max_evals must be an integer")
    max_evals = int(max_evals)
    if not 0 <= max_evals <= np.iinfo(np.int64).max:
        raise ValueError("max_evals must be between zero and int64 maximum")
    if time_limit is not None:
        time_limit = _positive(time_limit, "time_limit")
    for name, value in (("record_history", record_history), ("record_points", record_points), ("secant", secant)):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be bool")
    callback_error = []
    if backend == "python":
        api, core, private_module = _runtime, _core, _secant

        def wrapped(point):
            view = point.view()
            view.setflags(write=False)
            try:
                value, gradient = fg(view)
                if np.ndim(value) != 0 or np.iscomplexobj(value) or np.iscomplexobj(gradient):
                    raise ValueError("FG must return a real scalar and real gradient")
                gradient = np.asarray(gradient, dtype=np.float64)
                if gradient.shape != point.shape:
                    raise ValueError("gradient shape must equal x.shape")
                return float(value), np.ascontiguousarray(gradient)
            except Exception as exc:
                callback_error[:] = [f"{type(exc).__name__}: {exc}"]
                raise

        objective = wrapped
    elif backend == "numba":
        try:
            from ._numba import prepare
        except ImportError as exc:
            raise ImportError("Install uag-optimization[accelerated] for the Numba backend") from exc
        api, core, private_module = prepare(fg)
        objective = fg
    else:
        raise ValueError("backend must be 'python' or 'numba'")
    meter = api.Meter(objective, max_evals, tol)
    if record_history:
        meter.history = np.empty((max_evals, 4), dtype=np.float64)
    if record_points:
        meter.positions = np.empty((max_evals, x.size), dtype=np.float64)
    run = api.Runtime(scale, tol, max_evals, x.size, meter)
    private = private_module.PrivateState(x.size)
    journal = core.Journal(0)
    setup_seconds = time.perf_counter() - started_setup
    meter.started = time.perf_counter()
    meter.deadline = np.inf if time_limit is None else meter.started + time_limit
    error = ""
    try:
        if max_evals:
            core.solve(x, run, private, journal, bool(secant))
        else:
            run.status = 2
    except (ArithmeticError, ValueError) as exc:
        run.status = 7
        error = f" {type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - meter.started
    status = int(run.status)
    if status == 0:
        status = 7
        error = " Solver stopped without a terminal condition."
    chosen = run.last if status == 1 else run.minimum
    if not math.isfinite(chosen[1]) or not np.isfinite(chosen[2]).all():
        chosen = (x, np.nan, np.full(x.size, np.nan))
    calls = int(meter.calls)
    counts = _owned(run.counts.reshape(4, 8))
    if int(counts.sum()) != calls or int(run.calls) != calls:
        raise RuntimeError("FG accounting invariant failed")
    message = _MESSAGES[status] + error
    if status == 6 and callback_error:
        message += " " + callback_error[0]
    return OptimizeResult(
        _owned(chosen[0]), float(chosen[1]), _owned(chosen[2]), calls,
        int(run.events[123]), status == 1, status, message,
        WorkStats(counts, _owned(run.events)), backend, elapsed, setup_seconds,
        _owned(meter.history[:calls]) if record_history else None,
        _owned(meter.positions[:calls]) if record_points else None,
    )
