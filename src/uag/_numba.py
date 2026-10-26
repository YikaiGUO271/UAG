"""Compile the same solver functions with native state and callback types.

Importing :mod:`uag` never imports Numba. This module is loaded only after an
explicit ``backend='numba'`` request. No FG call is made to warm compilation.
"""
from functools import lru_cache
import types as python_types
import time
import numpy as np
from numba import njit, objmode, types
from numba.experimental import jitclass
from . import _core, _runtime, _secant

ARRAY = types.float64[::1]
POINT = types.Tuple((ARRAY, types.float64, ARRAY))
FG_SIGNATURE = types.Tuple((types.float64, ARRAY))(ARRAY)
FG_TYPE = types.FunctionType(FG_SIGNATURE)


@njit
def clock():
    # Optional deadline/history diagnostics need a portable monotonic clock.
    # The default native path never enters this bridge inside an FG query.
    with objmode(value="float64"):
        value = time.perf_counter()
    return value


def _class(cls, fields):
    # jitclass must not mutate the public Python backend's class dictionary.
    attributes = {k: v for k, v in vars(cls).items()
                  if k not in ("__dict__", "__weakref__")}
    clone = type(cls.__name__, cls.__bases__, attributes)
    return jitclass(fields)(clone)


def _functions(module, overrides):
    namespace = dict(vars(module))
    namespace.update(overrides)
    for name, function in vars(module).items():
        if isinstance(function, python_types.FunctionType) and function.__module__ == module.__name__:
            copied = python_types.FunctionType(function.__code__, namespace,
                                                name, function.__defaults__, function.__closure__)
            copied.__module__ = module.__name__
            namespace[name] = njit(cache=False, fastmath=False)(copied)
    # Overrides such as the portable compiled clock replace their Python body.
    namespace.update(overrides)
    compiled = python_types.ModuleType(module.__name__ + "_native")
    vars(compiled).update(namespace)
    return compiled


@lru_cache(maxsize=1)
def engine():
    meter = _class(_runtime.Meter, [
        ("fg", FG_TYPE), ("calls", types.int64), ("status", types.int64),
        ("max_queries", types.int64), ("eps", types.float64),
        ("started", types.float64), ("deadline", types.float64),
        ("history", types.float64[:, ::1]), ("positions", types.float64[:, ::1]),
    ])
    runtime = _class(_runtime.Runtime,
        [(n, types.float64) for n in ("L", "eps", "fref", "ell", "curvature", "legacy_weight", "scan_lam", "scan_delta")]
        + [(n, types.int64) for n in ("max_queries", "calls", "status", "kind", "phase")]
        + [(n, types.boolean) for n in ("has_curvature", "scanning", "scan_found")]
        + [("counts", types.int64[::1]), ("events", types.int64[::1])]
        + [(n, POINT) for n in ("last", "best", "minimum", "scan_target", "scan_anchor", "scan_point")]
        + [("meter", meter.class_type.instance_type)])
    api = _functions(_runtime, {"Meter": meter, "Runtime": runtime, "now": clock})
    private = _class(_secant.PrivateState,
        [("point", POINT)]
        + [(n, ARRAY) for n in ("previous_gradient", "previous_step", "previous_auxiliary", "work")]
        + [(n, types.boolean) for n in ("valid", "have_pair", "diag_started")]
        + [(n, types.int64) for n in ("clock", "diag_count", "diag_dropped")]
        + [(n, types.float64) for n in ("damping", "scale")]
        + [("diag", types.float64[:, ::1]), ("diag_events", types.int64[::1])])
    secant = _functions(_secant, {"API": api, "PrivateState": private})
    state = _class(_core.State, [
        ("log_mass", types.float64), ("z", ARRAY), ("point", POINT),
        ("entry_ratio", types.float64), ("backward", types.float64),
        ("previous_beta", types.float64), ("postcommit", types.boolean)])
    journal = _class(_core.Journal, [("rows", types.float64[:, ::1]), ("count", types.int64)])
    core = _functions(_core, {"API": api, "MS": secant, "State": state, "Journal": journal})
    return api, core, secant


def prepare(fg):
    from numba.core.registry import CPUDispatcher
    if not isinstance(fg, CPUDispatcher) or not fg.targetoptions.get("nopython", False):
        raise TypeError("backend='numba' requires an @numba.njit FG callable")
    # Signature compilation performs no objective evaluation. Precompiled fixed
    # signatures can be reused without asking a disabled dispatcher to compile.
    if FG_SIGNATURE.args not in fg.overloads:
        fg.compile(FG_SIGNATURE)
    actual = fg.overloads[FG_SIGNATURE.args].signature
    if actual.return_type != FG_SIGNATURE.return_type:
        raise TypeError("Numba FG must return (float64, contiguous float64[::1])")
    api, core, secant = engine()
    core.solve.compile((ARRAY, api.Runtime.class_type.instance_type,
                        secant.PrivateState.class_type.instance_type,
                        core.Journal.class_type.instance_type, types.boolean))
    return api, core, secant
