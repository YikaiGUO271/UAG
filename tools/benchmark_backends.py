"""Development implementation-cost check, not paper or general speed evidence.

Run after the formal timing campaign, in an otherwise idle process. Importing
this module does not import numerical libraries, compile, or run any solver.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[4, 256, 8192])
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--max-evals", type=int, default=32)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max-wall-seconds", type=float, default=115.)
    parser.add_argument("--output", type=Path, default=Path("benchmark_backends.json"))
    args = parser.parse_args()
    if (min(args.dimensions) < 1 or args.repetitions < 1 or args.max_evals < 1
            or not math.isfinite(args.tol) or args.tol <= 0
            or not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0):
        parser.error("dimensions, repetitions, max-evals, tol and wall seconds must be positive and finite")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"  # This standalone process only; never library policy.
    try:
        import numpy as np
        import numba
        import scipy
        from threadpoolctl import threadpool_info, threadpool_limits
        import uag
    except ImportError as exc:
        raise SystemExit("Install: python -m pip install -e '.[accelerated]' threadpoolctl") from exc
    numba.set_num_threads(1)
    started = time.perf_counter()

    def expired():
        return time.perf_counter()-started >= args.max_wall_seconds

    def digest(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    def summary(values):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        return dict(median_seconds=float(median), q1_seconds=float(q1),
                    q3_seconds=float(q3), iqr_seconds=float(q3-q1))

    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    def compare(a, b):
        checks = {
            "nfev": a.nfev == b.nfev, "status": a.status == b.status,
            "success": a.success == b.success, "outer_updates": a.nit == b.nit,
            "counts": bool(np.array_equal(a.work.counts, b.work.counts)),
            "events": bool(np.array_equal(a.work.events, b.work.events)),
            "x": bool(np.allclose(a.x, b.x, rtol=1e-10, atol=1e-12)),
            "jac": bool(np.allclose(a.jac, b.jac, rtol=1e-10, atol=1e-12)),
            "fun": bool(np.isclose(a.fun, b.fun, rtol=1e-10, atol=1e-12)),
            "valid_status": a.status in (1, 2) and b.status in (1, 2),
        }
        return dict(all_passed=all(checks.values()), checks=checks,
                    max_x_absolute_difference=float(np.max(np.abs(a.x-b.x))),
                    max_jac_absolute_difference=float(np.max(np.abs(a.jac-b.jac))),
                    function_absolute_difference=float(abs(a.fun-b.fun)))

    report = {
        "scope": "Development backend implementation costs; not paper benchmark or universal speed evidence.",
        "historical_comparison": "Not timed; separate trajectory validation: tests/data/frozen_prefixes.json.",
        "objective": "f(x)=0.5*dot(x,d*x), d=linspace(1,64,dimension)",
        "initial_point": "x0[j]=(-1)^j/sqrt(dimension)",
        "settings": dict(dimensions=args.dimensions, repetitions=args.repetitions,
                         max_evals=args.max_evals, tol=args.tol, initial_scale=1.,
                         secant=True, history=False, positions=False,
                         threads=1, wall_limit_between_calls=args.max_wall_seconds),
        "endpoint_tolerances": dict(rtol=1e-10, atol=1e-12),
        "environment": dict(python=platform.python_version(), platform=platform.platform(),
                            machine=platform.machine(), uag=uag.__version__,
                            numpy=np.__version__, numba=numba.__version__, scipy=scipy.__version__),
        "total_fg_calls": 0, "compilation_fg_calls": 0,
        "unreported_warmup_fg_calls": 0, "cases": [], "completed": True,
    }
    with threadpool_limits(limits=1):
        for dimension in args.dimensions:
            if expired():
                report["completed"] = False
                break
            diagonal = np.linspace(1., 64., dimension)
            x0 = np.where(np.arange(dimension) % 2, -1., 1.) / np.sqrt(dimension)

            def objective(x):
                gradient = diagonal*x
                return .5*float(x @ gradient), gradient

            callbacks = dict(python=objective, numba=numba.njit(objective))
            case = dict(dimension=dimension, x0_sha256=digest(x0),
                        diagonal_sha256=digest(diagonal), preparation=[], runs=[],
                        comparisons=[], completed=True)
            report["cases"].append(case)
            for backend in ("python", "numba"):
                before = time.perf_counter()
                prepared = uag.minimize(callbacks[backend], x0, backend=backend,
                                        tol=args.tol, max_evals=0)
                wall = time.perf_counter()-before
                assert prepared.nfev == 0 and prepared.status == 2
                case["preparation"].append(dict(backend=backend, nfev=0,
                    setup_and_compilation_seconds=prepared.setup_seconds, public_call_seconds=wall))
            baseline = None
            for repetition in range(args.repetitions):
                order = ("python", "numba") if repetition % 2 == 0 else ("numba", "python")
                pair = {}
                for backend in order:
                    if expired():
                        case["completed"] = report["completed"] = False
                        break
                    before = time.perf_counter()
                    result = uag.minimize(callbacks[backend], x0, backend=backend,
                        tol=args.tol, max_evals=args.max_evals, initial_scale=1., secant=True)
                    wall = time.perf_counter()-before
                    assert result.nfev == int(result.work.counts.sum()), "FG accounting mismatch"
                    assert result.history is None and result.points is None
                    pair[backend] = result
                    case["runs"].append(dict(backend=backend, repetition=repetition,
                        nfev=result.nfev, nit=result.nit, status=result.status,
                        success=result.success, message=result.message, fun=result.fun,
                        gradient_norm=result.grad_norm, x_sha256=digest(result.x),
                        jac_sha256=digest(result.jac), setup_seconds=result.setup_seconds,
                        solve_seconds=result.elapsed_seconds, public_call_seconds=wall,
                        phase_names=result.work.phase_names, kind_names=result.work.kind_names,
                        counts=result.work.counts.tolist(), events=result.work.events.tolist()))
                    report["total_fg_calls"] += result.nfev
                    if baseline is None and backend == "python":
                        baseline = result
                    if baseline is not None:
                        case["comparisons"].append(dict(compare(baseline, result),
                            against="first Python run", backend=backend, repetition=repetition))
                if len(pair) == 2:
                    case["comparisons"].append(dict(compare(pair["python"], pair["numba"]),
                        against="same repetition", repetition=repetition))
                if not case["completed"]:
                    break
            pools = threadpool_info()
            case["threadpools"] = [{key: pool.get(key) for key in
                ("user_api", "internal_api", "prefix", "num_threads", "version", "architecture")}
                for pool in pools]
            case["single_thread_verified"] = (bool(pools) and
                all(pool.get("num_threads") == 1 for pool in pools) and numba.get_num_threads() == 1)
            case["summary"] = {}
            for backend in ("python", "numba"):
                rows = [row for row in case["runs"] if row["backend"] == backend]
                if rows:
                    case["summary"][backend] = {field: summary([row[field] for row in rows])
                        for field in ("setup_seconds", "solve_seconds", "public_call_seconds")}
            eligible = (case["completed"] and len(case["runs"]) == 2*args.repetitions
                and case["single_thread_verified"] and all(c["all_passed"] for c in case["comparisons"]))
            case["speedup_eligible"] = eligible
            if eligible:
                a = case["summary"]["python"]["solve_seconds"]["median_seconds"]
                b = case["summary"]["numba"]["solve_seconds"]["median_seconds"]
                if b > 0:
                    case["python_over_numba_median_solve_ratio"] = a/b
            else:
                case["ratio_omitted_reason"] = "Incomplete runs, invalid status, endpoint/work discrepancy, or unverified thread count; inspect comparisons."
            print(json.dumps(clean(dict(dimension=dimension, completed=case["completed"],
                speedup_eligible=eligible, summary=case["summary"]))), flush=True)
            if not case["completed"]:
                break
    report["wall_seconds_including_preparation"] = time.perf_counter()-started
    report["completed"] = report["completed"] and len(report["cases"]) == len(args.dimensions)
    report["verification_passed"] = (report["completed"] and
        all(case["speedup_eligible"] for case in report["cases"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Saved {args.output}; charged development FG calls: {report['total_fg_calls']}", flush=True)
    return (0 if report["verification_passed"] else 1) if report["completed"] else 2


if __name__ == "__main__":
    sys.exit(main())
