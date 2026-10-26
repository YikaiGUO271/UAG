"""Native FG adapters for the nine unchanged formal-paper instances.

The frozen builder remains the sole source of data, initial points, and supplied
smoothness constants. Compilation replaces only the evaluation boundary. No
solver is run by this module, and ``load_cases`` makes no FG queries.

Run this file directly for the declared three-point, per-case validation only.
All construction, compilation, and validation costs are outside solver timing.
"""
from environment import HERE, ROOT, FORMAL

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import numba
from numba import njit, types
from threadpoolctl import threadpool_limits, threadpool_info

ARRAY = types.float64[::1]
FG_SIGNATURE = types.Tuple((types.float64, ARRAY))(ARRAY)
SOURCE = FORMAL / 'problems.py'
NAMES = (
    'logsumexp_rho0p5', 'logsumexp_rho0p05', 'logsumexp_rho0p005',
    'rosenbrock_d200', 't_regression_s0', 'super_logistic_a9a',
    'super_logistic_australian', 'super_logistic_ionosphere',
    'super_logistic_w8a',
)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_record(array):
    array = np.asarray(array)
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


@lru_cache(maxsize=1)
def _reference_module():
    spec = importlib.util.spec_from_file_location('_uad_formal_frozen_problems', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert tuple(module.PROBLEM_NAMES) == NAMES
    return module


def _closed_values(function):
    return dict(zip(function.__code__.co_freevars,
                    (cell.cell_contents for cell in function.__closure__ or ())))


def _lse_oracle(A, b, rho):
    transpose = A.T

    @njit(FG_SIGNATURE, cache=False, fastmath=False)
    def fg(x):
        residual = A @ x
        residual -= b
        maximum = np.max(residual)
        weights = residual - maximum
        weights /= rho
        weights = np.exp(weights)
        total = np.sum(weights)
        value = maximum + rho * math.log(total)
        weights /= total
        return value, transpose @ weights

    return fg


def _t_oracle(A, b, nu):
    transpose = A.T

    @njit(FG_SIGNATURE, cache=False, fastmath=False)
    def fg(x):
        residual = A @ x
        residual -= b
        weights = residual * residual
        value = np.sum(np.log1p(weights / nu))
        weights += nu
        weights = residual / weights
        weights *= 2.
        return value, transpose @ weights

    return fg


def _rosenbrock_oracle():
    @njit(FG_SIGNATURE, cache=False, fastmath=False)
    def fg(x):
        residual = x[1:] - x[:-1] * x[:-1]
        shift = 1. - x[:-1]
        value = 100. * (residual @ residual) + shift @ shift
        gradient = np.zeros_like(x)
        gradient[:-1] = -400. * x[:-1] * residual - 2. * shift
        gradient[1:] += 200. * residual
        return value, gradient

    return fg


def _logistic_oracle(data, indices, indptr, inverse_m):
    @njit(FG_SIGNATURE, cache=False, fastmath=False)
    def fg(x):
        value = 0.
        gradient = np.zeros(len(x))
        for i in range(len(indptr) - 1):
            margin = 0.
            for k in range(indptr[i], indptr[i + 1]):
                margin += data[k] * x[indices[k]]
            if margin >= 0.:
                tail = math.exp(-margin)
                value += margin + math.log1p(tail)
                probability = 1. / (1. + tail)
            else:
                tail = math.exp(margin)
                value += math.log1p(tail)
                probability = tail / (1. + tail)
            weight = inverse_m * probability
            for k in range(indptr[i], indptr[i + 1]):
                gradient[indices[k]] += weight * data[k]
        norm_squared = 0.
        for j in range(len(x)):
            norm_squared += x[j] * x[j]
            gradient[j] += inverse_m * x[j]
        return inverse_m * value + .5 * inverse_m * norm_squared, gradient

    return fg


@lru_cache(maxsize=len(NAMES))
def _build_case(name):
    if name not in NAMES:
        raise ValueError('Unknown formal case: ' + str(name))
    started = time.perf_counter()
    reference = _reference_module().make_problems([name])[0]
    construction_seconds = time.perf_counter() - started
    independent = reference['evaluate']
    closed = _closed_values(independent)
    family = reference['family']
    x0 = np.ascontiguousarray(reference['x0'], dtype=np.float64).copy()
    L1, L2 = map(float, reference['smoothness'])
    metadata = dict(
        source=str(SOURCE.relative_to(ROOT)), source_sha256=_file_hash(SOURCE),
        dimension=len(x0), x0=_array_record(x0),
        smoothness={'L1': L1, 'L2': L2},
        construction_seconds=construction_seconds,
        construction_FG_queries=0, compilation_FG_queries=0,
        dtype='float64', fastmath=False,
        parameter_source='Unmodified frozen problems.py::make_problems',
    )
    started = time.perf_counter()
    if family in ('logsumexp', 't_regression'):
        A, b = closed['A'], closed['b']
        assert A.flags.c_contiguous and b.flags.c_contiguous
        metadata.update(rows=A.shape[0], seed=0,
                        arrays={'A': _array_record(A), 'b': _array_record(b)})
        if family == 'logsumexp':
            rho = float(closed['rho'])
            metadata['rho'] = rho
            native_fg = _lse_oracle(A, b, rho)
        else:
            nu = float(closed['nu'])
            metadata.update(nu=nu, initial_point='numpy.linalg.lstsq(A,b,rcond=None)[0]')
            native_fg = _t_oracle(A, b, nu)
    elif family == 'rosenbrock':
        metadata['smoothness_scope'] = 'Supplied rectangle bounds; not a global Lipschitz claim.'
        native_fg = _rosenbrock_oracle()
    elif family == 'logistic':
        data, indices, indptr = (closed[k] for k in ('data', 'indices', 'indptr'))
        m = int(closed['m'])
        dataset = FORMAL / 'data' / (name + '.npz')
        metadata.update(rows=m, regularization=1. / m, signed_rows='a_i=-y_i*z_i',
                        dataset=str(dataset.relative_to(ROOT)), dataset_sha256=_file_hash(dataset),
                        arrays={k: _array_record(closed[k]) for k in ('data', 'indices', 'indptr')})
        native_fg = _logistic_oracle(data, indices, indptr, 1. / m)
    else:
        raise AssertionError(family)
    metadata['compilation_seconds'] = time.perf_counter() - started
    assert native_fg.nopython_signatures == [FG_SIGNATURE]
    return dict(name=name, family=family, x0=x0, native_fg=native_fg,
                independent=independent, L1=L1, L2=L2, metadata=metadata)


def load_cases(names=None):
    """Return case dicts with compiled native oracles, without querying them.

    ``independent`` is the original frozen evaluator, reserved for external
    endpoint checks. Every paid solver FG call must use ``native_fg``. Cases
    are built once per process; returned x0 and metadata are separate copies.
    The original dense reference evaluator has private mutable work arrays and
    is intended for serial checks, matching this single-threaded campaign.
    """
    selected = NAMES if names is None else tuple(names)
    with threadpool_limits(limits=1):
        result = []
        for name in selected:
            case = _build_case(name)
            result.append(dict(case, x0=case['x0'].copy(),
                               metadata=copy.deepcopy(case['metadata'])))
    return result


def _reconstruct_parameters(case):
    """Independent data/initial-point/bound reconstruction, with zero FG calls."""
    name, family = case['name'], case['family']
    meta = case['metadata']
    arrays = {}
    if family in ('logsumexp', 't_regression'):
        m, n = meta['rows'], meta['dimension']
        rng = np.random.default_rng(0)
        A = np.ascontiguousarray(rng.standard_normal((m, n)))
        b = np.ascontiguousarray(rng.standard_normal(m))
        arrays = {'A': _array_record(A), 'b': _array_record(b)}
        if family == 'logsumexp':
            rho = {'logsumexp_rho0p5': .5, 'logsumexp_rho0p05': .05,
                   'logsumexp_rho0p005': .005}[name]
            assert rho == meta['rho']
            radius = float(np.max(np.linalg.norm(A - A.mean(axis=0), axis=1)))
            x0 = np.zeros(n)
            bounds = (radius**2 / rho, 2. * radius**3 / rho**2)
        else:
            nu = .001
            assert nu == meta['nu']
            x0 = np.ascontiguousarray(np.linalg.lstsq(A, b, rcond=None)[0])
            spectral = float(np.linalg.norm(A, 2)**2)
            radius = float(np.max(np.linalg.norm(A, axis=1)))
            bounds = (2. * spectral / nu,
                      (3. + 2. * math.sqrt(2.)) * radius * spectral / (2. * nu**1.5))
    elif family == 'rosenbrock':
        x0 = np.zeros(200)
        B = 1. + math.sqrt(len(x0) - 1.)
        C = B * B + (B - 1.) / 10.
        R, S = 4. * B + 4., 4. * C + 4.
        bounds = (1200. * R * R + 400. * S + 800. * R + 202., 2400. * R + 1200.)
    else:
        from scipy.sparse import csr_matrix
        with np.load(FORMAL / 'data' / (name + '.npz')) as stored:
            data, indices, indptr = (stored[k] for k in ('data', 'indices', 'indptr'))
            m, n = stored['shape']
        arrays = {key: _array_record(value) for key, value in
                  (('data', data), ('indices', indices), ('indptr', indptr))}
        matrix = csr_matrix((data, indices, indptr), shape=(m, n))
        spectral = float(np.linalg.eigvalsh((matrix.T @ matrix).toarray())[-1])
        radius = float(np.sqrt(matrix.multiply(matrix).sum(axis=1).max()))
        x0 = np.zeros(n)
        bounds = (spectral / (4. * m) + 1. / m,
                  radius * spectral / (6. * math.sqrt(3.) * m))
        assert 1. / m == meta['regularization']
    return dict(
        x0_bitwise_equal=bool(np.array_equal(x0, case['x0'])),
        smoothness_bitwise_equal=bool(bounds == (case['L1'], case['L2'])),
        input_array_hashes_equal=bool(arrays == meta.get('arrays', {})),
        x0=_array_record(x0), L1=float(bounds[0]), L2=float(bounds[1]),
    )


def validate(destination=None):
    """Three predetermined points per case; no optimization or repeated timing."""
    destination = HERE / 'case_validation.json' if destination is None else Path(destination)
    total_started = time.perf_counter()
    checks = []
    # Declared before inspecting results. Metrics divide by max(1, reference norm).
    tolerance = 2.e-12
    with threadpool_limits(limits=1):
        cases = load_cases()
        for case in cases:
            reconstruction_started = time.perf_counter()
            parameters = _reconstruct_parameters(case)
            reconstruction_seconds = time.perf_counter() - reconstruction_started
            assert parameters['x0_bitwise_equal'] and parameters['smoothness_bitwise_equal']
            assert parameters['input_array_hashes_equal']
            x0 = case['x0']
            j = np.arange(1, len(x0) + 1, dtype=np.float64)
            points = (('x0', x0.copy()),
                      ('x0_plus_1e-3_sin_j', np.ascontiguousarray(x0 + 1.e-3 * np.sin(j))),
                      ('x0_plus_1e-2_cos_j', np.ascontiguousarray(x0 + 1.e-2 * np.cos(j))))
            observations = []
            for point_name, x in points:
                before = x.copy()
                started = time.perf_counter()
                f_ref, g_ref = case['independent'](x)
                reference_seconds = time.perf_counter() - started
                started = time.perf_counter()
                f, g = case['native_fg'](x)
                native_seconds = time.perf_counter() - started
                value_error = abs(f - f_ref) / max(1., abs(f_ref))
                gradient_error = float(np.linalg.norm(g - g_ref)) / max(1., float(np.linalg.norm(g_ref)))
                query_passed = bool(np.isfinite(f) and np.all(np.isfinite(g))
                                    and g.dtype == np.float64 and g.flags.c_contiguous
                                    and g.shape == x.shape and np.array_equal(x, before)
                                    and value_error <= tolerance and gradient_error <= tolerance)
                observations.append(dict(
                    point=point_name, point_sha256=_array_record(x)['sha256'],
                    value=float(f), reference_value=float(f_ref),
                    gradient_norm=float(np.linalg.norm(g)), reference_gradient_norm=float(np.linalg.norm(g_ref)),
                    normalized_value_error=float(value_error), normalized_gradient_error=gradient_error,
                    gradient_max_abs_error=float(np.max(np.abs(g - g_ref))),
                    native_seconds=native_seconds, reference_seconds=reference_seconds,
                    native_FG_queries=1, reference_FG_queries=1,
                    gradient_contiguous=bool(g.flags.c_contiguous), input_unchanged=bool(np.array_equal(x, before)),
                    passed=query_passed,
                ))
                assert query_passed, (case['name'], observations[-1])
            compiled = case['native_fg'].overloads[case['native_fg'].signatures[0]]
            # A nopython parent can still contain a lifted object-mode region;
            # reject both object mode and lifted callbacks explicitly.
            assert not compiled.objectmode and not compiled.lifted
            record = dict(name=case['name'], family=case['family'], passed=True,
                          metadata=case['metadata'], parameters=parameters,
                          parameter_reconstruction_seconds=reconstruction_seconds,
                          nopython_signatures=[str(s) for s in case['native_fg'].nopython_signatures],
                          python_callback_in_native_body=False, points=observations)
            checks.append(record)
            print(json.dumps(dict(case=case['name'], passed=True,
                                  maximum_normalized_value_error=max(p['normalized_value_error'] for p in observations),
                                  maximum_normalized_gradient_error=max(p['normalized_gradient_error'] for p in observations)),
                             ensure_ascii=False), flush=True)
        report = dict(
            passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
            validation_only=True, optimization_runs=0, cases=len(checks), points_per_case=3,
            native_FG_queries=3 * len(checks), reference_FG_queries=3 * len(checks),
            normalized_error_tolerance=tolerance,
            points='x0; x0+1e-3*sin(j); x0+1e-2*cos(j), j=1,...,d',
            source_sha256=_file_hash(SOURCE), adapter_sha256=_file_hash(__file__),
            array_hash_format='SHA256 of contiguous raw bytes; shape and dtype recorded separately',
            elapsed_seconds=time.perf_counter() - total_started,
            costs_scope='Construction, JIT, data/parameter reconstruction, and fixed-point validation only; no solver run.',
            compilation_seconds=sum(c['metadata']['compilation_seconds'] for c in checks),
            construction_seconds=sum(c['metadata']['construction_seconds'] for c in checks),
            native_query_seconds=sum(p['native_seconds'] for c in checks for p in c['points']),
            reference_query_seconds=sum(p['reference_seconds'] for c in checks for p in c['points']),
            reference_query_cost_note='The first original logistic evaluator call includes its original JIT compilation.',
            environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                             numba=numba.__version__, platform=platform.platform(),
                             numerical_threads=1, threadpools=threadpool_info()),
            checks=checks,
        )
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(HERE / 'case_validation.json'))
    args = parser.parse_args()
    report = validate(args.output)
    print(json.dumps({k: report[k] for k in ('passed', 'cases', 'native_FG_queries',
                                           'reference_FG_queries', 'optimization_runs', 'elapsed_seconds')}), flush=True)
