"""Four experiment families. Each evaluation returns one value and gradient.

LogSumExp and t regression use independent standard-normal data.
LogSumExp varies temperature on the same seed-0 data and zero initialization.
Logistic data are signed LIBSVM rows a_i = -y_i z_i, with lambda = 1/m.
Data generation and least-squares initialization run before solver timing.
"""
import math
from pathlib import Path

import numpy as np
from numba import njit


LSE_TEMPERATURES = {
    'logsumexp_rho0p5': .5,
    'logsumexp_rho0p05': .05,
    'logsumexp_rho0p005': .005,
}
PROBLEM_NAMES = (
    *LSE_TEMPERATURES,
    'rosenbrock_d200', 't_regression_s0',
    'super_logistic_a9a', 'super_logistic_australian',
    'super_logistic_ionosphere', 'super_logistic_w8a',
)
DATA = Path(__file__).resolve().parent / 'data'


def _rosenbrock(x):
    residual = x[1:] - x[:-1] * x[:-1]
    shift = 1. - x[:-1]
    value = 100. * float(residual @ residual) + float(shift @ shift)
    gradient = np.zeros_like(x)
    gradient[:-1] = -400. * x[:-1] * residual - 2. * shift
    gradient[1:] += 200. * residual
    return value, gradient


def _dense_problem(family, seed, rho=None):
    rng = np.random.default_rng(seed)
    m, n = (1000, 200) if family == 'logsumexp' else (500, 200)
    A = np.ascontiguousarray(rng.standard_normal((m, n)))
    b = np.ascontiguousarray(rng.standard_normal(m))
    transpose = A.T
    residual = np.empty(m)
    weights = np.empty(m)

    if family == 'logsumexp':
        x0 = np.zeros(n)
        radius = float(np.max(np.linalg.norm(A - A.mean(axis=0), axis=1)))
        smoothness = (radius**2 / rho, 2. * radius**3 / rho**2)

        def evaluate(x):
            np.dot(A, x, out=residual)
            residual[:] -= b
            maximum = float(np.max(residual))
            np.subtract(residual, maximum, out=weights)
            weights[:] /= rho
            np.exp(weights, out=weights)
            total = float(np.sum(weights))
            value = maximum + rho * math.log(total)
            weights[:] /= total
            return value, transpose @ weights
    else:
        nu = .001
        x0 = np.ascontiguousarray(np.linalg.lstsq(A, b, rcond=None)[0])
        spectral = float(np.linalg.norm(A, 2)**2)
        radius = float(np.max(np.linalg.norm(A, axis=1)))
        smoothness = (2. * spectral / nu,
                      (3. + 2. * math.sqrt(2.)) * radius * spectral / (2. * nu**1.5))

        def evaluate(x):
            np.dot(A, x, out=residual)
            residual[:] -= b
            np.square(residual, out=weights)
            value = float(np.sum(np.log1p(weights / nu)))
            weights[:] += nu
            np.divide(residual, weights, out=weights)
            weights[:] *= 2.
            return value, transpose @ weights
    return x0, evaluate, smoothness


@njit(cache=False, fastmath=False)
def _logistic(x, data, indices, indptr, inverse_m):
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


def _logistic_problem(name):
    from scipy.sparse import csr_matrix

    with np.load(DATA / (name + '.npz')) as stored:
        data, indices, indptr = (stored[key] for key in ('data', 'indices', 'indptr'))
        m, n = stored['shape']
    matrix = csr_matrix((data, indices, indptr), shape=(m, n))
    spectral = float(np.linalg.eigvalsh((matrix.T @ matrix).toarray())[-1])
    radius = float(np.sqrt(matrix.multiply(matrix).sum(axis=1).max()))
    smoothness = (spectral / (4. * m) + 1. / m,
                  radius * spectral / (6. * math.sqrt(3.) * m))

    def evaluate(x):
        return _logistic(x, data, indices, indptr, 1. / m)

    return np.zeros(n), evaluate, smoothness


def make_problems(names=None):
    """Build the selected instances in order; each oracle owns its work arrays."""
    problems = []
    for name in PROBLEM_NAMES if names is None else names:
        if name not in PROBLEM_NAMES:
            raise ValueError(f'Unknown problem: {name}')
        if name.startswith('logsumexp_'):
            family = 'logsumexp'
            x0, evaluate, smoothness = _dense_problem(family, 0, LSE_TEMPERATURES[name])
        elif name == 'rosenbrock_d200':
            family = 'rosenbrock'
            x0, evaluate = np.zeros(200), _rosenbrock
            # Analytic bounds on a rectangle containing every CUG query;
            # Rosenbrock is not globally smooth. See the notebook derivation.
            B = 1. + math.sqrt(len(x0) - 1.)
            C = B * B + (B - 1.) / 10.
            R, S = 4. * B + 4., 4. * C + 4.
            smoothness = (1200. * R * R + 400. * S + 800. * R + 202.,
                          2400. * R + 1200.)
        elif name == 't_regression_s0':
            family = 't_regression'
            x0, evaluate, smoothness = _dense_problem(family, 0)
        else:
            family = 'logistic'
            x0, evaluate, smoothness = _logistic_problem(name)
        problems.append(dict(name=name, family=family, x0=x0, evaluate=evaluate,
                             smoothness=smoothness))
    return problems
