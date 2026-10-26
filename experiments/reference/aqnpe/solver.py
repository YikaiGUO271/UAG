"""Jiang--Mokhtari A-QNPE, Algorithm 1 and Subroutines 1--4.

The matrix W is represented as w I + Q C Q.T, with orthonormal Q.  No history
is dropped and no rank limit is imposed: this is an algebraic representation
of the original full matrix, not a limited-memory quasi-Newton modification.
All objective/gradient evaluations use the supplied combined FG callable.

With supplied L1, the original matrix-learning recurrence is used.  L1=None
selects an explicitly experimental online-scale extension, documented in the
adjacent notes: observed secants can enlarge the scale and reset the learner.
It is not the theorem-covered original algorithm with an unknown global L1.
"""

import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _action(w, Q, C, rank, x):
    coefficients = np.empty(rank)
    for j in range(rank):
        coefficients[j] = np.dot(Q[:, j], x)
    transformed = np.zeros(rank)
    for i in range(rank):
        for j in range(rank):
            transformed[i] += C[i, j] * coefficients[j]
    result = w * x
    for j in range(rank):
        for i in range(len(x)):
            result[i] += transformed[j] * Q[i, j]
    return result


@njit(cache=True, fastmath=False)
def _metric_action(w, Q, C, rank, scale, shift, x):
    # Combine the identity coefficients before multiplying.  In particular,
    # B0=(L/2)(I-I)=0 is exact even when L is very large.
    return (shift + scale * w) * x + scale * _action(0., Q, C, rank, x)


@njit(cache=True, fastmath=False)
def _cr(w, Q, C, rank, scale, shift, eta, gradient, alpha):
    """Original conjugate residual solver, using a matrix-free B action."""
    s = np.zeros_like(gradient)
    r = -eta * gradient
    p = r.copy()
    Ar = r + eta * _metric_action(w, Q, C, rank, scale, shift, r)
    Ap = Ar.copy()
    rr = np.dot(r, Ar)
    products = 1
    for iteration in range(4 * len(gradient) + 50):
        if np.dot(r, r) <= alpha * alpha * np.dot(s, s):
            return s, products
        denominator = np.dot(Ap, Ap)
        if denominator <= 0. or rr <= 0.:
            raise ArithmeticError("CR breakdown")
        coefficient = rr / denominator
        s = s + coefficient * p
        r = r - coefficient * Ap
        Ar = r + eta * _metric_action(w, Q, C, rank, scale, shift, r)
        products += 1
        next_rr = np.dot(r, Ar)
        beta = next_rr / rr
        p = r + beta * p
        Ap = Ar + beta * Ap
        rr = next_rr
    raise ArithmeticError("CR tolerance not attained")


@njit(cache=True, fastmath=False)
def _lanczos(w, Q, C, rank, iterations, start):
    dimension = len(start)
    basis = np.zeros((iterations, dimension)).T
    diagonal = np.zeros(iterations)
    off_diagonal = np.zeros(max(0, iterations - 1))
    q = start / math.sqrt(np.dot(start, start))
    previous = np.zeros(dimension)
    beta = 0.
    count = iterations
    for j in range(iterations):
        basis[:, j] = q
        residual = _action(w, Q, C, rank, q) - beta * previous
        diagonal[j] = np.dot(q, residual)
        residual -= diagonal[j] * q
        # Identical full reorthogonalization to the previous reproduction.
        for h in range(j + 1):
            residual -= np.dot(basis[:, h], residual) * basis[:, h]
        norm = math.sqrt(np.dot(residual, residual))
        if norm <= 1e-13 * max(1., abs(diagonal[j])) or j == iterations - 1:
            count = j + 1
            break
        off_diagonal[j] = norm
        previous = q
        q = residual / norm
        beta = norm
    T = np.zeros((count, count))
    for j in range(count):
        T[j, j] = diagonal[j]
        if j + 1 < count:
            T[j, j + 1] = off_diagonal[j]
            T[j + 1, j] = off_diagonal[j]
    values, vectors = np.linalg.eigh(T)
    low = basis[:, :count] @ np.ascontiguousarray(vectors[:, 0])
    high = basis[:, :count] @ np.ascontiguousarray(vectors[:, -1])
    return values[0], low, values[-1], high, count


@njit(cache=True, fastmath=False)
def _project_vector(Q, rank, vector):
    residual = vector.copy()
    coefficients = np.zeros(rank)
    for repeat in range(2):
        for j in range(rank):
            c = np.dot(Q[:, j], residual)
            coefficients[j] += c
            for i in range(len(residual)):
                residual[i] -= c * Q[i, j]
    return coefficients, residual


class _Matrix:
    def __init__(self, dimension):
        self.dimension = dimension
        self.rank = 0
        self.w = -1.
        self.Q = np.zeros((dimension, min(8, dimension)), order="F")
        self.C = np.zeros((min(8, dimension), min(8, dimension)))
        self.peak_bytes = self.Q.nbytes + self.C.nbytes

    def action(self, x):
        return _action(self.w, self.Q, self.C, self.rank, x)

    def _coordinates(self, vector):
        coefficients, residual = _project_vector(self.Q, self.rank, vector)
        norm = float(np.linalg.norm(residual))
        original_norm = float(np.linalg.norm(vector))
        # Only roundoff-level linear dependence is discarded.  This is not a
        # numerical low-rank truncation of a matrix or of its history.
        threshold = 32. * np.finfo(float).eps * max(1., math.sqrt(self.dimension)) * original_norm
        if norm > threshold and self.rank < self.dimension:
            if self.rank == self.Q.shape[1]:
                capacity = min(self.dimension, max(8, 2 * self.rank))
                expanded_Q = np.zeros((self.dimension, capacity), order="F")
                expanded_C = np.zeros((capacity, capacity))
                expanded_Q[:, :self.rank] = self.Q
                expanded_C[:self.rank, :self.rank] = self.C
                self.Q, self.C = expanded_Q, expanded_C
            self.Q[:, self.rank] = residual / norm
            coefficients = np.append(coefficients, norm)
            self.rank += 1
            self.peak_bytes = max(self.peak_bytes, self.Q.nbytes + self.C.nbytes)
        return coefficients

    def update(self, e, s, coefficient, separation_vector, separation_coefficient):
        e_coordinates = self._coordinates(e)
        s_coordinates = self._coordinates(s)
        sep_coordinates = self._coordinates(separation_vector) if separation_coefficient else np.empty(0)
        # Later insertions append zero coordinates to previously represented
        # vectors because the appended basis vectors are orthogonal to them.
        ec = np.zeros(self.rank); ec[:len(e_coordinates)] = e_coordinates
        sc = np.zeros(self.rank); sc[:len(s_coordinates)] = s_coordinates
        active = self.C[:self.rank, :self.rank]
        active += coefficient * (np.outer(ec, sc) + np.outer(sc, ec))
        if separation_coefficient:
            uc = np.zeros(self.rank); uc[:len(sep_coordinates)] = sep_coordinates
            active += separation_coefficient * np.outer(uc, uc)
        # Frobenius projection is exact for w I + Q C Q.T and avoids forming d².
        shifted = active.copy()
        shifted.flat[::self.rank + 1] += self.w
        norm_squared = (self.dimension - self.rank) * self.w * self.w + float(np.sum(shifted * shifted))
        norm = math.sqrt(max(0., norm_squared))
        factor = min(1., math.sqrt(self.dimension) / norm) if norm else 1.
        self.w *= factor
        active *= factor


def _separation(matrix, delta, probability, rng):
    d = matrix.dimension
    log_factor = math.log(11. * d / (probability * probability))
    count = min(d, int(math.ceil(log_factor + .5)))
    low, low_v, high, high_v, products = _lanczos(
        matrix.w, matrix.Q, matrix.C, matrix.rank, count, rng.normal(size=d))
    maximum = max(high, -low)
    if maximum <= .5:
        return 2. * maximum, np.zeros(d), 0., products
    if maximum >= 2.:
        return (2. * maximum, high_v, 3., products) if high > -low else (2. * maximum, low_v, -3., products)
    count = min(d, int(math.ceil(log_factor / (4. * math.sqrt(2. * delta)) + .5)))
    low, low_v, high, high_v, next_products = _lanczos(
        matrix.w, matrix.Q, matrix.C, matrix.rank, count, rng.normal(size=d))
    products += next_products
    maximum = max(high, -low)
    if maximum <= 1. - delta:
        return maximum + delta, np.zeros(d), 0., products
    return (maximum + delta, high_v, 1., products) if high >= -low else (maximum + delta, low_v, -1., products)


class _LocalStop(Exception):
    def __init__(self, status):
        self.status = status


class _ExternalFailure(BaseException):
    def __init__(self, original):
        self.original = original


def solve_aqnpe(evaluate, x0, eps, max_queries=100000, *, L1=None, seed=0):
    """Run supplied-L1 A-QNPE, or its explicitly labelled online-scale extension.

    The caller owns wall timing, FG histories and external resource limits.
    Exceptions raised by ``evaluate`` are deliberately never caught here.
    Additional matrix work is reported separately from FG evaluations.
    """
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    if max_queries < 1 or not np.isfinite(eps) or eps < 0.:
        raise ValueError("Invalid query budget or gradient tolerance")
    adaptive = L1 is None
    scale = 1. if adaptive else float(L1)
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError("L1 must be finite and positive")
    initial_scale = scale
    calls = 0
    minimum = None
    minimum_norm = np.inf
    last = None
    iterations = 0
    updates = 0
    resets = 0
    cr_products = 0
    sep_products = 0
    peak_rank = 0
    peak_state_bytes = 0

    def query(location):
        nonlocal calls, minimum, minimum_norm, last
        if calls >= max_queries:
            raise _LocalStop("query_budget")
        # Call count increases before evaluation, including a raised external
        # exception.  Such external exceptions propagate to the outer ledger.
        calls += 1
        try:
            f, gradient = evaluate(location)
        except BaseException as error:
            raise _ExternalFailure(error) from error
        point = (location.copy(), float(f), np.ascontiguousarray(gradient, dtype=np.float64))
        last = point
        norm = float(np.linalg.norm(point[2]))
        if not np.isfinite(f) or not np.isfinite(norm):
            raise _LocalStop("nonfinite_oracle")
        if norm < minimum_norm:
            minimum, minimum_norm = point, norm
        if norm <= eps:
            raise _LocalStop("stationarity")
        return point

    rng = np.random.RandomState(seed)
    matrix = _Matrix(len(x0))
    gamma_sep = 1.
    separation_vector = np.zeros(len(x0))
    separation_coefficient = 0.
    t = 0
    try:
        cached = query(x0)
        if adaptive:
            offset = rng.uniform(0., 1., len(x0))
            initial = query(x0 + offset)
            dg = initial[2] - cached[2]
            observed = math.sqrt(float(np.dot(dg, dg)) / float(np.dot(offset, offset)))
            scale = observed if observed > 0. and math.isfinite(observed) else 1.
            initial_scale = scale
        x_location = x0.copy()
        z = x0.copy()
        A = 0.
        eta = 1.
        while True:
            a = .5 * (eta + math.sqrt(eta * eta + 4. * eta * A))
            y_location = (A * x_location + a * z) / (A + a)
            y = cached if np.array_equal(y_location, cached[0]) else query(y_location)
            eta_trial = eta
            rejected = False
            while True:
                # B=(L1/2)(I+W/max(1,gamma)); original CR recurrence.
                normalizer = max(1., gamma_sep)
                displacement, products = _cr(
                    matrix.w, matrix.Q, matrix.C, matrix.rank,
                    .5 * scale / normalizer, .5 * scale, eta_trial, y[2], .25)
                cr_products += products
                candidate = query(y[0] + displacement)
                residual = displacement + eta_trial * candidate[2]
                if np.dot(residual, residual) <= .25 * np.dot(displacement, displacement):
                    break
                rejected = True
                last_s = displacement
                last_w = candidate[2] - y[2]
                eta_trial *= .5
                if eta_trial < 1e-300:
                    raise ArithmeticError("A-QNPE line search underflow")
            if not rejected:
                x_location = candidate[0]
                z -= a * candidate[2]
                A += a
                eta = 2. * eta_trial
            else:
                gamma = eta_trial / eta
                x_location = ((1. - gamma) * A * x_location + gamma * (A + a) * candidate[0]) / (A + gamma * a)
                z -= gamma * a * candidate[2]
                A += gamma * a
                eta = eta_trial
                ss = float(np.dot(last_s, last_s))
                if ss <= 0.:
                    raise ArithmeticError("A-QNPE zero rejected displacement")
                observed_scale = math.sqrt(float(np.dot(last_w, last_w)) / ss)
                if adaptive and observed_scale > scale:
                    # This is the sole mathematical change in online mode.
                    # Reset only the matrix learner; preserve MS state/steps.
                    scale *= 2. ** math.ceil(math.log2(observed_scale / scale))
                    peak_state_bytes = max(peak_state_bytes, matrix.peak_bytes)
                    peak_rank = max(peak_rank, matrix.rank)
                    matrix = _Matrix(len(x0))
                    gamma_sep = 1.
                    separation_coefficient = 0.
                    t = 0
                    resets += 1
                normalizer = max(1., gamma_sep)
                action_s = matrix.action(last_s) / normalizer
                error = .5 * scale * (last_s + action_s) - last_w
                inner = 4. / (scale * ss) * float(np.dot(error, action_s))
                correction = max(0., -inner) if gamma_sep > 1. else 0.
                matrix.update(error, last_s, -2. / (128. * scale * ss),
                              separation_vector, -correction * separation_coefficient / 128.)
                t += 1
                updates += 1
                delta = 1. / (math.sqrt(t + 2.) * math.log(t + 2.))
                probability = .01 / (2.5 * (t + 1.) * math.log(t + 1.) ** 2)
                gamma_sep, separation_vector, separation_coefficient, products = _separation(matrix, delta, probability, rng)
                sep_products += products
            cached = candidate
            iterations += 1
    except _ExternalFailure as failure:
        raise failure.original
    except _LocalStop as stopped:
        status = stopped.status
    except (ArithmeticError, OverflowError, FloatingPointError):
        status = "numerical_failure"
    peak_state_bytes = max(peak_state_bytes, matrix.peak_bytes)
    peak_rank = max(peak_rank, matrix.rank)
    point = last if status in ("stationarity", "nonfinite_oracle") else minimum
    if point is None:
        raise RuntimeError("A-QNPE stopped without a finite initial point")
    return dict(x=point[0], f=point[1], gradient=point[2], queries=calls,
                status=status, method="A-QNPE-online" if adaptive else "A-QNPE",
                iterations=iterations, matrix_updates=updates,
                scale_resets=resets, L1_initial=initial_scale, L1_final=scale,
                matrix_rank_peak=peak_rank, matrix_state_bytes_peak=peak_state_bytes,
                matrix_vector_products_cr=cr_products,
                matrix_vector_products_separation=sep_products)


def warmup():
    """Compile every numerical kernel on unrelated synthetic data, outside runs."""
    matrix = _Matrix(4)
    matrix.update(np.array([1., 2., 0., 0.]), np.array([0., 1., 1., 0.]), -.01,
                  np.array([0., 0., 0., 1.]), .001)
    _cr(matrix.w, matrix.Q, matrix.C, matrix.rank, .5, .5, 1., np.ones(4), .25)
    _separation(matrix, .1, .001, np.random.RandomState(0))
