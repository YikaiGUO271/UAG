"""Direct-native-FG A-QNPE factory; no object-mode or Python callback boundary."""

import math
import numpy as np
from numba import njit, types
from numba.experimental import jitclass

from .solver import _action, _cr, _lanczos, _project_vector


@jitclass([
    ("dimension", types.int64), ("rank", types.int64),
    ("w", types.float64), ("Q", types.float64[::1, :]),
    ("C", types.float64[:, ::1]), ("peak_bytes", types.int64),
])
class NativeMatrix:
    def __init__(self, dimension):
        self.dimension = dimension
        self.rank = 0
        self.w = -1.
        capacity = min(8, dimension)
        # Explicit transpose of a C-contiguous allocation makes Q F-contiguous.
        self.Q = np.zeros((capacity, dimension)).T
        self.C = np.zeros((capacity, capacity))
        self.peak_bytes = self.Q.nbytes + self.C.nbytes

    def action(self, x):
        return _action(self.w, self.Q, self.C, self.rank, x)

    def coordinates(self, vector):
        coefficients, residual = _project_vector(self.Q, self.rank, vector)
        norm = math.sqrt(np.dot(residual, residual))
        original_norm = math.sqrt(np.dot(vector, vector))
        threshold = 32. * 2.220446049250313e-16 * max(1., math.sqrt(self.dimension)) * original_norm
        if norm > threshold and self.rank < self.dimension:
            if self.rank == self.Q.shape[1]:
                capacity = min(self.dimension, max(8, 2 * self.rank))
                expanded_Q = np.zeros((capacity, self.dimension)).T
                expanded_C = np.zeros((capacity, capacity))
                expanded_Q[:, :self.rank] = self.Q
                expanded_C[:self.rank, :self.rank] = self.C
                self.Q = expanded_Q
                self.C = expanded_C
            self.Q[:, self.rank] = residual / norm
            expanded = np.zeros(self.rank + 1)
            expanded[:self.rank] = coefficients
            expanded[self.rank] = norm
            coefficients = expanded
            self.rank += 1
            self.peak_bytes = max(self.peak_bytes, self.Q.nbytes + self.C.nbytes)
        return coefficients

    def update(self, e, s, coefficient, separation_vector, separation_coefficient):
        e_coordinates = self.coordinates(e)
        s_coordinates = self.coordinates(s)
        sep_coordinates = self.coordinates(separation_vector) if separation_coefficient != 0. else np.empty(0)
        ec = np.zeros(self.rank)
        sc = np.zeros(self.rank)
        uc = np.zeros(self.rank)
        ec[:len(e_coordinates)] = e_coordinates
        sc[:len(s_coordinates)] = s_coordinates
        uc[:len(sep_coordinates)] = sep_coordinates
        for i in range(self.rank):
            for j in range(self.rank):
                self.C[i, j] += coefficient * (ec[i] * sc[j] + sc[i] * ec[j])
                self.C[i, j] += separation_coefficient * uc[i] * uc[j]
        norm_squared = (self.dimension - self.rank) * self.w * self.w
        for i in range(self.rank):
            for j in range(self.rank):
                value = self.C[i, j]
                if i == j:
                    value += self.w
                norm_squared += value * value
        norm = math.sqrt(max(0., norm_squared))
        factor = min(1., math.sqrt(self.dimension) / norm) if norm != 0. else 1.
        self.w *= factor
        self.C *= factor


@njit(cache=True, fastmath=False)
def native_separation(matrix, delta, probability):
    d = matrix.dimension
    log_factor = math.log(11. * d / (probability * probability))
    count = min(d, int(math.ceil(log_factor + .5)))
    low, low_v, high, high_v, products = _lanczos(
        matrix.w, matrix.Q, matrix.C, matrix.rank, count, np.random.normal(0., 1., d))
    maximum = max(high, -low)
    if maximum <= .5:
        return 2. * maximum, np.zeros(d), 0., products
    if maximum >= 2.:
        if high > -low:
            return 2. * maximum, high_v, 3., products
        return 2. * maximum, low_v, -3., products
    count = min(d, int(math.ceil(log_factor / (4. * math.sqrt(2. * delta)) + .5)))
    low, low_v, high, high_v, next_products = _lanczos(
        matrix.w, matrix.Q, matrix.C, matrix.rank, count, np.random.normal(0., 1., d))
    products += next_products
    maximum = max(high, -low)
    if maximum <= 1. - delta:
        return maximum + delta, np.zeros(d), 0., products
    if high >= -low:
        return maximum + delta, high_v, 1., products
    return maximum + delta, low_v, -1., products


def make_native_solver(query):
    """Bind a compiled common ``query(x, runtime)`` function once.

    ``runtime`` has calls, status, minimum, last and other baseline-compatible
    fields.  Status 7 denotes a numerical failure.  A query-set status is
    checked immediately and the caller's stop/budget condition is preserved.
    """
    @njit(cache=False, fastmath=False)
    def aqnpe_native(x0, run, L1, seed, adaptive):
        np.random.seed(seed)
        dimension = len(x0)
        cached = query(x0, run)
        iterations = 0
        updates = 0
        resets = 0
        cr_products = 0
        sep_products = 0
        peak_rank = 0
        peak_state_bytes = 0
        scale = L1
        if run.status != 0:
            return cached, iterations, updates, resets, scale, peak_rank, peak_state_bytes, cr_products, sep_products
        if adaptive:
            # A paid seed-0 initial secant supplies a local scale only.  It is
            # deliberately not described as a certified global smoothness L.
            offset = np.random.uniform(0., 1., dimension)
            initial = query(x0 + offset, run)
            if run.status != 0:
                return initial, iterations, updates, resets, scale, peak_rank, peak_state_bytes, cr_products, sep_products
            dg = initial[2] - cached[2]
            observed = math.sqrt(np.dot(dg, dg) / np.dot(offset, offset))
            scale = observed if observed > 0. and math.isfinite(observed) else 1.
        matrix = NativeMatrix(dimension)
        gamma_sep = 1.
        separation_vector = np.zeros(dimension)
        separation_coefficient = 0.
        t = 0
        x_location = x0.copy()
        z = x0.copy()
        A = 0.
        eta = 1.
        while True:
            a = .5 * (eta + math.sqrt(eta * eta + 4. * eta * A))
            y_location = (A * x_location + a * z) / (A + a)
            y = cached if np.array_equal(y_location, cached[0]) else query(y_location, run)
            if run.status != 0:
                peak_state_bytes = max(peak_state_bytes, matrix.peak_bytes)
                peak_rank = max(peak_rank, matrix.rank)
                return y, iterations, updates, resets, scale, peak_rank, peak_state_bytes, cr_products, sep_products
            eta_trial = eta
            rejected = False
            while True:
                normalizer = max(1., gamma_sep)
                try:
                    displacement, products = _cr(
                        matrix.w, matrix.Q, matrix.C, matrix.rank,
                        .5 * scale / normalizer, .5 * scale, eta_trial, y[2], .25)
                except Exception:
                    run.status = 7
                    peak_state_bytes = max(peak_state_bytes, matrix.peak_bytes)
                    peak_rank = max(peak_rank, matrix.rank)
                    return run.minimum, iterations, updates, resets, scale, peak_rank, peak_state_bytes, cr_products, sep_products
                cr_products += products
                candidate = query(y[0] + displacement, run)
                if run.status != 0:
                    peak_state_bytes = max(peak_state_bytes, matrix.peak_bytes)
                    peak_rank = max(peak_rank, matrix.rank)
                    return candidate, iterations, updates, resets, scale, peak_rank, peak_state_bytes, cr_products, sep_products
                residual = displacement + eta_trial * candidate[2]
                if np.dot(residual, residual) <= .25 * np.dot(displacement, displacement):
                    break
                rejected = True
                last_s = displacement
                last_w = candidate[2] - y[2]
                eta_trial *= .5
                if eta_trial < 1e-300:
                    run.status = 7
                    return run.minimum, iterations, updates, resets, scale, peak_rank, peak_state_bytes, cr_products, sep_products
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
                ss = np.dot(last_s, last_s)
                if ss <= 0.:
                    run.status = 7
                    return run.minimum, iterations, updates, resets, scale, peak_rank, peak_state_bytes, cr_products, sep_products
                observed_scale = math.sqrt(np.dot(last_w, last_w) / ss)
                if adaptive and observed_scale > scale:
                    scale *= 2. ** math.ceil(math.log2(observed_scale / scale))
                    peak_state_bytes = max(peak_state_bytes, matrix.peak_bytes)
                    peak_rank = max(peak_rank, matrix.rank)
                    matrix = NativeMatrix(dimension)
                    gamma_sep = 1.
                    separation_coefficient = 0.
                    t = 0
                    resets += 1
                normalizer = max(1., gamma_sep)
                action_s = matrix.action(last_s) / normalizer
                error = .5 * scale * (last_s + action_s) - last_w
                inner = 4. / (scale * ss) * np.dot(error, action_s)
                correction = max(0., -inner) if gamma_sep > 1. else 0.
                matrix.update(error, last_s, -2. / (128. * scale * ss),
                              separation_vector, -correction * separation_coefficient / 128.)
                t += 1
                updates += 1
                delta = 1. / (math.sqrt(t + 2.) * math.log(t + 2.))
                probability = .01 / (2.5 * (t + 1.) * math.log(t + 1.) ** 2)
                gamma_sep, separation_vector, separation_coefficient, products = native_separation(matrix, delta, probability)
                sep_products += products
            cached = candidate
            iterations += 1
    return aqnpe_native
