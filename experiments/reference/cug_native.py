"""Carmon et al. (2017), Algorithms 1--3 and Theorem 1.

https://proceedings.mlr.press/v70/carmon17a/carmon17a.pdf
Fixed supplied L1,L2; alpha=2*sqrt(L2*eps), eta=alpha/L2.
Every inner iteration performs CERTIFY-PROGRESS. Witnesses are recovered
by the O(d)-storage replay described after Corollary 1; replay FG are paid.
As for every comparator, any queried original gradient below eps stops.
"""
import math
import numpy as np
from numba import njit
import native_api as API


@njit(cache=False, fastmath=False)
def value(point, center, alpha):
    displacement = point[0] - center[0]
    return point[1] - center[1] + alpha * np.dot(displacement, displacement)


@njit(cache=False, fastmath=False)
def gradient(point, center, alpha):
    return point[2] + 2. * alpha * (point[0] - center[0])


@njit(cache=False, fastmath=False)
def witness(center, w, count, scale, alpha, omega, run):
    """Algorithm 1's scan, regenerated in its original order."""
    y = center
    x = center
    for j in range(count):
        gx = gradient(x, center, alpha)
        fx = value(x, center, alpha)
        for u in (y, w):
            displacement = u[0] - x[0]
            defect = (value(u, center, alpha) - fx - np.dot(gx, displacement)
                      - .5 * alpha * np.dot(displacement, displacement))
            if defect < 0.:
                return u, x, True
        if j + 1 < count:
            next_y = API.query(x[0] - gx / scale, run)
            if run.status != 0:
                return next_y, next_y, False
            next_x = API.query(next_y[0] + omega * (next_y[0] - y[0]), run)
            if run.status != 0:
                return next_x, next_x, False
            y, x = next_y, next_x
    return center, center, False


@njit(cache=False, fastmath=False)
def cug_native(x0, run, L1, L2):
    alpha = 2. * math.sqrt(L2 * run.eps)
    scale = L1 + 2. * alpha
    root_kappa = math.sqrt(scale / alpha)
    omega = (root_kappa - 1.) / (root_kappa + 1.)
    eta = alpha / L2
    center = API.query(x0, run)
    while run.status == 0:
        x = center
        previous_y = center
        best = center
        inner = 0
        while run.status == 0:
            gx = gradient(x, center, alpha)
            y = API.query(x[0] - gx / scale, run)
            if run.status != 0:
                break
            inner += 1
            if y[1] < best[1]:
                best = y
            # CERTIFY-PROGRESS precedes the regularized-gradient stop.
            gy = gradient(y, center, alpha)
            guilty = value(y, center, alpha) > 0.
            w = center
            if not guilty:
                z = API.query(y[0] - gy / scale, run)
                if run.status != 0:
                    break
                displacement = z[0] - center[0]
                psi = -value(z, center, alpha) + .5 * alpha * np.dot(displacement, displacement)
                guilty = np.dot(gy, gy) > 2. * scale * psi * math.exp(-inner / root_kappa)
                w = z
            if guilty:
                u, v, found = witness(center, w, inner, scale, alpha, omega, run)
                if run.status != 0:
                    break
                if not found:
                    run.status = 7
                    break
                if u[1] < best[1]:
                    best = u
                direction = u[0] - v[0]
                direction /= math.sqrt(np.dot(direction, direction))
                plus = API.query(u[0] + eta * direction, run)
                if run.status != 0:
                    break
                minus = API.query(u[0] - eta * direction, run)
                if run.status != 0:
                    break
                if plus[1] < best[1]:
                    best = plus
                if minus[1] < best[1]:
                    best = minus
                center = best
                break
            if math.sqrt(np.dot(gy, gy)) <= run.eps / 10.:
                center = y
                break
            x = API.query(y[0] + omega * (y[0] - previous_y[0]), run)
            previous_y = y
    return run.last if run.status == 1 else run.minimum
