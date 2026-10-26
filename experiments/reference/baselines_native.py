"""Bidirectional NAG (with gradient restart) and the author AC-FGM recurrence.

NAG restart: O'Donoghue and Candes, adaptive gradient restart, q=0.
Each iteration tries twice the last step, then backtracks as necessary.
AC-FGM: Li and Lan, doi:10.1007/s10107-025-02250-z; implementation:
https://github.com/tli432/AC-FGM-Implementation/blob/main/first_order_methods.py
The author's beta=.184, alpha=.1, and two consecutive t=2 conditions are kept.
"""

import math
import numpy as np
from numba import njit
import native_api as API


@njit(cache=False, fastmath=False)
def nag_native(x0, run, restart, explore):
    x = API.query(x0, run)
    if run.status != 0:
        return x
    y = x
    theta = 1.0
    while True:
        if explore:
            run.L = math.ldexp(run.L, -1)
            if run.L <= 0.0 or not math.isfinite(run.L):
                raise ArithmeticError("NAG exploratory scale underflowed")
        run.ell = min(run.ell, run.L)
        while True:
            location = y[0] - y[2] / run.L
            candidate = API.query(location, run)
            if run.status != 0:
                return candidate
            displacement = candidate[0] - y[0]
            model_change = np.dot(y[2], displacement) + 0.5 * run.L * np.dot(
                displacement, displacement
            )
            tolerance = 1e-15 * max(1.0, abs(y[1]), abs(candidate[1]))
            accepted_model = candidate[1] - y[1] <= model_change + tolerance
            if accepted_model and abs(model_change) <= tolerance:
                dg = candidate[2] - y[2]
                gradient_bound = run.L * math.sqrt(np.dot(displacement, displacement))
                accepted_model = np.dot(dg, dg) <= gradient_bound * gradient_bound
            if accepted_model:
                break
            run.L = math.ldexp(run.L, 1)
            if not math.isfinite(run.L):
                raise ArithmeticError("NAG backtracking scale overflowed")
        if restart == 0:
            trigger = np.dot(y[2], candidate[0] - x[0]) > 0.0
        elif restart == 1:
            trigger = candidate[1] > x[1]
        else:
            trigger = False
        if trigger:
            next_y = candidate
            theta = 1.0
        elif restart == 3:
            next_y = candidate
            theta = 1.0
        else:
            theta_next = (1.0 + math.sqrt(1.0 + 4.0 * theta * theta)) / 2.0
            momentum = (theta - 1.0) / theta_next
            if momentum == 0.0:
                next_y = candidate
            else:
                extrapolated = candidate[0] + momentum * (candidate[0] - x[0])
                if np.array_equal(extrapolated, candidate[0]):
                    next_y = candidate
                else:
                    next_y = API.query(extrapolated, run)
                    if run.status != 0:
                        return next_y
            theta = theta_next
        x = candidate
        y = next_y


@njit(cache=False, fastmath=False)
def acfgm_native(x0, run, initial_scale):
    current = API.query(x0, run)
    y = x0.copy()
    eta, local_scale, tau, tau_old = (0.0, initial_scale, 0.0, 0.0)
    beta, alpha = (0.184, 0.1)
    if run.status == 0:
        accepted = False
        for i in range(200):
            if run.calls >= run.max_queries:
                run.status = 2
                break
            local_scale = 1.5**i * initial_scale / 4.0
            eta = 1.0 / 2.5 / local_scale
            candidate = API.query(current[0] - eta * current[2], run)
            if run.status != 0:
                break
            difference = candidate[2] - current[2]
            displacement = candidate[0] - current[0]
            left = np.linalg.norm(difference) ** 2 / 2.0 / local_scale
            right = local_scale * np.linalg.norm(displacement) ** 2 / 2.0
            if left <= right:
                if i == 199:
                    run.status = 7
                else:
                    current = candidate
                    accepted = True
                break
        if run.status == 0 and (not accepted):
            run.status = 7
    t = 2
    while run.status == 0:
        if run.calls >= run.max_queries:
            run.status = 2
            break
        if t == 2:
            eta = min((1.0 - beta) * eta, 1.0 / 4.0 / local_scale)
            tau_old, tau = (0.0, 1.0)
        if local_scale > 0.0:
            eta = min(4.0 / 3.0 * eta, (tau_old + 1.0) / tau * eta, tau / 4.0 / local_scale)
        tau_old = tau
        tau = tau + 2.0 * (1.0 - alpha) * eta * local_scale / tau + alpha / 2.0
        if not (math.isfinite(eta) and eta > 0.0 and math.isfinite(tau)):
            run.status = 7
            break
        z = y - eta * current[2]
        y = (1.0 - beta) * y + beta * z
        candidate = API.query((z + tau * current[0]) / (1.0 + tau), run)
        if run.status != 0:
            break
        difference_norm = np.linalg.norm(candidate[2] - current[2])
        if difference_norm == 0.0:
            local_scale = 0.0
        else:
            denominator = 2.0 * (
                current[1] - candidate[1] - np.dot(candidate[2], current[0] - candidate[0])
            )
            if denominator == 0.0:
                run.status = 7
                current = candidate
                break
            local_scale = difference_norm**2 / denominator
            if local_scale < 0.0:
                local_scale = 0.0
            if not math.isfinite(local_scale):
                run.status = 7
                current = candidate
                break
        current = candidate
        t += 1
    return run.last
