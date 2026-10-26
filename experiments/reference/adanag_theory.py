"""Suh--Ma AdaNAG, arXiv:2505.11670v1, Algorithm 1 (theory coefficients).

The initial step uses the exact r_0 in Lemma 8, rather than its rounded
0.4255 display in Theorem 1.  The only paid evaluations go through the
supplied shared native ``query``.  There is no near-zero residual clipping,
line search, nonconvex fallback, restart, or empirical gamma parameter.

Factory signature matches the formal native runtime:
    solve = make_native_solver(native_api.query)
    solve(x0, run, initial_offset)

Status 7 denotes an invalid/degenerate numerical initialization or step;
status 8 denotes a negative convex Bregman residual.  The runtime owns
stationarity, oracle, budget, and time-limit statuses.  The paper guarantees
the method on smooth convex problems with a minimizer, not general NC ones.
"""

import math

import numpy as np
from numba import njit


METHOD_NAME = "AdaNAG-Alg1-theory"
PAPER_ID = "arXiv:2505.11670v1"


def _initial_coefficients():
    theta = [1.0]
    for _ in range(5):
        theta.append(0.5 * (1.0 + math.sqrt(1.0 + 4.0 * theta[-1] ** 2)))
    alpha1, alpha2, alpha3 = [
        0.5 * (1.0 - 1.0 / theta[j + 2]) for j in (1, 2, 3)
    ]
    alpha0 = (2.0 * theta[2] / (theta[2] - 1.0)) / (
        1.0 / alpha3 + 1.0 / alpha2**2 - 1.0 / alpha1
    )
    c0 = alpha2**2 * alpha3 / (alpha3 + alpha2**2)
    r0 = theta[3] * (theta[3] - 1.0) / theta[2] * c0 / alpha0
    first_growth = alpha0 / alpha1 * theta[2] / (theta[3] * (theta[3] - 1.0))
    return theta[2], theta[3], alpha0, c0, r0, first_growth, c0 / alpha1


(THETA2, THETA3, ALPHA0, C0, R0, FIRST_GROWTH,
 FIRST_CURVATURE_COEFFICIENT) = _initial_coefficients()


@njit(cache=False, fastmath=False)
def next_step_from_residual(step, growth, coefficient, residual, gradient_difference_sq):
    """Equation (7), with the paper's exact zero-division conventions.

    ``coefficient / L`` equals ``2*coefficient*residual/||dg||^2``.
    For ||dg||=0 and residual>=0, L=0, including 0/0=0, so this cap is inf.
    A zero residual with nonzero ||dg|| violates the smooth-convex lemma
    and would give a zero step; it is reported rather than patched.
    Returns (step_next, status); status=0 means valid.
    """
    if (not math.isfinite(residual) or not math.isfinite(gradient_difference_sq)
            or gradient_difference_sq < 0.0):
        return 0.0, 7
    if residual < 0.0:
        return 0.0, 8
    if gradient_difference_sq == 0.0:
        curvature_step = np.inf
    elif residual == 0.0:
        return 0.0, 7
    else:
        curvature_step = 2.0 * coefficient * residual / gradient_difference_sq
    step_next = min(growth * step, curvature_step)
    if step_next <= 0.0 or not math.isfinite(step_next):
        return step_next, 7
    return step_next, 0


def make_native_solver(query):
    """Bind Algorithm 1 to the same paid FG boundary as every comparator."""

    @njit(cache=False, fastmath=False)
    def solve(x0, run, initial_offset):
        x = query(x0, run)
        if run.status != 0:
            return x
        initial = query(x0 + initial_offset, run)
        if run.status != 0:
            return initial

        # Equation (3) uses the actual queried displacement.  No unqueried
        # estimate, global-L input, or fallback scale enters Algorithm 1.
        displacement0 = initial[0] - x[0]
        dg0 = initial[2] - x[2]
        displacement0_sq = np.dot(displacement0, displacement0)
        dg0_sq = np.dot(dg0, dg0)
        if (displacement0_sq <= 0.0 or dg0_sq <= 0.0
                or not math.isfinite(displacement0_sq) or not math.isfinite(dg0_sq)):
            run.status = 7
            return run.minimum
        L0 = math.sqrt(dg0_sq) / math.sqrt(displacement0_sq)
        if L0 <= 0.0 or not math.isfinite(L0):
            run.status = 7
            return run.minimum
        step = R0 / L0
        if step <= 0.0 or not math.isfinite(step):
            run.status = 7
            return run.minimum

        z = x[0].copy()
        theta_current, theta_next = THETA2, THETA3
        alpha = ALPHA0
        k = 0
        while True:
            # At loop entry: theta_current=theta_{k+2}, theta_next=theta_{k+3}.
            y = x[0] - step * x[2]
            z = z - step * alpha * theta_current * x[2]
            next_x = query((1.0 - 1.0 / theta_next) * y + z / theta_next, run)
            if run.status != 0:
                return next_x

            displacement = next_x[0] - x[0]
            dg = next_x[2] - x[2]
            residual = x[1] - next_x[1] + np.dot(next_x[2], displacement)
            alpha_next = 0.5 * (1.0 - 1.0 / theta_next)
            if k == 0:
                growth = FIRST_GROWTH
                coefficient = FIRST_CURVATURE_COEFFICIENT
            else:
                growth = alpha / alpha_next
                coefficient = alpha**2 / (alpha_next + alpha**2)
            step, status = next_step_from_residual(
                step, growth, coefficient, residual, np.dot(dg, dg)
            )
            if status != 0:
                run.status = status
                return run.minimum

            x = next_x
            alpha = alpha_next
            theta_current = theta_next
            theta_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * theta_current**2))
            k += 1

    return solve


# Descriptive export used by the formal campaign; both names bind the same code.
make_adanag_theory_solver = make_native_solver
