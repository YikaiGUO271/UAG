from dataclasses import FrozenInstanceError
import time
import numpy as np
import pytest
from uag import minimize


def quadratic(x):
    diagonal = np.array([1., 3., 17., 63.])
    return .5*float(x @ (diagonal*x)), diagonal*x


def nonlinear(x):
    diagonal = np.array([1., 3., 17., 63.])
    return .5*float(x @ (diagonal*x)) + .2*np.sum(1.-np.cos(x)), diagonal*x+.2*np.sin(x)


def nonconvex(x):
    return .5*float(x@x) + 2.*np.cos(x[0]) + .02*x[0], np.array([x[0]-2.*np.sin(x[0])+.02])


def test_convergence_and_charged_work():
    calls = []
    def fg(x):
        calls.append(x.copy())
        return quadratic(x)
    result = minimize(fg, [2., -1., .5, -2.], tol=1e-7,
                      record_history=True, record_points=True)
    assert result.success and result.grad_norm <= 1e-7
    assert result.nfev == len(calls) == result.work.counts.sum()
    assert result.nfev > result.nit
    assert result.work.secant_evals > 0
    np.testing.assert_array_equal(result.points, calls)
    np.testing.assert_array_equal(result.history[:, 0], np.arange(1, result.nfev+1))
    assert np.all(np.diff(result.history[:, 1]) >= 0)
    assert result.setup_seconds >= 0 and result.elapsed_seconds >= 0


@pytest.mark.parametrize("budget", [0, 1, 2, 7, 21])
def test_hard_budget_includes_failed_trials(budget):
    calls = 0
    def fg(x):
        nonlocal calls
        calls += 1
        return nonlinear(x)
    result = minimize(fg, [2., -1., .5, -2.], tol=1e-15, max_evals=budget)
    assert result.status == 2 and not result.success
    assert calls == result.nfev == budget == result.work.counts.sum()
    if not budget:
        assert np.isnan(result.fun) and np.isnan(result.jac).all()
    else:
        assert np.isfinite(result.fun)
    assert result.history is None and result.points is None


def test_reused_gradient_and_alias_input_are_owned():
    gradient = np.empty(4)
    x0 = np.array([2., -1., .5, -2.])
    before = x0.copy()
    def reused(x):
        value, grad = nonlinear(x)
        gradient[:] = grad
        return value, gradient
    a = minimize(reused, x0, tol=1e-9, max_evals=128, record_points=True)
    b = minimize(nonlinear, x0, tol=1e-9, max_evals=128, record_points=True)
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.jac, b.jac)
    np.testing.assert_array_equal(x0, before)
    gradient[:] = 99.
    assert not np.all(a.jac == 99.)
    alias = minimize(lambda x: (.5*float(x@x), x), [1., 2.])
    assert alias.success
    with pytest.raises(ValueError):
        a.x[0] = 10.
    with pytest.raises(FrozenInstanceError):
        a.status = 2


@pytest.mark.parametrize("bad_fg,status", [
    (lambda x: (np.nan, x.copy()), 3),
    (lambda x: (1., np.full(len(x), np.inf)), 3),
    (lambda x: (1., np.zeros(len(x)+1)), 6),
    (lambda x: (1., np.zeros((len(x), 1))), 6),
    (lambda x: (1.+1j, x), 6),
    (lambda x: (_ for _ in ()).throw(RuntimeError("callback failure")), 6),
])
def test_invalid_callback_paid_and_reported(bad_fg, status):
    result = minimize(bad_fg, [1., 2.])
    assert result.status == status and result.nfev == 1
    assert not result.success and np.isnan(result.fun)
    assert result.work.counts.sum() == 1


def test_callback_cannot_modify_point():
    def fg(x):
        x[0] = 2.
        return .5*float(x@x), x.copy()
    result = minimize(fg, [1., 2.])
    assert result.status == 6 and "read-only" in result.message


@pytest.mark.parametrize("x,kwargs", [
    ([], {}), ([[1.]], {}), ([np.nan], {}), ([1j], {}),
    ([1.], {"tol": 0.}), ([1.], {"tol": np.inf}),
    ([1.], {"max_evals": -1}), ([1.], {"max_evals": 1.5}),
    ([1.], {"max_evals": True}), ([1.], {"initial_scale": 0}),
    ([1.], {"backend": "automatic"}), ([1.], {"time_limit": -1}),
    ([1.], {"record_history": "yes"}),
])
def test_invalid_input_never_calls_objective(x, kwargs):
    def fg(x):
        pytest.fail("validation called the objective")
    with pytest.raises((TypeError, ValueError)):
        minimize(fg, x, **kwargs)


@pytest.mark.parametrize("tol", [1e-250, 1e200])
def test_tolerance_square_outside_float64_range_is_rejected(tol):
    # With the tiny tolerance, dot([1e-200], [1e-200]) would round to zero
    # and falsely declare this nonstationary starting point successful.
    calls = []
    def fg(x):
        calls.append(x.copy())
        return float(1e-200*x[0]), np.array([1e-200])

    with pytest.raises(ValueError, match="squared tolerance"):
        minimize(fg, [1.], tol=tol)
    assert not calls


def test_time_budget_is_checked_between_calls():
    calls = 0
    def fg(x):
        nonlocal calls
        calls += 1
        time.sleep(.025)
        return nonlinear(x)
    result = minimize(fg, [2., -1., .5, -2.], time_limit=.01)
    assert result.status == 4 and result.nfev == calls == 1
    assert np.isfinite(result.fun)


@pytest.mark.parametrize("secant", [False, True])
def test_nonconvex_stationarity(secant):
    result = minimize(nonconvex, [0.], tol=1e-7, secant=secant, max_evals=5000)
    assert result.success and result.grad_norm <= 1e-7
    assert result.fun < 2.


def test_stationary_initial_point_is_charged_once():
    result = minimize(quadratic, np.zeros(4))
    assert result.status == 1 and result.nfev == 1 and result.nit == 0


def test_numerical_error_preserves_charged_work():
    # A nonstationary linear objective cannot end the downward calibration;
    # eventually its search scale underflows. The paid prefix is retained.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = minimize(lambda x: (float(x[0]), np.ones(1)), [0.],
                          initial_scale=np.finfo(float).tiny, max_evals=100)
    assert result.status in (3, 5, 7)
    assert result.nfev == result.work.counts.sum()
