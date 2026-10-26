import numpy as np
import pytest
numba = pytest.importorskip("numba")
from uag import minimize
from test_api import nonlinear, nonconvex

pytestmark = pytest.mark.accelerated
native_nonlinear = numba.njit(nonlinear)
native_nonconvex = numba.njit(nonconvex)


@pytest.mark.parametrize("secant", [False, True])
def test_backend_prefix_parity(secant):
    x0 = [2., -1., .5, -2.]
    options = dict(tol=1e-15, max_evals=128, secant=secant,
                   record_points=True, record_history=True)
    python = minimize(nonlinear, x0, **options)
    native = minimize(native_nonlinear, x0, backend="numba", **options)
    assert native.status == python.status == 2
    assert native.nfev == python.nfev == 128
    np.testing.assert_allclose(native.points, python.points, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(native.history[:, [0, 2, 3]], python.history[:, [0, 2, 3]], rtol=1e-11, atol=1e-13)
    np.testing.assert_array_equal(native.work.counts, python.work.counts)
    np.testing.assert_array_equal(native.work.events, python.work.events)


def test_native_nonconvex():
    result = minimize(native_nonconvex, [0.], backend="numba", tol=1e-7)
    assert result.success and result.grad_norm <= 1e-7


def test_compilation_does_not_evaluate_objective():
    @numba.njit
    def always_raises(x):
        if len(x) > 0:
            raise RuntimeError("paid callback")
        return 0., x.copy()
    result = minimize(always_raises, [1.], backend="numba", max_evals=0)
    assert result.nfev == 0 and result.status == 2
    result = minimize(always_raises, [1.], backend="numba")
    assert result.nfev == 1 and result.status == 6


def test_native_requires_explicit_njit_callback():
    with pytest.raises(TypeError, match="njit"):
        minimize(nonlinear, [1., 1., 1., 1.], backend="numba")


def test_wrong_native_gradient_size_is_charged():
    @numba.njit
    def bad(x):
        return 0., np.zeros(len(x)+1)
    result = minimize(bad, [1.], backend="numba")
    assert result.status == 6 and result.nfev == 1
