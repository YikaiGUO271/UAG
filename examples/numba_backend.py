"""Optional native execution; compilation makes zero objective evaluations."""
import numpy as np
from numba import njit
from uag import minimize


@njit
def fg(x):
    diagonal = np.array([1., 3., 17., 63.])
    return .5*float(x @ (diagonal*x)), diagonal*x


if __name__ == "__main__":
    result = minimize(fg, [2., -1., .5, -2.], tol=1e-7, backend="numba")
    print(result.message)
    print(f"Setup/compile: {result.setup_seconds:.3f} s; solve: {result.elapsed_seconds:.3f} s")
    print(f"FG calls: {result.nfev}; gradient norm: {result.grad_norm:.3g}")
