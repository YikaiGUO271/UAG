"""Run with ``python examples/quadratic.py`` after installing the package."""
import numpy as np
from uag import minimize


def fg(x):
    diagonal = np.array([1., 3., 17., 63.])
    return .5*float(x @ (diagonal*x)), diagonal*x


if __name__ == "__main__":
    result = minimize(fg, [2., -1., .5, -2.], tol=1e-7)
    print(result.message)
    print(f"FG calls: {result.nfev}; gradient norm: {result.grad_norm:.3g}")
    print(result.x)
