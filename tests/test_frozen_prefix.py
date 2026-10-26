"""Regression against a frozen formal implementation's recorded prefix.

The JSON beside the fixture records reference source hashes and the protocol.
Time columns are excluded because they are not mathematical trajectories.
"""
from pathlib import Path
import numpy as np
import pytest
from uag import minimize
from test_api import nonlinear


@pytest.mark.parametrize("secant", [False, True])
def test_frozen_128_evaluation_prefix(secant):
    prefix = "secant_" if secant else "core_"
    with np.load(Path(__file__).parent / "data/frozen_prefixes.npz") as data:
        result = minimize(nonlinear, [2., -1., .5, -2.], tol=1e-15,
                          max_evals=128, secant=secant, record_history=True,
                          record_points=True)
        np.testing.assert_allclose(result.points, data[prefix+"points"], rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(result.history[:, [0, 2, 3]], data[prefix+"values"], rtol=1e-11, atol=1e-13)
        np.testing.assert_array_equal(result.work.counts.ravel(), data[prefix+"counts"])
        np.testing.assert_array_equal(result.work.events, data[prefix+"events"])
