# UAG

**Universal Accelerated Gradient Method for Unconstrained Smooth Optimization**

UAG is a research method for smooth unconstrained optimization using joint function-and-gradient (FG) evaluations. Its theoretical aim is a single execution that adapts to unknown convexity and higher smoothness. This repository contains an importable implementation and the experimental material accompanying the manuscript by Yikai Guo.

The contribution is primarily theoretical. The available experiments do not establish practical competitiveness with L-BFGS or nonlinear conjugate-gradient methods. Residual logarithmic factors and the complexity of the algorithm remain open issues.

## Install

Python 3.10 or newer is required. Install from a local checkout:

```console
git clone https://github.com/YikaiGUO271/UAG.git
cd UAG
python -m pip install .
```

The default backend needs only NumPy. For the optional compiled backend:

```console
python -m pip install ".[accelerated]"
```

The distribution name is `uag-optimization`; the import name is `uag`. These instructions install the repository directly; a PyPI release is not assumed.

## Use

```python
import numpy as np
from uag import minimize

diagonal = np.array([1.0, 3.0, 17.0])

def fg(x):
    gradient = diagonal * x
    return 0.5 * float(x @ gradient), gradient

result = minimize(fg, np.ones(3), tol=1e-7, max_evals=100_000)
print(result.success, result.message)
print(result.x, result.fun, result.grad_norm, result.nfev)
assert result.nfev == result.work.counts.sum()
```

`fg(x)` must return a real scalar and a gradient with the same shape as `x`. It must be deterministic and must not modify its input: historical replay relies on the same FG values at the same points. Reusing a gradient output buffer is supported. The initial point must be a finite, nonempty, one-dimensional vector. Computation uses float64.

The stopping test is the **absolute Euclidean gradient norm** `<= tol`. The float64 implementation accepts tolerances between `sqrt(numpy.finfo(float).tiny)` and `sqrt(numpy.finfo(float).max)` so that the squared stopping tolerance remains representable. No convexity label or smoothness bound is required. `initial_scale` is a positive starting guess, not a supplied Lipschitz bound.

### Compiled execution

```python
import numpy as np
from numba import njit
from uag import minimize

@njit
def compiled_fg(x):
    gradient = x.copy()
    return 0.5 * (x @ x), gradient

result = minimize(compiled_fg, np.ones(3), backend="numba", tol=1e-7)
print(result.setup_seconds, result.elapsed_seconds)
```

The compiled backend compiles the same solver function bodies used by the Python backend. It requires an explicit `@njit` FG callback returning `(float64, contiguous float64 vector)`. Compilation uses type signatures and makes no hidden objective calls. `setup_seconds` includes first-use compilation and initialization; `elapsed_seconds` measures the subsequent solve. Report both when startup cost matters.

The default native execution has no Python callback at each FG evaluation. Requesting a time limit or timed history uses a portable clock bridge and can add overhead. No library import changes global thread counts or environment settings.

## Results and accounting

The result contains `x`, `fun`, `jac`, `grad_norm`, `nfev`, `nit`, `success`, `status`, `message`, `work`, and timing fields. Returned arrays are independent, read-only snapshots.

- Every attempted FG call counts toward `max_evals`, including rejected trials, calibration, historical replay, and failing callbacks.
- `work.counts` separates four execution phases and eight query kinds. Its sum equals `nfev`; `work.replay_evals` and `work.secant_evals` expose useful subtotals.
- `nit` counts accepted outer updates. A terminal successful query need not increment it.
- On success, `x` is the first queried point meeting the tolerance. On an unsuccessful exit it is the finite queried point with the smallest gradient norm; if none exists, the function and gradient are NaN. Always check `success`.
- `record_history=True` stores FG index, elapsed time, function value and gradient norm, using O(`max_evals`) storage. `record_points=True` uses O(d `max_evals`) storage. Both are off by default, leaving O(d) solver working storage.
- `time_limit` is checked before an FG call and cannot interrupt an in-progress callback.

| Status | Meaning |
| --- | --- |
| 1 | Gradient tolerance reached |
| 2 | FG budget exhausted |
| 3 | Nonfinite function value or computed gradient norm |
| 4 | Time limit reached |
| 5 | Nonfinite trial point |
| 6 | Callback failure or invalid gradient |
| 7 | Numerical safeguard or internal invariant failure |

`secant=False` is an explicit diagnostic ablation. The manuscript method uses the default `secant=True`. Theoretical guarantees are exact-arithmetic statements under the manuscript assumptions. Detected floating-point failures return a failure status; the implementation does not certify exact-arithmetic behavior on every numerical scale.

## Reproduce the experiments

See [the experimental guide](docs/experiments.md) for fixed problem manifests, stopping rules, failure categories, source checksums, frozen observations and traces, profile regeneration, data preparation, and new runs. Historical observations are kept separately from new execution of this package.

The [26 October repetition campaign](experiments/repeated_20261026/README.md) contains four complete repetitions (1,828 observations), with unchanged comparison algorithms and settings. Manuscript profiles use median costs and require success in all four repetitions. A partial fifth repetition (44 observations) is retained separately and excluded from the main aggregate. Per-run traces, failures, interquartile ranges and sensitivity profiles are included.

The four logistic-regression datasets are downloaded separately from LIBSVM. Third-party benchmark provenance and licensing are documented in [third-party notices](docs/third_party.md). Saved manuscript results must not be interpreted as a speed benchmark of every backend or as repeated measurements unless explicitly labeled that way.

## Development

```console
python -m pip install -e ".[test,accelerated]"
python -m pytest
python -m pip wheel --no-deps .
```

Tests cover charged failures, replay, budgets, callback buffer reuse, backend agreement and fixed prefixes of the frozen research implementation. CI is configured to exercise installation and tests on Windows and Linux. CI results are evidence only after the corresponding workflow has run.

## License and citation

The original UAG package and original supporting code are released under the [MIT license](LICENSE). Separately identified third-party benchmark code retains its own license; MIT does not relicense externally obtained datasets. See [third-party notices](docs/third_party.md).

Please cite the accompanying manuscript, *Universal Accelerated Gradient Method for Unconstrained Smooth Optimization*, by Yikai Guo, and identify the software version or commit used. Software citation metadata is in [CITATION.cff](CITATION.cff).
