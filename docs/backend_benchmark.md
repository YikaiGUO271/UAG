# Development check of backend execution costs

Run `tools/benchmark_backends.py` in an idle process after formal timing ends:

```console
python -m pip install -e ".[accelerated]" threadpoolctl
python tools/benchmark_backends.py --output artifacts/backend_benchmark.json
```

This compares ordinary NumPy and native Numba execution of the new UAG package.
It is a development implementation-cost check, not a paper benchmark, a
comparison with other algorithms, or evidence of universal practical superiority.

Defaults: dimensions 4/256/8192, seven repetitions, tolerance `1e-12`, initial
scale 1, and 32 FG evaluations. The objective is `0.5 * dot(x,d*x)`, with
`d=linspace(1,64,dimension)` and `x0[j]=(-1)^j/sqrt(dimension)`. Both backends
use the same expression and full UAG secant proposals. History and query-point
recording are disabled. At most 1344 development FG calls are made by default.

Backend order alternates. All evaluations, including rejected trials and
replay, are charged; no warm-up solves are hidden. Zero-budget preparations
compile types without calling the objective. Their time is recorded separately.
Repeated measurements separate setup, solve, and complete public-call time,
with median, quartiles and IQR; first-run observations remain in the report.

The standalone process sets and verifies one computational thread. The library
does not change thread settings or environment variables. Every result is
checked against the first Python result and its paired repetition: status,
evaluations, accepted outer updates, all 32 FG categories, all event counters,
and returned point/function/gradient. Endpoint tolerances are `rtol=1e-10`,
`atol=1e-12`. The report includes discrepancies and endpoint hashes. A failed
check suppresses that case's timing ratio.

The default 115-second allowance is checked between public calls. A running
call may finish after it; this is a short-run target, not a verified runtime
bound. Partial records are saved without ratios for incomplete cases. Exit
codes are 0 for all checks passed, 1 for a completed verification discrepancy,
and 2 for incomplete measurements. Increase `--max-wall-seconds` only outside
protected timing windows.

The historical implementation is not timed here. Separate 128-evaluation
prefix validation is recorded in `tests/data/frozen_prefixes.json`; it does not
establish a speed improvement over the historical version. Native compilation
can dominate short solves and must be considered separately.

The saved `artifacts/backend_benchmark.json` records the completed 26 October
2026 development check on Windows/Python 3.13.5. All 42 solves and 1344 charged
FG calls completed, and all endpoint, status and accounting checks passed.
Compilation made zero FG calls and its elapsed cost is reported separately.
These short diagonal-quadratic measurements describe this implementation and
machine only; they do not replace the manuscript comparison campaign.
