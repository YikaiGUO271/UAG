# Experiment reproduction

The earlier results in `experiments/frozen` are the unchanged single-observation campaign of 24 October 2026: 9 concrete problems with 8 methods and 55 fixed CUTEst problems with 7 methods, for 457 observations. Historical files retain the method identifier `UAD`; figures display its current name, `UAG`. The importable package is a separate, validated implementation. New runs are never silently substituted for saved measurements.


The final manuscript profiles use the separate 26 October repetition campaign in `experiments/repeated_20261026`: four complete repetitions (1828 observations) and a separately retained fifth partial repetition (44 observations). Five repetitions were planned; the fixed UTC 11:30 deadline stopped new observations after the fourth complete repetition, while the current observation completed under its original soft cap at UTC 11:31:04. The partial repetition is excluded from common-repetition statistics. The original native Windows runtime, methods, parameters, data, starts, targets and per-run budgets were unchanged. The exact executed driver is saved as `driver_used.py`; the public rerun driver has subsequently received portability and metadata corrections.

To audit and redraw the current manuscript profiles without objective calls:

```sh
python experiments/audit_repetitions.py experiments/repeated_20261026
python experiments/analyze_repetitions.py experiments/repeated_20261026 --output experiments/generated/repeated-profiles
```

Each complete repetition has 457 entries and the same case-method set SHA256 `237746c98ee1497cd0db14f8f7f319950879e33860c83cbd94fa4c5a858dcdd0`. All 1872 traces were audited, including the partial repetition: FG counters and targets match the protocol, and all 1557 successful observations stop at the first recorded target hit. The 133 originally frozen source files also retained their recorded SHA256 values. Per-repetition ZIP archives retain the exact NPZ bytes without additional compression. `archive_manifest.json` and `SHA256.json` cover these public artifacts.

Under the preregistered all-four-success rule, UAG solves all 9 concrete and all 55 CUTEst cases. It has the smallest FG count and median time on all 9 concrete cases, and on 52 and 50 CUTEst cases, respectively. There are no ties for the best cost in these aggregates; profile comparisons use relative tolerance 1e-12. The sums of per-problem median UAG costs are 7422 FG / 3.49955 seconds (concrete) and 50044 FG / 4.54057 seconds (CUTEst). These are not medians of suite totals. Full per-method values are in `analysis/analysis.json` and per-problem medians/IQRs in `analysis/repetition_details.csv`.

Timing variability remains visible: across problems, UAG's time IQR divided by its median ranges from 1.17% to 92.23% (median 31.29%) on the concrete suite and 1.68% to 62.23% (median 16.48%) on CUTEst. The absolute times of short observations should be interpreted accordingly. Across all methods, 12 problem-method combinations have different FG counts across repetitions; none has differing FG counts after time-limit observations are removed. The complete repetitions contain 1528 successes, 156 query-budget terminations, 46 time limits, 90 convexity-residual terminations and 8 numerical failures. The status change on AdaNAG/w8a is caused by reaching the time limit before the otherwise observed residual termination, not by a changed recurrence.

## Installation and saved-result checks

From the repository root:

```sh
python -m pip install -e ".[accelerated]"
python -m pip install -r experiments/requirements-frozen.txt
python experiments/audit_saved.py --check-traces
python experiments/regenerate_profiles.py
```

The recorded campaign used Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, Numba 0.61.0, Matplotlib 3.10.0 and threadpoolctl 3.5.0 on Windows 11 / Intel Core Ultra 9 275HX, with one numerical thread. The requirements file records this environment; it is not a claim that other versions give bitwise identical floating-point trajectories or timings.

`audit_saved.py` verifies the frozen file checksums, all 457 method/problem entries, and, with `--check-traces`, every original NPZ byte stream, FG sequence, FG category sum, monotone timestamps, terminal target and success flag. `frozen/traces.zip` contains every original trace without changing its NPZ bytes. `manifests/trace_hashes.json` records the individual SHA256 values. The full archive is approximately 55 MiB.

`regenerate_profiles.py` reconstructs both monochrome paper profiles, including all failed observations in the denominator. Every cost ratio is checked against the saved analysis. It makes no objective calls. Output defaults to `experiments/generated`; `--output` selects another non-frozen directory. PDF metadata timestamps can differ, while curves and styles are unchanged.

## Problems, data and fixed parameters

`manifests/cases.json` contains the complete list of 64 cases, dimensions, starting-point hashes, smoothness constants supplied to the baselines, seeds, construction parameters, initial gradients, CUTEst constructor arguments and source hashes. `frozen/protocol.json` records the original method sets, budgets and source hashes. Paths inside that historical metadata describe the original experiment layout; executable scripts resolve dependencies within this repository.

The three LogSumExp instances share NumPy `default_rng(0)` Gaussian data with 1000 rows and 200 columns and use temperatures 0.5, 0.05 and 0.005; all start at zero. Student-t regression uses seed 0, 500 rows, 200 columns, nu=0.001, and a least-squares initial point. Rosenbrock uses dimension 200 and a zero initial point. Its supplied smoothness bounds concern the documented enclosing rectangle, not global smoothness.

The four logistic instances use signed LIBSVM CSR rows `a_i=-y_i*z_i`, regularization `1/m`, and a zero initial point. No intercept, additional scaling, shuffle or train/test merge is introduced. Source dataset filenames, raw-file hashes, array hashes, dtypes and shapes are in `manifests/logistic_data.json`. The dataset files are not redistributed here. Acquire them separately after reviewing the source terms:

```sh
python experiments/prepare_logistic.py --download
```

Alternatively, supply already acquired source files with `--raw-dir PATH`. The script verifies both downloaded bytes and every resulting signed CSR array against the frozen inputs, then writes `experiments/cases/data`. It refuses to replace existing files. NPZ container timestamps may differ; the numeric arrays must match exactly. The preparation path was exercised on all four downloads and all array hashes matched. These checks do not confer redistribution rights over the datasets.

The 55 CUTEst evaluators are the exact generated Numba FG code and parameter arrays used by the frozen campaign, lowered from the official S2MPJ Python translations at commit `fea6a70048eaad28b13a08703ddbfdbf65cd9c30`. They are not the Fortran CUTEst runtime. The upstream BSD licence is retained in `experiments/third_party/S2MPJ_LICENSE.txt`; generation hashes and original validation are in `manifests/cutest_generated_validation.json`. Original constructor parameters and upstream source hashes are listed per problem. No new instance selection is performed during a rerun.

## New runs of the package and reference baselines

A short package check, written into a fresh directory:

```sh
python experiments/run_package.py --problems rosenbrock_d200 --max-evals 256 --output experiments/generated/package-smoke
python experiments/run_package.py --suite cutest --problems ARWHEAD --max-evals 256 --output experiments/generated/package-cutest-smoke
```

These capped checks exercise the implementation; they are not full benchmark results. Omit the cap override to use the paper budgets. The package command records setup separately from solver time, uses the native combined FG oracle, checks the returned point externally, and records every paid FG when history is requested. Select `--backend python` to exercise the Python backend instead; it is a different runtime and must not be presented as a repeat of the native timing campaign.

The portable reference driver includes the published comparison recurrences and the historical UAG recurrence:

```sh
python experiments/repeat_campaign.py --suite specific --problems rosenbrock_d200 --methods UAD GD --max-fg 256 --repeats 1 --output experiments/generated/reference-smoke
python experiments/repeat_campaign.py --repeats 5 --output experiments/generated/repeated-campaign
```

The second command runs 5 complete repetitions of the original comparison set. Prepare the four logistic datasets first. Output must be a new directory; frozen observations, source code, generated cases and manifests are protected. Diagnostics with selected methods/problems or overridden budgets are explicitly labelled in their protocol. The driver uses one process and one numerical thread, rotates and reverses method order across repetitions, records warm-up and initial/endpoint checks separately, and retains all failures and traces. A monotonic Python clock bridge replaces the original Windows-only clock, so these portable-run timings are new measurements. `--historical-root PATH` is an explicit archival option for a locally retained complete 2026-10-24 campaign; it imports that unmodified original runtime, verifies all original source hashes and never replaces its result, protocol, source or trace files. The historical runtime may access its original JIT cache. No such outside directory is required for the default portable commands. On Windows, add `--clock windows-native` to use the bundled exact historical QPC clock with the repository-local reference solver. This mode requires no historical workspace. `reference/native_api_windows.py` retains the original SHA256 `e5d9a85003e3e9f4a32a50df1acea07fbd0288cc9ab6e2b5cc408225693b765a`; the portable default uses `reference/native_api.py`. Neither mode silently replaces the other in recorded metadata. The portable CUTEst loader makes zero FG calls; its separately charged initial-gradient check makes one. The historical loader makes an additional setup FG call, recorded separately.

The concrete targets are gradient norm <=1e-7, 100000 FG calls and a 120-second soft limit. CUTEst targets are <=1e-6*max(1, initial gradient norm), 20000 FG calls and a 30-second soft limit. Every FG checks the soft time limit. JIT compilation, object/data construction and endpoint checks are excluded from solver timing. Warm-up records, setup-check counts and endpoint-check counts are saved separately; these are not a complete end-to-end preparation-time measurement. All rejected trials and replay FG calls are charged. Machine load and clock boundaries matter; a repeat on another machine is not expected to reproduce the saved seconds.

## Comparator settings and scope

The source files in `experiments/reference` expose every constant. All methods start with the same point and use the same combined FG boundary. NAG, RNAG and GD use bidirectional line search with exploratory doubling of the step and common model tests; RNAG uses the gradient restart. AC-FGM retains beta=0.184, alpha=0.1 and the author's two initial t=2 conditions. CUG is supplied the fixed L1/L2 bounds and is used only for the nine concrete problems. AdaNAG is Suh--Ma Algorithm 1 with the exact Lemma 8 initialization coefficient and original recurrence, initialized by a paid `x0+default_rng(0).uniform(0,1,d)` query; there is no near-zero keep-step heuristic or nonconvex fallback. A-QNPE uses alpha1=alpha2=1/4, beta=1/2, sigma0=1, B0=0, rho=1/128, p=0.01, seed 0 for Lanczos, and the fixed schedules in its source. It is supplied L1 on concrete problems. Its CUTEst variant uses the labelled online-scale extension and does not inherit the original known-L1 theorem. AdaNAG and A-QNPE on nonconvex cases are empirical applications of the unchanged recurrences.

There are no L-BFGS or nonlinear conjugate-gradient observations in this comparison. The benchmark does not establish practical superiority to those widely used methods. Mechanism ablations are also not supplied by these unchanged-comparison repeats.

## Failures and repetition statistics

The frozen 457 observations contain 382 successful runs and 75 unsuccessful runs: 39 query-budget terminations, 11 soft time limits, 23 nonpositive convexity-residual terminations, and 2 numerical failures. `audit_saved.py` emits per-method counts and every unsuccessful row. A status is a recorded stopping reason, not a diagnosis that the mathematical method cannot converge. In particular, the convexity-residual status comes from the unchanged AdaNAG recurrence and is not silently repaired on nonconvex problems. Nonfinite oracles/iterates, oracle exceptions and endpoint-verification failures remain distinct supported categories even if their count is zero in this campaign.

New repetition analyses use only complete repetitions shared by every method/problem. A method/problem is solved only if all R/R repetitions succeed; successful costs are the medians of all R observations, otherwise profile costs are infinite. Every status, success count, median and interquartile range is retained. Profiles over problem-by-repetition instances are also saved as a sensitivity check. If a time deadline stops an incomplete repetition, that partial set remains available but is excluded from the common-R aggregate. The unchanged single-observation campaign remains separate.

Analyze a completed (or deadline-stopped) repetition campaign with:

```sh
python experiments/audit_repetitions.py experiments/generated/repeated-campaign
python experiments/analyze_repetitions.py experiments/generated/repeated-campaign
```

This writes median-cost monochrome profiles, per-repetition sensitivity profiles, all status counts, per-combination medians/IQRs, best-cost counts, sums of per-problem median observed costs, the case-method set hash, and both full and non-time-limit FG-variation lists. Diagnostic or unlabelled campaigns are refused. The analysis refuses to aggregate until at least one complete 457-observation repetition is present.

After a repeated campaign stops, `python experiments/audit_repetitions.py CAMPAIGN` verifies every new trace, including any partial final repetition, and the exact launched driver hash. The audit accepts raw NPZ files or the per-repetition ZIP archives distributed with the results.


## Distribution validation

The downloaded-data recipe was exercised on all four datasets and reproduced every frozen signed CSR array hash. The repository-local reference smoke runs cover portable-clock UAG/GD on Rosenbrock and Windows-native-clock UAG on ARWHEAD. Their protocols explicitly mark them as diagnostics. Importable-package checks reached the target in 164 FG on Rosenbrock and 27 FG on ARWHEAD; all paid objective/gradient trace values, endpoints and phase/kind counts matched the original frozen traces bitwise. Their timings are implementation checks, not additional formal repetitions. Evidence is retained in `experiments/validation`, while acquired dataset files remain excluded from the repository.
