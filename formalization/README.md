# Partial formalization of UAG

This directory contains Lean proofs of selected results supporting *Universal Accelerated Gradient Method for Unconstrained Smooth Optimization*, by Yikai Guo. All five modules have passed the Lean kernel checker with the pinned dependencies below. The scope includes algebraic identities, a historical-curvature inequality, and conditional replay and query-accounting results. It does **not** yet include a formal proof of the complete UAG algorithm or of the four final complexity bounds.

The mathematical types and hypotheses in the source files specify precisely what is proved. In particular, the replay and accounting results have explicit interfaces that the manuscript's concrete algorithm must satisfy. The remaining obligations are listed below, so that reviewers can distinguish checked calculations from analysis and algorithm correspondence that still require examination.

## Reproduce the checks

Install [elan](https://github.com/leanprover/elan) and Git, then run from this directory:

```console
lake exe cache get Mathlib.Data.Real.Sqrt Mathlib.Algebra.BigOperators.Ring.Finset Mathlib.Tactic.FieldSimp Mathlib.Tactic.Linarith Mathlib.Tactic.Positivity Mathlib.Tactic.Ring Mathlib.Analysis.SpecialFunctions.Pow.Real Mathlib.Analysis.SpecialFunctions.Log.Basic Mathlib.Tactic.NormNum
lake build UAG
lake env lean verification/AxiomAudit.lean
```

The first command downloads the pinned dependencies and their relevant compiled cache; it requires network access. Keep the committed `lake-manifest.json`. No dependency update is required. Once the toolchain and dependencies are available, `bash verify.sh` or `./verify.ps1` repeats the build and audit.

- Lean: `4.35.0`, selected by `lean-toolchain`.
- mathlib: commit `56a005855ea17e0b0958f73ec801ae140b0a1679`.
- Other dependency revisions: `lake-manifest.json`.
- Original validation date: 27 October 2026.

The recorded build emits Lean module-system migration warnings for legacy imports of mathlib. These are not proof errors. A successful check must include both the build and the final `PASS` line from the axiom audit.

## What has been checked

The source has 110 explicit theorem declarations. The audit also checks generated theorem constants, giving 320 theorem constants in the recorded run. These counts describe the project; they do not measure the fraction of the manuscript verified.

| Module | Checked result | Main entry point | Manuscript correspondence |
| --- | --- | --- | --- |
| [Quadratic.lean](UAG/Quadratic.lean) | Positive weight root and extrapolation coefficients; initial quadratic representation and minimum; gap identity; displacement update; invariants for any finite accepted prefix | `UAG.Quadratic.accepted_history_invariants` | Eq. (2.3); Lemma 3.1; Eqs. (3.5)-(3.18). Labels `eq:nag-parameters`, `lem:inner-model-identity`, `eq:inner-momentum-displacement` |
| [History.lean](UAG/History.lean) | Normalization of the original remainder expression; cancellation of cubic polynomial terms; historical symmetric-descent inequality, with constant 4 and both radius orderings | `UAG.History.symmetric_descent_from_manuscript` | Eq. (3.46), the polynomial cancellation used in Lemma 3.9, and the conclusion of Lemma 3.10 / Eq. (3.47). Labels `eq:outer-remainder`, `lem:remainder-smoothness`, `lem:outer-history` |
| [Replay.lean](UAG/Replay.lean) | Reexecution of the same deterministic step along a valid finite prefix finds the first existing witness within the forward cost; auxiliary continuation erasure; fuel and budget lemmas | `UAG.Replay.replay_finds_within_inner_budget` | Algorithm 4 and the replay argument before Proposition 3.5. Label `alg:complete-replay` |
| [Accounting.lean](UAG/Accounting.lean) | Bounds for finite branch traces, including failed trials and terminal partial calls; the doubling/halving ledger; conditional full FG accounting | `UAG.Accounting.full_query_bound_of_replay_cost` | Eqs. (3.37)-(3.42), Proposition 3.5, and Sec. 3.2. Labels `eq:inner-one-call-scale-ledger`, `prop:inner-global-cost`, `eq:all-fg` |
| [ScaleBridge.lean](UAG/ScaleBridge.lean) | Exact relation between the real persistent scale and the doubling/halving exponent; conversion to the manuscript's real logarithmic FG bound | `UAG.ScaleBridge.real_full_query_bound_of_replay_cost` | Eqs. (3.39)-(3.42), conditional on the scale upper bound. Labels `eq:inner-startup-log`, `eq:inner-global-fg`, `eq:all-fg` |

The numbers refer to the manuscript snapshot identified in [the verification manifest](verification/manifest.json). LaTeX labels provide stable locators if subsequent editing changes numbering.

### Important qualifications

- **Quadratic model.** The finite-dimensional coordinate calculations do not establish that the algorithm's supplied gradient is the derivative of the regularized objective. The concrete state updates, successful tests, and entry conditions must still be matched to the formal definitions. No nonnegativity assumption on the model gap is introduced merely to obtain its invariant.
- **Historical curvature.** The formal proof assumes the stated linewise remainder bounds in both orientations. It uses a finite-scale argument to prove the same symmetric-descent conclusion, with the manuscript's constant 4. It does not verify each sentence of the manuscript's auxiliary-limit argument involving `chi`, or derive the remainder bound from Holder-continuous derivatives.
- **Replay.** The same deterministic `step` and a valid forward prefix are explicit premises. The proof assumes a witness exists in that prefix. Best-point resets, cache behavior, omitted secant calls, and global stopping rules must be shown to preserve the relevant concrete prefix. Ghost history lists occur only in the specification and proof; the formal `replay` function does not read them.
- **Accounting.** The branch trace constructors express allowed local query costs. Every concrete execution branch must still be mapped to these constructors. The full cost theorem accepts a replay-cost upper bound; it does not silently identify two independently defined execution models.
- **Scale bridge.** The bound on the final persistent scale, `ell <= 2L`, is an analytical input. The module proves its consequences, not the line-search theorem that supplies it.

## Remaining mathematical and correspondence obligations

These are coverage boundaries, not claims that the manuscript is incorrect.

1. Match Algorithms 1-4 to the formal model, replay, and accounting states, including all rejection and early-exit branches, caches, preserved state, and query counters.
2. Prove initialization and calibration terminate, establish the initial scale bounds, and derive the warm-start budget from entry clipping, including zero-weight and zero-displacement cases.
3. Prove that a failed residual-monitor test followed by a successful gradient descent trial supplies an actually queried historical witness (Lemma 3.2), and that a sufficiently small residual causes an actual accepting branch (Lemma 3.3).
4. Prove finite backtracking, the persistent scale bound, and inner termination with the stated weight-growth estimate and integer iteration bound (Theorem 3.4).
5. Derive the linewise remainder inequality from Holder regularity of the Hessian or third derivative, with the stated constants (Lemma 3.9). Check the auxiliary-limit proof of Lemma 3.10 if that proof remains in the manuscript.
6. Establish localization and admissibility of the historical pairs, then the directional decrease threshold and its constants (Lemmas 3.8 and 3.11).
7. Establish the analytical calibration and secant-quota bounds used in the complete cost estimate, including degenerate secant cases.
8. Prove the convex, strongly convex, nonconvex, and higher-regularity complexity results and all chain-charging arguments (Theorem 3.12). The two convex results concern first attainment of function error with gradient stopping disabled; the nonconvex results concern gradient stopping.
9. Connect the executable storage to the claimed real-cell `O(d)` working memory. Verification of the Python floating-point implementation, bit complexity, and numerical safeguards is outside this project.

## Evidence and trust boundary

- [build.log](verification/build.log): successful build of all modules.
- [axioms.log](verification/axioms.log): transitive axiom dependencies of every theorem in the `UAG` namespace.
- [AxiomAudit.lean](verification/AxiomAudit.lean): reproducible audit; it rejects any local axiom or transitive axiom other than `propext`, `Quot.sound`, and `Classical.choice`.
- [manifest.json](verification/manifest.json): theorem list, source and log hashes, toolchain revisions, and the exact manuscript source snapshot hash used for correspondence.

The proof sources contain no unfinished proof placeholders or added research axioms. This is a classical Lean proof development, not an axiom-free foundation. Logs record one successful run; rerunning the checker verifies the committed formal statements. The separate task of showing that those statements faithfully capture the relevant manuscript arguments remains explicit above.

The original formalization code is covered by the repository's [MIT license](../LICENSE). Lean and mathlib are obtained separately and retain their own licenses. The repository does not bundle the compiler, dependency source trees, or compiled caches.
