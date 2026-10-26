# Third-party provenance and distribution

The project-authored solver, benchmark drivers, baseline reimplementations, reporting code, manifests and research measurements are distributed under the repository MIT licence. Algorithm citations and implementation provenance do not imply endorsement by the cited authors. The package does not bundle NumPy, SciPy, Numba, Matplotlib or threadpoolctl; these retain their respective upstream licences when installed as dependencies.

## S2MPJ-derived evaluators

`experiments/cases/cutest_generated` contains 55 generated evaluator modules and their fixed parameter arrays. They are derived from [Gratton and Toint's S2MPJ](https://github.com/GrattonToint/S2MPJ) at commit `fea6a70048eaad28b13a08703ddbfdbf65cd9c30`. The generation lowers analytic element/group callbacks to native FG code; it does not change problem definitions. These files retain the upstream BSD-3-Clause terms, reproduced in `experiments/third_party/S2MPJ_LICENSE.txt`, and are excluded from any claim that all repository contents originate with this project's author. That licence text is the exact file addition recorded in the archived upstream commit metadata. The [licence at the fixed commit](https://github.com/GrattonToint/S2MPJ/blob/fea6a70048eaad28b13a08703ddbfdbf65cd9c30/LICENCE.txt) and per-problem Python sources are the primary provenance. Problem-specific references remain available in those upstream source files. Original and generated SHA256 values and fixed constructor arguments are recorded in the manifests.

## External datasets

The four logistic input datasets are not redistributed. Their [LIBSVM distribution page](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) identifies the source collections and preprocessing. The preparation script downloads only the specified a9a, australian_scale, ionosphere_scale and w8a files when explicitly requested, or reads files supplied by the user. It verifies exact source bytes and resulting numeric arrays. The MIT licence of this repository does not apply to those datasets. The download cache and transformed data are ignored by Git. UCI/Statlog provenance and any applicable dataset terms should be reviewed at the linked source before use or redistribution.

## Comparison algorithms

The repository contains project-authored native benchmark implementations of published recurrences, not vendored copies of upstream solver repositories. Their scientific sources are:

- Nesterov acceleration and the O'Donoghue--Candes gradient-restart rule.
- Li and Lan, AC-FGM, [doi:10.1007/s10107-025-02250-z](https://doi.org/10.1007/s10107-025-02250-z). The [authors' implementation](https://github.com/tli432/AC-FGM-Implementation) was a recurrence/parameter reference; its files are not bundled.
- Carmon et al., [Convex Until Proven Guilty](https://proceedings.mlr.press/v70/carmon17a.html).
- Suh and Ma, [AdaNAG, arXiv:2505.11670v1](https://arxiv.org/abs/2505.11670v1), Algorithm 1 and exact initialization from Lemma 8.
- Jiang and Mokhtari, [Accelerated Quasi-Newton Proximal Extragradient](https://proceedings.neurips.cc/paper_files/paper/2023/hash/19c9708f31ec44b5b1cbd67f91d05d95-Abstract.html), Algorithm 1 and Subroutines 1--4. The CUTEst online-scale extension is separately labelled.

Every reference implementation routes charged function/gradient work through the common native boundary. The portable clock adaptation is identified separately from the preserved solver recurrences. Original benchmark source hashes are retained in `experiments/frozen/protocol.json`; differences in repository paths and timing infrastructure are documented in `docs/experiments.md`.
