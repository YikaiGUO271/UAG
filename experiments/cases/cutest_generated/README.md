# Fixed generated S2MPJ evaluators

These 55 Python modules and 55 numeric parameter archives are derived from
Gratton and Toint, S2MPJ, commit fea6a70048eaad28b13a08703ddbfdbf65cd9c30.
Copyright (c) 2026, S. Gratton and Ph. L. Toint. All rights reserved.

The BSD-3-Clause terms and disclaimer are reproduced in
`../../third_party/S2MPJ_LICENSE.txt` and govern these derived files.
The project MIT licence does not replace those terms.

The generated code lowers the original analytic element and group callbacks
into Numba FG evaluators. Fixed arrays retain the original instance parameters.
See `../../manifests/cases.json` and `../../manifests/cutest_generated_validation.json`
for constructor arguments, original source hashes, generated hashes and checks.
The sources are available at https://github.com/GrattonToint/S2MPJ/tree/fea6a70048eaad28b13a08703ddbfdbf65cd9c30/python_problems .
