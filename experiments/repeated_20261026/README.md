# Repetition campaign, 26 October 2026

Four complete repetitions (1828 observations) and 44 observations from a deadline-stopped fifth repetition. Only the four complete repetitions enter the manuscript aggregate. Every original trace remains byte-exact inside the per-repetition ZIP archives; the partial archive is explicitly labelled.

Run `python experiments/audit_repetitions.py experiments/repeated_20261026` from the repository root, then `python experiments/analyze_repetitions.py experiments/repeated_20261026 --output experiments/generated/repeated-profiles` to redraw the two manuscript profiles and the two repetition-level sensitivity profiles. Details and scientific scope are in `docs/experiments.md`.

`driver_used.py` is the exact executed archival driver, saved for audit, not the portable entry point. To generate new observations, use `experiments/repeat_campaign.py`; on Windows `--clock windows-native` selects the bundled historical QPC clock. Full reruns also require the four separately acquired logistic datasets. The public reference driver records zero FG for the repository CUTEst loader and the separate one-FG initial check; the actual archived historical loader made an additional one-FG setup check, as recorded in this campaign.
