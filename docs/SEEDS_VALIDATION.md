# SEEDS-only external catalog comparison (initial offline version)

This is a **post-hoc evaluation**, separate from C0 detection, E1 tracking and
segmentation. It consumes an already completed run's independently generated
`tables/cme_frame_measurements.csv`.

Run from the repository root:

```bash
python -m scripts.validate_seeds \
  --run-dir runs/clean_cor2a_verify
```

Outputs under `runs/clean_cor2a_verify/validation/seeds/`:

- `event_pair_scores.csv`: each possible SEEDS-to-detection event pair,
  mean per-frame circular angular IoU and radial error; matched assignment flag.
- `event_status.csv`: matched, unmatched SEEDS, unmatched detection, or
  detection outside evaluation timestamps.
- `summary.json`: event-level conditional match rates and configuration.

The default reference is the historical 2010-12-12
`validation/frozen/20101212/experiments/04_d7_external_catalog_validation/results/04_event_scores/seeds_frame_scores.csv`.
Only the **SEEDS source measurements** (`parent_event_id`, `seeds_timestamp`,
`seeds_bpa`, `seeds_width`, `seeds_lead`, `is_evaluable`) are read.
The older candidate association and matching score columns are **ignored**.

For another date use `--seeds-csv /path/to/independently-prepared-SEEDS.csv`
with the same source columns. This first version does not download/parse
the upstream SEEDS catalog automatically; source CSV provenance must be
documented separately.

Default matching uses the nearest observation within 900 s, at least one
angularly overlapping frame, mean IoU >= 0.1, and a globally one-to-one
maximum-weight event assignment. These are *initial operational thresholds*
that should be sensitivity-tested before reporting scientific conclusions.

**Interpretation:** SEEDS may omit events and the historical reference marks
some observation gaps as non-evaluable. Report conditional precision, recall
and F1 as *SEEDS agreement*, **not** independently verified CME detection
accuracy. Detections outside SEEDS-evaluable timestamps are reported separately
and excluded from conditional precision. Width-based angular IoU is not
pixel-level segmentation IoU. Speed is not evaluated in this first version.

Run tests:
```bash
python -m pytest -q tests/test_seeds_validation.py
```
