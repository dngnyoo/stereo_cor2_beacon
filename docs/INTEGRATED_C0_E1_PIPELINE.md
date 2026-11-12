# Integrated C0 → E1 → CME Pipeline (development)

The `feature/data-preparation` branch includes the new **`run-data`** command.
It starts from an existing C0 detection, a prepared North-up RD directory, or
raw local/downloaded COR2 d7 FITS. The E1 component association is computed
independently; this path does **not** read the historical E1 tracking table as
an input. The original `python -m cme_pipeline run --config ...` remains
unchanged as the frozen baseline runner.

## Verified 2010 benchmark replay (from existing C0)

For the previously built frozen 60-pair C0 detection fixture:

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 \
  --run-name integrated_frozen_c0_e1 \
  --detection-dir runs/dual_archive_20101212/frozen_replay/detection_c0_gaussian \
  --frozen-regression
```

This opt-in mode uses historical E1 geometry and checks recomputed state keys
against the Frozen E1 reference. The resulting output is isolated under
`runs/integrated_frozen_c0_e1/`, with independent recomputed track files in
`runs/integrated_frozen_c0_e1_e1_recomputed/`. It also writes
`integrated_run_summary.json`. No existing baseline run is overwritten.

**Note:** The previous standalone replay was numerically tested to match
447 patch components, 219 tracks, 554 states, 185 candidate families,
15 admitted families, 37 tracks, 5,174 recovered patches, and a bitwise equal
50,665-pixel segmentation. The new combined command needs its own execution
test before that result is attributed to it.

## Existing 63-pair RD input (general production path)

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 \
  --run-name integrated_full_63   --preprocessed-dir runs/dual_archive_20101212/preprocessed
```

This **does not trim the input to the historical 60 pairs**. C0 uses all
available RD history; E1 tracks valid evaluation-date observations only.
The 63-pair processing is not expected to match the Frozen 60-pair benchmark.

## Run from raw FITS

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20240512 \
  --run-name cor2_20240512   --source local \
  --input-dir /path/to/cor2_d7_fits \
  --warmup-days 2 \
  --phase-sec 1434
```

Use `--source download` instead of `--source local --input-dir ...` to
download dual-archive d7. Both modes select cadence, perform No-MSB/North-up
preprocessing, run causal C0, independently recompute E1 and execute recovery,
association, gates, segmentation and CME products. An existing preprocessed
directory can instead be supplied with `--preprocessed-dir`.

For dates other than 20101212, the historical external catalog validation
stage is skipped because those fixed reference records are not appropriate
for other days.

## Scientific scope and limitations

- The new combined command is **not yet locally end-to-end tested**.
- The general-date E1 geometry uses the WCS solar center; solar-radius and
  occulter scaling retain the historical pixel constants, which require
  recalibration for fully generalized quantitative interpretation.
- The legacy E1 algorithm tracks valid evaluation frames; missing/unavailable
  observations contribute elapsed time through actual timestamps but are not
  individually represented as track observations.
- New-date external catalog measurements and physical CME identity require
  separate validation. Candidate event catalogs are not authoritative CME
  detections.
- C0 is the sole accepted method for this integrated E1 path. Existing C3
  tracking remains in `track-data` with A3/A4/A5b.
- Outputs are created in new run folders, never in place of the Frozen
  benchmark. Choose a new `--run-name` for each attempt.
- Avoid assuming numerical identity simply from aggregate counts. For a Frozen
  replay, additionally inspect the E1 comparison and final per-pixel mask.

Run contract tests with:

```bash
python -m pytest -q tests/test_run_data_cli.py
```
