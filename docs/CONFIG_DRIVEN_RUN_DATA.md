# Config-driven run-data: C3 / A5b first live integration

The `run-data` CLI reads `detection.method`, `tracking.method`, and
`association.method` from YAML. The legacy C0/E1 recomputation route is
preserved separately for numerical regression until its live tracking adapter
is migrated; A3/A4/A5b use the registered stages directly.

## 2010-12-12 C3 / A5b / Identity / SEEDS

The experiment configuration is
`configs/dec20101212_c3_a5b.yaml`. It selects C3 Gaussian,
A5b, Identity, Signed PA Recovery, gates, pixel segmentation and CME
catalog/visualization. It uses the previously verified historical 2010
solar radius conversion (8.163 pixels/R_sun); independently verify the
chosen data's WCS/solar radius before interpreting heights or speeds.
Other observing epochs MUST supply their own verified `geometry.rsun_px`.

The official per-frame SEEDS observations are prepared by the existing
acquisition and parsing scripts:

```bash
python -m scripts.fetch_seeds_monthly --date 20101212
python -m scripts.fetch_seeds_details --date 20101212
```

Skip these commands when
`data/catalogs/seeds/20101212/seeds_frames.csv` already exists.

To run from **existing north-up RD files** (C3 detection computed within
this run), replace the directory with an actually available 2010-12-12
preprocessing output:

```bash
python -m cme_pipeline run-data \
  --config configs/dec20101212_c3_a5b.yaml \
  --date 20101212 \
  --run-name dec20101212_c3_a5b_full \
  --preprocessed-dir /PATH/TO/20101212/preprocessed
```

Or reuse completed C3 detection without regenerating it:

```bash
python -m cme_pipeline run-data \
  --config configs/dec20101212_c3_a5b.yaml \
  --date 20101212 \
  --run-name dec20101212_c3_a5b_full \
  --detection-dir /PATH/TO/20101212/detection_c3_gaussian
```

For new FITS inputs, use `--source local --input-dir /PATH/TO/d7`
instead. A date is not sufficient unless the required previous days
are available for C3's 24-hour threshold warmup.

Outputs are written under `runs/<run-name>/`:

- `tables/live_tracking_states.csv`
- `arrays/track_ownership_masks.npy`
- `arrays/final_pixel_segmentation_union.npy`
- `tables/cme_event_catalog.csv`
- `tables/cme_frame_measurements.csv`
- `visualizations/` (when enabled)
- `validation/seeds/event_pair_scores.csv`
- `validation/seeds/event_status.csv`
- `validation/seeds/summary.json`
- `integrated_run_summary.json`

Validation calls the existing official SEEDS parser-compatible evaluator
(`scripts/validate_seeds.py`) and uses UTC timestamps, not the frozen
32-frame index alignment. Report SEEDS event agreement, NOT ground-truth
CME detection accuracy. If the SEEDS frame reference is unavailable,
remove SEEDS from `validation.catalogs` to produce CME products alone.

The old `--frozen-regression` C0/E1 route remains available. Use separate
run names for every experiment and do not overwrite baseline outputs.

## Tests

```bash
python -m pytest -q tests/test_configured_run_data.py \
  tests/test_parse_seeds_details.py tests/test_seeds_validation.py
```

## Deferred tasks

- Add a general independent E1 live adapter so the separate historical
  C0/E1 runner becomes a common registered path.
- Derive and verify solar radius from each FITS/WCS automatically.
- Replace experimental algorithm nicknames with self-descriptive canonical
  names while retaining the old identifiers as aliases.
- Compare new C3/A5b and frozen C0/E1 results on a common, explicitly
  documented SEEDS-evaluable timestamp set.
