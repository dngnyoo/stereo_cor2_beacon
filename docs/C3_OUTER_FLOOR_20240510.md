# 2024-05-10 C3 outer-background floor diagnostic

This experiment targets **2024-05-10**, not the original 2010-12-12 benchmark.
First compare detectors on identical North-up RD before running tracking or
producing CME catalogs. There is no verified 2024 `geometry.rsun_px`
configured, so do **not** reuse the historical 2010 value of 8.163 for
2024 CME heights or speeds. No 2024 SEEDS reference is assumed.

```bash
python -m cme_pipeline prepare-data --date 20240510 --source download --product d7 --warmup-days 1 --preprocess --output-dir runs/outer_floor_20240510_input

python -m cme_pipeline detect-data --preprocessed-dir runs/outer_floor_20240510_input/preprocessed --method c3 --date 20240510 --floor-mode frame_median --output-dir runs/outer_floor_20240510_frame_median

python -m cme_pipeline detect-data --preprocessed-dir runs/outer_floor_20240510_input/preprocessed --method c3 --date 20240510 --floor-mode outer_background --outer-floor-inner-px 88 --outer-floor-outer-px 116 --output-dir runs/outer_floor_20240510_annulus
```

Compare `frame_detection_summary.csv` in both detector outputs:
`status`, `floor_p04`, `floor_sample_count_p04`,
`effective_threshold_p04`, and `n_active_p04`.
The ring is an **unvalidated hypothesis**, not a scientifically optimized
background region. A high foreground occupancy or instrumental background
variation can contaminate it. The detector uses the upper-tail-trimmed median
of P04 absolute Z-score P95 samples from the ring, then applies
`max(T24_p04, floor_p04)`; it does not directly subtract a background image.
The result may equal the baseline if T24 dominates.

A 2024 full `run-data` CME catalog requires independently verifying the
2024 WCS-based `geometry.rsun_px`. Only then should downstream tracking,
catalog metrics and external references be compared.

## Detection diagnostic PNGs

Generate an identical set of evaluation-day frames from both outputs:

```bash
python -m cme_pipeline visualize-detection --detection-dir runs/outer_floor_20240510_frame_median --date 20240510 --all-frames
python -m cme_pipeline visualize-detection --detection-dir runs/outer_floor_20240510_annulus --date 20240510 --all-frames
```

Images are saved separately to `<detection-dir>/visualizations/detection/`.
Each includes: original north-up RD (DN/s), normalized P04 absolute-Z P95
score field, and final active P04 masks over RD. The outer-floor branch
draws the sampled radial bounds in cyan. Titles show timestamp, detector
status, T24, floor, effective threshold, and whether floor binds.
Use `--max-frames 6` (default) for a small sample instead of
`--all-frames`. The command only reads persisted outputs; it never
recomputes or modifies detection arrays.


## Robust outer floor sweep (k=1, 2, 3)

Use existing **2024-05-10** preprocessed input at
`runs/may20240510_c3_test/preprocessed`. Never overwrite the original
`runs/may20240510_c3_test/detection_c3_gaussian` baseline.

`outer_background_mad` computes `median + k * 1.4826 * MAD` on the lower
80% of valid P04 activity scores in the same 88–116 px annulus, then uses
`max(T24_p04, outer_floor)`. The exclusion rule is fixed for this sweep.

```bash
for k in 1 2 3; do
  python -m cme_pipeline detect-data \
    --preprocessed-dir runs/may20240510_c3_test/preprocessed \
    --method c3 --date 20240510 \
    --floor-mode outer_background_mad --outer-floor-k "$k" \
    --outer-floor-inner-px 88 --outer-floor-outer-px 116 \
    --output-dir "runs/20240510_c3_outer_mad_k${k}" \
    --visualize --visualize-all-frames
done
```

Compare per-frame `status`, `T24_p04`, `floor_p04`,
`effective_threshold_p04`, `n_active_p04`, and full P04 patch masks.
Ensure baselines use the **same RD input and VALID frame set** before
interpreting active-patch differences. Larger k increases the threshold
for a fixed frame, but actual CME retention is not guaranteed.


## A5b tracking-only comparison on 2024-05-10

Both runs reuse existing detection outputs (no download, preprocessing or detection
rerun). The `track-data` command persists ownership masks to
`runs/<run_name>/arrays/track_ownership_masks.npy`; this does not require
`geometry.rsun_px` or run downstream CME catalog stages.

```bash
python -m cme_pipeline track-data \
  --detection-dir runs/may20240510_c3_test/detection_c3_gaussian \
  --date 20240510 --method a5b --run-name may10_a5b_baseline
python -m cme_pipeline track-data \
  --detection-dir runs/20240510_c3_outer_p95 \
  --date 20240510 --method a5b --run-name may10_a5b_outer_p95
python -m cme_pipeline compare-tracking \
  --baseline-run runs/may10_a5b_baseline \
  --experiment-run runs/may10_a5b_outer_p95 \
  --date 20240510
```

Output: `runs/may10_a5b_outer_p95/comparison_tracking_full_day/frames/`
plus `comparison_summary.csv`. All frames ending on May 10 are rendered,
including unavailable-status frames. Red outlines denote detected active P04
patches; colored small squares indicate patch owner (track ID) **within each
run**, not corresponding track identity across runs. Tracking uses default A5b
parameters and is exploratory until track continuity is reviewed.


### Enhanced tracking visualization (existing tracking runs reused)

`compare-tracking` now writes to:
`runs/may10_a5b_outer_p95/visualizations/comparison_tracking_full_day/`.

For each May 10 evaluation-date RD pair, the left/right panels draw distinct
contours of 8-connected *track-owned* P04 regions. Labels have
`T<track-ID>/C<raw-component-ID>.<owned-subregion-index>`,
the leading owned-patch's WCS PA, the connected region's circular-mean CPA,
PA span, and its outermost radius in pixels. Yellow stars denote the
outermost **observed owned patch** of each region. To avoid covering image
features, the labels no longer use dark callout boxes: solid rays show
front PA, dashed rays show observed component mean CPA, and outer arcs
show each component's front at its minimal enclosing PA interval. Short
labels sit next to those geometric elements; very small regions omit
the angular labels. Stored track-level `tracked_cpa`, `front_radius`,
and `leading_front_state` remain available in
`tables/live_tracking_states.csv` rather than an overlaid box.
Track-level fronts may be predicted rather than observed.
`component_geometry.csv` records these measured owned-region geometries
separately from original A5b component tables. Ownership subregions can
split an original C3 component during contested-track allocation.
Re-run only `compare-tracking`; no need to rerun A5b for the changed
visualization. Existing earlier comparison folders under the run root
remain untouched.
