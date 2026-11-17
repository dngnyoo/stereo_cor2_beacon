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

## Recovery / Identity / Birth Gate / Size Gate (May 10, 2024)

Continue from existing A5b track products; **no new download, preprocessing,
detection, or tracking**. Outputs are separate from the original tracking runs.

**Important physical caveat:** a 2024-specific solar-radius-to-pixels
calibration has not been confirmed. The following `22.0` is an explicit,
**exploratory operational recovery minimum radius in pixels**, not a
verified equivalent of the intended physical 2-Rsun inner boundary.
Use the same specified pixel cutoff in both runs and do not interpret
recovered areas or kinematics as final calibrated measurements.

```bash
python -m cme_pipeline run-gates-data \
  --tracking-run runs/may10_a5b_baseline \
  --run-name may10_gates_baseline_r22 \
  --recovery-inner-radius-px 22 \
  --birth-max-radius-px 30 --min-detected-patches 5

python -m cme_pipeline run-gates-data \
  --tracking-run runs/may10_a5b_outer_p95 \
  --run-name may10_gates_outer_p95_r22 \
  --recovery-inner-radius-px 22 \
  --birth-max-radius-px 30 --min-detected-patches 5

python -m cme_pipeline compare-gates \
  --baseline-run runs/may10_gates_baseline_r22 \
  --experiment-run runs/may10_gates_outer_p95_r22 \
  --date 20240510
```

Stage files include `stages/size_gate/family_summary.csv`, `stages/size_gate/size_gate_decisions.csv`,
`arrays/recovered_patch_masks_pipeline_b_full_day.npy`,
`arrays/family_recovered_patch_masks.npz`, and `gates_summary.json`.
The comparative PNGs and frame CSV are in
`runs/may10_gates_outer_p95_r22/visualizations/comparison_gates_full_day/`.
Gray contour = detected patch, green = admitted tracked patch, red = rejected,
cyan = recovery of admitted families. Predicted-only observations are not
represented as detected patches. A5b's multiple components are preserved under
a single identity Family per Track ID; this experiment does not run Rule B.
No segmentation or CME catalog is executed.

## Canonical visualizations for new single runs

The latest single-run outputs are now standard (default on):
- `detect-data` automatically writes **all evaluation-day** diagnostic PNGs
  to `<detection-run>/visualizations/detection/`. Final active P04 patches
  are **red outlines with no opaque yellow fill**; the RD and patch-score
  diagnostic panels are retained.
- `track-data --method a5b` automatically writes all evaluation-day frames
  to `<tracking-run>/visualizations/tracking/`. Each Track ID receives **one**
  PA Start line, PA End line, A5b tracked CPA dashed line, observed outer-front
  arc and concise labels. Disconnected owned regions remain one Track geometry.
  Geometry values are also saved to `track_geometry.csv`.

Use `--no-visualize` to skip PNG creation for faster batch experiments.
These defaults change only the visualization; C3 floor computation and A5b
tracking remain unchanged. Side-by-side comparisons remain **explicit**
commands, `compare-detection` and `compare-tracking`, because they need
two separate runs. Existing run outputs are never overwritten.

For a previously completed run, render only the missing standard diagnostic
without rerunning any algorithm:

```bash
python -m cme_pipeline visualize-tracking \
  --tracking-run runs/may10_a5b_outer_p95 --date 20240510 \
  --output-dir runs/may10_a5b_outer_p95/visualizations/tracking_standard_v1

python -m cme_pipeline visualize-detection \
  --detection-dir runs/20240510_c3_outer_p95 --date 20240510 \
  --all-frames \
  --output-dir runs/20240510_c3_outer_p95/visualizations/detection_standard_v1
```


## Duration + size gate sensitivity sweep (after A5b)

A5b already applies an outward-front tolerance, but it does **not**
guarantee monotonically positive radial motion. Do not introduce a
second outward-motion rule in this experiment. Preserve the existing
Family Birth Gate at 30 px and compare alternative **Size Gate**
admission criteria:

- Peak owned P04 patch count >= 10 or >= 20 in any observed frame.
- At least 3 or 5 distinct `OBSERVED` pairs across the complete
  processed sequence (including warmup; not necessarily consecutive).
- Exclude `PREDICTED_ONLY` states from the observed-pair count.

The original defaults (peak >=5, minimum observed pairs >=1) remain
unchanged for backwards compatibility. Two independent A5b tracking
runs are reused without rerunning detection or tracking. A quick
**selection-only** sensitivity report can be generated from the
previously saved Size Gate family summaries without recomputing
Recovery:

```bash
python - <<'PY'
import pandas as pd
paths = {
    'baseline': 'runs/may10_gates_baseline_r22/stages/size_gate/family_summary.csv',
    'outer_p95': 'runs/may10_gates_outer_p95_r22/stages/size_gate/family_summary.csv',
}
for name, path in paths.items():
    df = pd.read_csv(path)
    print(name)
    for duration in (3, 5):
        for size in (10, 20):
            passed = df[(df.anchor_count > 0) &
                        (df.n_active_pairs >= duration) &
                        (df.max_detected_patches >= size)]
            print(f'duration={duration}, peak_size={size}: {len(passed)} candidate families')
PY
```

To run full Recovery and create Gate-specific outputs for a chosen
threshold combination, use `run-gates-data` with a **new** run name and
`--min-observed-pairs <3 or 5>` together with
`--min-detected-patches <10 or 20>`. For example:

```bash
python -m cme_pipeline run-gates-data \
  --tracking-run runs/may10_a5b_outer_p95 \
  --run-name may10_gates_outer_p95_d3_s10_r22 \
  --recovery-inner-radius-px 22 \
  --birth-max-radius-px 30 \
  --min-observed-pairs 3 --min-detected-patches 10
```

Only compare gate runs with *matching thresholds* using
`compare-gates`. Distinguish whole-sequence candidate counts from
May-10-only visible tracks in reporting.
