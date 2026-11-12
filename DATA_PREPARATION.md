# Data Preparation: Input Contract and Integration Plan

**Development status:** the `feature/data-preparation` branch implements **d7 acquisition, strict cadence selection, No-MSB North-up RD and an initial C0/C3 detector replay**. A numerical reference comparison and availability-aware Tracking handoff are still required; the `main` Frozen Benchmark is unchanged.

## Target workflow

```text
Date (YYYYMMDD)                         Researcher-supplied COR2 FITS
       |                                           |
  NASA STEREO archive                          Local input folder
       |                                           |
       +------------------+------------------------+
                          |
                Data acquisition & inventory
                          |
           FITS calibration, timestamp ordering,
             image QC and WCS normalization
                          |
           North-up running-difference (RD) +
           valid-pixel masks + observation manifest
                          |
         Per-scale patch P95 statistics + causal T24
                          |
       C3 QC-first 16 -> 8 -> 4 hierarchical detection
                          |
           DetectionOutput shared interface
                          |
     A3/A4/A5b Tracking -> Recovery -> Family Association
              -> Gates -> Segmentation
              -> CME Products -> External Validation*
```

*External validation currently expects the frozen 2010-12-12 reference catalogs and will require a separate **date-aware optional mode** for other dates.

## Current commands (implemented)

Select **d7 only** from a local COR2 FITS directory, validate `DATE-OBS` against
the timestamp in the FITS filename, and record strict 30-minute pairs and gaps:

```bash
python -m cme_pipeline prepare-data --date 20240510 \
  --source local --input-dir /path/to/d7_images --preprocess
```

Download only `_d7c2` image files from the NASA STEREO-A INS_DATA
`L0/a/img/cor2/<YYYYMMDD>/` archive, then run the same selection:

```bash
python -m cme_pipeline prepare-data --date 20240510 --source download --preprocess
```

The default `--product d7` is now the intended data stream; an explicit
`--phase-sec 1434` reproduces the historic May 2024 phase (23m54s).
This default phase is **not validated as universal**. Changing the date alone
does not guarantee a usable cadence, and a mismatch intentionally raises an error.

Output is `runs/data_prep_<date>_d7/input_manifest.json`, including all
qualified d7 FITS files, selected observations, strict 30-minute pairs and
nonbridged gaps. Downloads reside in `raw/d7/<date>/` within the same run.
The manifest is marked `INVENTORIED_NOT_PREPROCESSED`.

**Not implemented yet:** dispatch to availability-aware Tracking. C0/C3 and RD preprocessing code is implemented but requires a real-data numerical regression run. Do not
treat an input manifest as a detector result.

## No-MSB / North-up outputs (`--preprocess`)

The new science arrays appear under
`runs/data_prep_<date>_d7/preprocessed/`:

```text
north_up_RD_DNs_per_second.npy     # float32 [N,256,256] DN/s
north_up_pair_valid_masks.npy     # bool [N,256,256]
target_wcs_header.txt             # fixed solar-centered North-up WCS
01_rd_manifest.csv                # pair UTC start/end, source paths, dt
preprocessing_metadata.json       # units, method provenance, status
```

`Calibration/cor2_mask.fts` must have a valid WCS and must reproject
to a usable mask on the selected grid. Override with `--mask-path` when
necessary; do not bypass occulter masking silently.

The calibration path follows legacy `scc_img_trim`, EXPTIME correction,
SEBIP and bias removal; **get_calfac(MSB) is never called**.
Registered North-up imaging is performed before pair subtraction.
Only exactly matched 1800 +/- 2 s pairs enter the RD stack; missing
30-minute intervals are recorded but never bridged.

**Scientific verification still required:** legacy FITS trim orientation,
solar WCS plate-scale fidelity, occulter mask footprint, and a framewise
comparison to previously generated No-MSB RD. The output is explicitly
named `DNs_per_second` to avoid misrepresenting it as frozen 2010 raw DN.

**24-hour warmup warning:** Selecting a single date alone does not
supply a full preceding 24-hour score history. A date-window or
previous-day warmup fetch must be added before C3 can provide routine
valid detections throughout the requested day.

## C0 / C3 detector runs (newly implemented, unverified on NASA FITS)

Run **C3 (default target detector)** on a requested day, with the preceding
two UTC days fetched for causal T24 warmup:

```bash
python -m cme_pipeline prepare-data --date 20240510 \
  --source download --preprocess --detect c3
```

To compare C0 **on exactly the same RD arrays** without downloading or
preprocessing a second time:

```bash
python -m cme_pipeline detect-data \
  --preprocessed-dir runs/data_prep_20240510_d7/preprocessed \
  --method c0 --date 20240510
```

A separate C3 replay on the same RD is also available via
`detect-data --method c3`, provided that detector's output directory
does not already exist (writes are intentionally overwrite-protected).
For local FITS, use `--source local --input-dir /path/to/three_days_of_d7`
on the initial preparation command.

The `--detect` flag defaults to **two preceding UTC dates** (configurable
with `--warmup-days`). A local directory must include those earlier dates
itself; files are selected from the requested window using FITS `DATE-OBS`.
With no warmup, frames are explicitly marked `WARMUP` and not silently
represented as usable detections. The 2010 Frozen Baseline is unchanged.

New detector artifacts:

```text
runs/data_prep_<date>_d7/detection_c3_gaussian/
  patch_state_p04.npy           # [N,64,64] signed int8: -1, 0, 1
  patch_component_labels.npy    # [N,64,64]; -1 on unavailable frames
  frame_detection_summary.csv   # UTC, T24, floor, activation QC, status
  detection_metadata.json       # detector, warmup and provenance
```

`--detect c0` writes `detection_c0_gaussian/` instead. Alternative
`--denoiser median` is supported. Detector runs cover warmup and analysis
observations; `in_requested_date` in the summary identifies requested
day rows. The C0 output uses an explicit unavailable sentinel during warmup
for interface safety, unlike legacy archival C0 saved-empty convention.

**Scientific qualification:** C0 and C3 are **algorithmic reimplementations**
from research source, not yet numerically regression-matched to the frozen
C0/C3 reference masks. Preprocessing and detection have not yet been tested
end-to-end on real NASA FITS. Tracking is not wired to the new arrays; its
active-mask contract currently cannot represent C3 `-1` observations
without an availability-aware adapter. Do not treat these results as a
validated finished CME catalog.

## Intended detector variants

- `c0` — baseline Dynamic-T detector from
  `stereo_cor2_dev/segmentation_pipeline/05_run_dynamicT_detection.py`
  (historical neighboring-parent refinement).
- `c3` — intended **default** for future science runs, from
  `one_month_preprocessing_detection_survey/09_frame_median_floor_qc/`
  (pure hierarchy, pre-floor global-activation QC, p04-only floor, -1 unavailable).
- Frozen 2010 benchmark — unchanged in `main`; loading frozen masks is not
  equivalent to replaying either algorithm on new FITS.

## Canonical acquisition and preparation sources

1. `stereo_cor2_dev/download_1M.py` identifies STEREO-A COR2 `_d7c2`
   files from INS_DATA. This filtering is reused in the current entry point.
2. `stereo_cor2_dev/segmentation_pipeline/01_select_d7_30min.py` checks
   header/filename timestamps and the 1800 ± 2 second cadence. This
   algorithm is adapted to arbitrary requested dates with configurable phase.
3. `segmentation_pipeline/16_no_MSB_preprocessing.py`: trim, EXPTIME,
   SEBIP, bias correction. Preserve **DN/s**, do not multiply MSB factor.
4. `handover_archive/preprocessing/reproject.py`: fixed North-up WCS and
   footprint. Reprojection order and calibration provenance require a
   science regression test before use.
5. `segmentation_pipeline/17_no_MSB_RD.py`: strictly paired RD, no gap bridge.
6. C0 from stage 05 and C3 from its isolated experiment; implement with
   shared RD/P95/T24 inputs so scientifically meaningful comparisons are possible.

## Existing code surveyed

| Existing component | What it actually does | Gap before new-date detection |
| --- | --- | --- |
| `preprocessing/download.py` | Downloads separate Beacon/science image and sequence archive products for a hard-coded date / interactive prompt | Reusable API, strict date handling, explicit product selection and provenance |
| `preprocessing/1_beacon_gcs.py` | Raw Beacon image Z-score RD | Not scientifically equivalent to frozen 2010 north-up RD DNs |
| `preprocessing/2_beacon_cmad.py` | Sequence L1 / CMAD pB synthesis + Z-score RD | Depends on background/calibration availability; groups every three sorted FITS without independent polarization/time checks |
| `preprocessing/3_science_hv.py` | Science raw trim + exposure-normalized simple running difference | Not the frozen calibration/geometry process |
| `preprocessing/4_science_cmad.py` | Science sequence pB / CMAD + Z-score RD | Same calibration, polarization grouping and alignment concerns |
| `cme_pipeline/stages/detection/frozen_detection.py` | Loads 2010-12-12 north-up `RD_DNs.npy`, valid masks, thresholds, and precomputed patch grids/labels | Hardcoded 2010 paths; does **not** generate detector inputs from raw data |
| `feature/c3-detection-stage` | Pure per-frame QC-first 16/8/4 gating and 8-connected components | No FITS loading, multiscale P95 scoring, causal T24 history, or dataset adapter |
| `cme_pipeline/stages/tracking/v01_baseline_tracking.py` | E1 tracking uses **frozen precomputed track-state CSV** | Cannot be used to track previously unseen observations |
| A3/A4/A5b tracker stages | Implement actual detection-component tracking from `DetectionOutput` when supplied | Need an upstream detector stage to supply a date-specific `DetectionOutput` |

## Next integration steps

1. Choose a documented **canonical science-image stream** (e.g., L0 science raw vs polarization sequences) and reproduce an exact North-up RD DN preprocessing path with WCS, cadence gaps, valid masks and instrument calibration. Avoid reusing display-only Z-score images as detector inputs.
2. Build multiscale patch scores and strictly causal trailing T24 thresholds; call the already-isolated C3 kernel, retaining explicit `WARMUP`, `NO_HISTORY`, `MISSING_DATA` and `GLOBAL_ACTIVATION` states rather than encoding them as empty detections.
3. Map complete resulting arrays, manifests and WCS to `DetectionOutput`. Verify shapes: `(N, 256, 256)` RD/masks, `(N, 64, 64)` 4-pixel grid and components.
4. Add source-mode dispatch to the pipeline while leaving `frozen` the default; only allow `a3`, `a4`, `a5b` on new data until E1 is generalized. Do not fall back to 2010 files for a non-2010 date.
5. Make external validation optional when date-specific catalogs have not been provided. Run a controlled end-to-end new-date experiment, check provenance, and document exact data requirements.

No numbers from the frozen 2010 benchmark are claimed for new dates. Installation, the local-input tests and the download HTTP path still require runtime verification.
