# Data Preparation: Input Contract and Integration Plan

**Development status:** the `feature/data-preparation` branch provides the **image acquisition + FITS metadata inventory** entry point. Raw COR2 FITS preprocessing and the new C3 detector are **NOT yet integrated** with the eight-stage research pipeline. The current `main` benchmark is unchanged.

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

Inventory and validate a user's local FITS images:

```bash
python -m cme_pipeline prepare-data \
    --date 20240510 \
    --source local \
    --product science_raw \
    --input-dir /path/to/cor2_images
```

Download one STEREO-A COR2 product from the NASA archive and inventory it:

```bash
python -m cme_pipeline prepare-data \
    --date 20240510 \
    --source download \
    --product science_raw
```

Product choices: `beacon_raw`, `science_raw`, `beacon_seq`, `science_seq`. You must specify which observational product is intended. A valid UTC `DATE-OBS` is required in each input FITS; files with a different calendar date are rejected rather than silently mixed.

Outputs: `runs/data_prep_<date>_<product>/input_manifest.json` and, for `download`, downloaded FITS under that run directory's `raw/<product>/<date>/`.

`input_manifest.json` reports `status = INVENTORIED_NOT_PREPROCESSED`, intentionally. **Do not claim that these images are already processed or that a new-date CME catalog has been generated.**

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
