# Calibration and frozen validation assets

## Repository layout

- `Calibration/`: COR2 instrument masks and vignette correction resources.
- `Calibration/backgrounds/monthly_min/<spacecraft>/<YYYYMM>/`: historical monthly-minimum COR2 background frames for the legacy CMAD / polarimetric preprocessing utilities.
- `validation/frozen/20101212/`: historical frozen Dec 12, 2010 reference data, tracking fixtures, and external-catalog comparisons.
- `data/raw/cor2/`: date-based raw FITS download cache (ignored locally).
- `runs/`: experiment-specific outputs (ignored locally).

Frozen fixtures are **reference inputs**, not current end-to-end results and not a scientifically certified general-date detector benchmark. Keep the 2010 historical results separate from official COR2-A E2E runs.

## Background FITS limitations

**Important:** The monthly-minimum background FITS objects currently committed to GitHub are zero-byte placeholders. They are **not usable background calibration data**. Before running the legacy CMAD/polarimetric background-subtraction path, populate these paths with verified, nonempty FITS from an authoritative source and check the observation date and polarization.

The `preprocessing/common_utils.py` path provider now resolves calibration and background resources relative to the repository root, rather than `DATA/Backgrounds`. The legacy `Create_COR2_L2_P_CMAD.py` still includes example hard-coded raw/output paths and depends on an external L1 import; it has **not** been certified as an executable end-to-end workflow.

## Moving existing checkouts

After updating to this branch, `git` relocates tracked frozen fixtures and background placeholders. Any **untracked** files previously placed under `validation_20101212/` or `Backgrounds/` must be reviewed and moved separately. Do not overwrite valid local background FITS with GitHub placeholder files.

For local validation, run targeted regression tests and the frozen replay in the environment configured for COR2; do not interpret a successful path test as validation of physical calibration accuracy.
