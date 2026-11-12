# Calibration and frozen reference assets

- `Calibration/*.fts`: official instrument calibration FITS and masks.
- `Calibration/backgrounds/monthly_min/a/YYYYMM/`: monthly CMAD background images for legacy preprocessing.
- `validation/frozen/20101212/`: historical frozen regression reference inputs; not current COR2-A general-date outputs.
- `data/raw/cor2/`: shared raw observational FITS.
- `runs/`: generated experiments.

**Important:** Currently tracked monthly background FITS are **zero-byte placeholders**. Restore genuine FITS before using CMAD background correction. Preprocessing filters placeholders and otherwise proceeds *without* background subtraction; do not treat those results as CMAD-corrected.

Legacy preprocessing functions accept the background **base** directory `Calibration/backgrounds`, then select `monthly_min/{a,b}/YYYYMM/` beneath it.
