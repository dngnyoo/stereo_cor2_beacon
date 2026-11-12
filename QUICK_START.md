# Quick Start — STEREO-A COR2 CME Pipeline

Two supported workflows coexist:

1. **General raw-FITS E2E:** NASA Science + Beacon FITS → strict 30-minute
   RD → No-MSB calibration → North-up reprojection → C0 → independent E1
   tracking → Recovery/Association/Gates → Segmentation → CME candidate
   catalog and visualizations.
2. **Historical Frozen benchmark:** the original fixed 2010-12-12
   detection/tracking reference. It is scientifically distinct from the
   general workflow and uses the legacy mask.

Linux / WSL and Python 3.12 are recommended. Run commands from the repository root.

## 1. Clone and install

```bash
git clone https://github.com/dngnyoo/stereo_cor2_beacon.git
cd stereo_cor2_beacon
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

For development, use a dedicated branch and never overwrite the verified
baseline directories. Check `git status --short` before switching branches.

## 2. Install COR2-A calibration asset

General E2E processing defaults to `Calibration/cor2A_mask.fts`, a genuine
2048×2048 COR2-A static FITS mask. The original empty placeholder in a fresh
clone is **not** usable. Install the verified 17 MB reference:

```bash
cp ~/stereo_cor2/handover/calibration/cor2A_mask.fts Calibration/cor2A_mask.fts
sha256sum Calibration/cor2A_mask.fts
```

Expected SHA-256:

```text
7cde9474fa9d1fe93d9a6ff08464598d5d2250b423b00be449ca24b2ec7ca606
```

The pipeline checks that digest automatically for its default COR2-A mask
and aborts if missing or corrupt. `--mask-path` explicitly overrides it,
which is useful for controlled comparison; those runs must have different names.
The historical Frozen regression may instead require the legacy
`Calibration/cor2_mask.fts`, which is a **different mask**.

## 3. Verify dependencies and offline tests

```bash
python -m compileall -q cme_pipeline scripts
python -m pytest -q \
  tests/test_download_cache.py \
  tests/test_dual_archive.py \
  tests/test_migrate_shared_raw.py \
  tests/test_run_data_cli.py \
  tests/test_cor2_mask_asset.py
```

A successful offline test suite does not substitute for an actual E2E replay.

## 4. Raw FITS storage and acquisition reuse

Every `run-data --source download` run uses the **same** raw archive:

```text
data/raw/cor2/
  science/20101211/*.fts
  science/20101212/*.fts
  beacon/20101211/*.fts
  beacon/20101212/*.fts
```

Further dates accumulate under `YYYYMMDD/`. Each completed archive/date
gets a `.download_complete.json` file containing file sizes and SHA-256.
A validated complete date is reused without an HTTP request; new dates are
downloaded, and changed/missing files are re-fetched. Science d7 is preferred
over Beacon d7 when both archives supply the same filename. Both raw source
copies are retained for provenance.

**Reusing your old 1.2 GB acquisition**: on the original MELONHOUSE checkout,
preview before migrating; never delete or move the legacy source:

```bash
python scripts/migrate_shared_raw.py \
  --from-raw runs/acquisition_20101212_raw_e2e_20101212/raw
python scripts/migrate_shared_raw.py \
  --from-raw runs/acquisition_20101212_raw_e2e_20101212/raw \
  --apply --hardlink
```

Hardlinks preserve disk space on the same filesystem. The old paths remain
valid for previous manifests. On the first download invocation after
migration, the pipeline may still request a NASA directory listing to
establish day-level completion markers; existing nonempty FITS are reused.

## 5. Run E2E from the downloaded archives

For 2010-12-12 using one preceding UTC warmup day:

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 \
  --run-name 20101212_cor2a_e2e_v1 \
  --source download \
  --warmup-days 1
```

This reads the verified official COR2-A mask and creates **one**
`runs/20101212_cor2a_e2e_v1/` containing:

```text
run_request.json
acquisition/input_manifest.json
acquisition/manifests/{raw_inventory.csv,selected_d7_sources.csv}
acquisition/preprocessed/{preprocessing_metadata.json,01_rd_manifest.csv,*.npy}
detection/{detection_metadata.json,frame_detection_summary.csv,*.npy}
tracking/{summary.json,03_patch_e1_track_states.csv,...}
stages/
arrays/final_pixel_segmentation_union.npy
tables/cme_event_catalog.csv
visualizations/
manifest.json
integrated_run_summary.json
report.md
pipeline.log
```

Raw FITS remain outside the run under `data/raw/cor2/`. New runs **never
overwrite** existing runs. For another analysis date, supply a new
`--date YYYYMMDD` and unique `--run-name`, optionally increasing
`--warmup-days` for the causal 24-hour detection history.

## 6. Recover an interrupted run

Repeat the **identical** command with `--resume` appended:

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 \
  --run-name 20101212_cor2a_e2e_v1 \
  --source download \
  --warmup-days 1 \
  --resume
```

Resume compares the recorded request (config content, mask checksum and
input arguments). Completed acquisition, RD, C0 and E1 outputs can be reused.
Ambiguous or partially written stages fail closed; do **not** manually delete
their files. Instead inspect and preserve them, then choose a new run name.
This resume implementation requires local execution validation.

## 7. Regression comparison with the prior 63-pair baseline

The **old-mask general** 2010-12-12 run produced 63 RD pairs, 31 evaluation
frames, 201 E1 tracks, 15 admitted families, 4,661 recovered patches and
45,504 union-mask pixels. Its 31×256×256 final array reproduced pixel-for-pixel
(0 changed pixels) in a separate checkout.

To verify that reorganizing outputs did not change the algorithm, explicitly
use the **same old mask**:

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 \
  --run-name 20101212_oldmask_layout_regression \
  --source download \
  --warmup-days 1 \
  --mask-path Calibration/cor2_mask.fts
```

This requires the verified legacy mask to have been installed separately.
Then compare the array against the prior checkout:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
old = np.load(Path.home() / "stereo_cor2_e2e_test/runs/raw_e2e_baseline_verify/arrays/final_pixel_segmentation_union.npy")
new = np.load("runs/20101212_oldmask_layout_regression/arrays/final_pixel_segmentation_union.npy")
print("Shapes:", old.shape, new.shape)
print("Exact match:", np.array_equal(old, new))
if old.shape == new.shape:
    print("Differing pixels:", np.count_nonzero(old != new))
PY
```

**Expected:** both shapes `(31,256,256)`, `Exact match: True`,
`Differing pixels: 0` **only if the FITS selection and legacy mask match**.
For new official COR2-A runs, pixel and CME counts may legitimately differ;
do not use the old-mask 45,504-pixel value as a pass/fail threshold. Instead
verify the input manifest and resulting products. Exact input file identity,
metadata and checksums provide stronger provenance than pair count alone.

## 8. Separate historical Frozen benchmark

```bash
python -m cme_pipeline run \
  --config configs/baseline_rule_b.yaml --force
```

This runs the historical benchmark based on available frozen fixtures and
writes `runs/baseline_rule_b/`. The original 32-frame target is
219 tracks, 15 admitted families and 50,665 segmented pixels, with historical
external-catalog comparisons. It is **not** the general raw FITS mode.

## Troubleshooting and scope

- **Empty/corrupt mask:** install the real FITS; do not replace COR2-A with
  legacy `cor2_mask.fts` without explicit `--mask-path`.
- **No 30-minute matching phase:** the code infers observed `DATE-OBS` phase;
  inspect file/date availability or pass `--phase-sec` deliberately.
- **Partial raw download:** rerun with the same `--resume` request; cached
  completed source/dates are reused, missing dates can be downloaded.
- **External validation skipped:** intentional in general mode; the historical
  32-frame catalog validator runs only with `--frozen-regression`.
- **Scientific caveat:** E1 general-date geometry retains some historical
  scales; other observation dates require independent physical validation.
- **Git hygiene:** `data/` and `runs/` are gitignored; never commit raw FITS,
  temporary `.part` files or generated visualizations.

For storage details and safe migration, read
[Run/data layout](docs/RUN_DATA_LAYOUT.md). The high-level overview is in
[README.md](README.md).
