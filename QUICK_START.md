# Quick Start

Run STEREO-A COR2 CME detection and segmentation from NASA raw FITS to the final CME catalog.

**Requirements:** Linux/WSL, Python 3.12, and the verified `cor2A_mask.fts` calibration file.

## 1. Install

```bash
git clone https://github.com/dngnyoo/stereo_cor2_beacon.git
cd stereo_cor2_beacon

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Install the COR2-A mask

Place the **COR2-A** mask at `Calibration/cor2A_mask.fts`. On the reference MELONHOUSE machine:

```bash
cp ~/stereo_cor2/handover/calibration/cor2A_mask.fts Calibration/
sha256sum Calibration/cor2A_mask.fts
```

Expected SHA-256:

```text
7cde9474fa9d1fe93d9a6ff08464598d5d2250b423b00be449ca24b2ec7ca606
```

**Use `cor2A_mask.fts` only.** The older `cor2_mask.fts` was an incorrect mask for this project and is not the production calibration input. The repository may contain a zero-byte placeholder; replace it with the verified real FITS file.

## 3. Run the pipeline

Change the date and run name as needed:

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 \
  --run-name cor2a_20101212 \
  --source download \
  --warmup-days 1
```

The pipeline downloads missing NASA data to `data/raw/cor2/` and reuses files already downloaded for each date. All processing results are saved under `runs/cor2a_20101212/`.

## 4. View the results

```text
runs/cor2a_20101212/
├── acquisition/preprocessed/     # Running-difference images
├── detection/                    # C0 detection
├── tracking/                     # E1 tracking
├── arrays/                       # Final segmentation masks
├── tables/                       # CME candidate catalog
├── visualizations/               # Figures and animation
├── integrated_run_summary.json   # Summary
└── report.md                     # Report
```

To process another date, change `--date` and choose a **new** `--run-name`. To retry an interrupted run, repeat the same command with `--resume`.

For additional options, dataset migration, and technical details, see [README.md](README.md) and [Run/data layout](docs/RUN_DATA_LAYOUT.md).
