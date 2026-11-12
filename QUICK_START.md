# Quick Start

Run the STEREO-A COR2 pipeline from NASA raw FITS to CME catalog.

**Requirements:** Linux/WSL and Python 3.12.

## 1. Install

```bash
git clone https://github.com/dngnyoo/stereo_cor2_beacon.git
cd stereo_cor2_beacon
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The official `Calibration/cor2A_mask.fts` is included in the repository. No extra calibration setup is needed.

## 2. Run

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 \
  --run-name cor2a_20101212 \
  --source download \
  --warmup-days 1
```

Change `--date` and `--run-name` for another experiment. Downloaded FITS are kept by date in `data/raw/cor2/` and reused automatically. To continue an interrupted run, repeat the same command with `--resume`.

## 3. Results

```text
runs/cor2a_20101212/
├── acquisition/preprocessed/   # Running-difference data
├── detection/                  # C0
├── tracking/                   # E1
├── arrays/                     # Segmentation masks
├── tables/                     # CME candidate catalog
├── visualizations/             # Images and animation
└── integrated_run_summary.json
```

See [README.md](README.md) for details and [Run/data layout](docs/RUN_DATA_LAYOUT.md) for storage and migration.
