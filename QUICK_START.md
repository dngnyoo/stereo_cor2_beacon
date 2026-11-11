# Quick Start

Reproduce the **2010-12-12 STEREO-A COR2 CME benchmark** using the included frozen detection inputs.

## Input

The pipeline uses precomputed running-difference images, masks, tracking seeds, and reference-catalog data stored in:

```text
validation_20101212/
```

No separate data download is required for this benchmark. This is **not** a raw-FITS-to-detection pipeline.

## Run the baseline

From the repository root (Linux / WSL; Python 3.12):

```bash
git clone https://github.com/dngnyoo/stereo_cor2_beacon.git
cd stereo_cor2_beacon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m cme_pipeline run --config configs/baseline_rule_b.yaml --force
```

Baseline configuration: `configs/baseline_rule_b.yaml` (E1 tracking + Rule B family association).

## Output

All generated results are saved under:

```text
runs/baseline_rule_b/
├── arrays/
│   ├── final_pixel_segmentation_union.npy  # 32-frame pixel masks
│   └── family_pixel_masks.npz              # Per-family masks
├── tables/
│   ├── cme_event_catalog.csv               # CME candidate summary
│   └── cme_frame_measurements.csv          # Per-frame measurements
├── visualizations/
│   ├── segmentation_overlays/              # Pair/UTC-labeled PNGs
│   ├── family_tracking/                    # Family PNGs
│   ├── height_time/                        # Height–time PNGs
│   └── cme_animation.gif
├── report.md                               # Summary and validation
└── manifest.json                           # Run metadata
```

Expected baseline: **32 frames, 219 tracks, 15 admitted families, 50,665 segmented union pixels**.

The event catalog contains algorithm-generated **CME candidates**, not independently confirmed physical events.

For implementation details, see [README.md](README.md) and [cme_pipeline/README.md](cme_pipeline/README.md).
