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
