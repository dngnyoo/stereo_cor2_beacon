import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from astropy.io import fits
import numpy as np

# Let's inspect the geometry context / WCS header from runs/shared_d4_20101212 or 01_preprocessing
pre_dir = Path("/home/aababc/stereo_cor2_clean/runs/dual_archive_20101212/preprocessed")
meta_path = pre_dir / "preprocessing_metadata.json"
wcs_path = pre_dir / "target_wcs_header.txt"
man_path = pre_dir / "01_rd_manifest.csv"

print("Checking preprocessed dir:", pre_dir)
if meta_path.exists():
    print("Metadata:", meta_path.read_text())

if wcs_path.exists():
    hdr_text = wcs_path.read_text()
    print("WCS Header sample:")
    for line in hdr_text.splitlines()[:20]:
        print(" ", line)

# Also check tracking metadata or context in exp_20101212_a5b_baseline or nr1
trk_meta = (ROOT / "runs/exp_20101212_a5b_nr1/03_tracking/data/tracking_metadata.json")
if trk_meta.exists():
    print("Tracking metadata:", trk_meta.read_text()[:500])

# Check frozen detection geometry
from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs

cfg = PipelineConfig()
cfg.date_str = "20101212"
ctx = PipelineContext(cfg)
det_out = load_frozen_detection_inputs(cfg, ctx)
geom = det_out.geometry
print("\nFrozen detection GeometryContext:")
print("  rsun_px:", geom.rsun_px)
print("  plate_scale_arcsec:", geom.plate_scale_arcsec)
print("  center_xy:", geom.center_xy)
