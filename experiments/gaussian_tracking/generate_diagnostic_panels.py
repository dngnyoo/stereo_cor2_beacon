"""Generates 4-panel diagnostic figures for key CME progression frames."""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from cme_pipeline.cadence_selection import parse_timestamp
from cme_pipeline.utils.geometry import compute_wcs_rasters
from experiments.gaussian_tracking.gaussian_evidence_model import (
    compute_pixel_evidence,
    apply_spatial_gaussian_spreading,
)
from experiments.gaussian_tracking.visualize_results import plot_frame_diagnostic_panel

data_dir = root / "runs/dual_archive_20101212/preprocessed"
vis_dir = root / "experiments/gaussian_tracking/visualizations"
vis_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(data_dir / "01_rd_manifest.csv")
rd_images = np.load(data_dir / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
valid_masks = np.load(data_dir / "north_up_pair_valid_masks.npy", mmap_mode="r")

hdr_str = (data_dir / "target_wcs_header.txt").read_text(encoding="utf-8")
wcs = WCS(fits.Header.fromstring(hdr_str, sep="\n"))
sx, sy = wcs.world_to_pixel_values(0.0, 0.0)
rasters = compute_wcs_rasters(wcs=wcs, center_xy=(sx, sy), patch_size=4)

df_g1 = pd.read_csv(root / "experiments/gaussian_tracking/results/tracks_G1_Spatial_sigma2.csv")
df_g3 = pd.read_csv(root / "experiments/gaussian_tracking/results/tracks_G3_MotionAware_sigma2.csv")

target_pairs = [35, 38, 41, 46, 52, 61]

for pair_idx in target_pairs:
    row = manifest[manifest["pair_index"] == pair_idx]
    if row.empty:
        continue
    f_idx = int(row.index[0])
    t_stamp = parse_timestamp(row.iloc[0]["end_time"])

    image = np.asarray(rd_images[f_idx], dtype=np.float32)
    val = np.asarray(valid_masks[f_idx], dtype=bool)

    # Frame smoothed z-score
    num = gaussian_filter(np.where(val, image, 0.0), 1.0, mode="constant", cval=0.0)
    den = gaussian_filter(val.astype(float), 1.0, mode="constant", cval=0.0)
    good = val & (den > 0)
    smoothed = np.full(image.shape, np.nan, dtype=np.float32)
    smoothed[good] = (num[good] / den[good]).astype(np.float32)
    fin = smoothed[good & np.isfinite(smoothed)]
    z = np.zeros(image.shape, dtype=np.float32)
    if fin.size and float(fin.std()) > 0:
        z[good] = (smoothed[good] - float(fin.mean())) / float(fin.std())

    # Raw evidence & spread evidence (sigma=2px)
    ev_raw = compute_pixel_evidence(z, val, formulation="positive", radius_raster=rasters["radius_raster"])
    ev_spread = apply_spatial_gaussian_spreading(ev_raw, val, sigma_s=2.0, radius_raster=rasters["radius_raster"])

    out_file = vis_dir / f"diag_tracking_pair_{pair_idx}.png"
    plot_frame_diagnostic_panel(
        frame_idx=f_idx,
        timestamp=t_stamp,
        rd_image=image,
        z_map=z,
        spread_ev=ev_spread,
        valid_mask=val,
        sx=sx,
        sy=sy,
        r_min_px=22.0,
        r_max_px=122.5,
        df_tracks_g1=df_g1,
        df_tracks_g3=df_g3,
        output_path=out_file,
    )
