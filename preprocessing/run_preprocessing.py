#!/usr/bin/env python3
"""CLI Script to preprocess raw STEREO COR2 Beacon FITS into science-ready RD arrays."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from .calibration import calibrate_raw_fits, load_occulter_mask
from .reproject import derive_target_wcs, reproject_to_target, calculate_geometry
from .detector import compute_frame_zscore, causal_rolling_baseline, compute_patch_activities


def main():
    parser = argparse.ArgumentParser(description="Preprocess STEREO COR2 Beacon FITS data.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing raw .fts / .fits files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save preprocessed science arrays")
    parser.add_argument("--calibration-dir", type=str, default="calibration", help="Directory containing cor2_mask.fts")
    parser.add_argument("--date-str", type=int, default=20101212, help="Observation date (YYYYMMDD)")
    parser.add_argument("--cadence-min", type=float, default=30.0, help="Target pair difference cadence in minutes")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    calib_dir = Path(args.calibration_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning raw FITS in: {input_dir}")
    fits_files = sorted(list(input_dir.glob("*.fts")) + list(input_dir.glob("*.fits")))
    if not fits_files:
        print(f"No FITS files found in {input_dir}.")
        return

    print(f"Found {len(fits_files)} raw FITS files.")

    # 1. Read headers and sort by date
    headers = []
    file_records = []
    for f in fits_files:
        hdr = fits.getheader(f)
        date_obs = str(hdr.get("DATE-OBS", ""))
        headers.append(hdr)
        file_records.append({"path": f, "date_obs": date_obs, "header": hdr})

    file_records.sort(key=lambda x: x["date_obs"])

    # 2. Derive target North-Up WCS grid
    print("Deriving solar-centered North-Up target WCS grid...")
    target_wcs = derive_target_wcs([r["header"] for r in file_records], target_shape=(256, 256))
    pa_raster, radius_raster, center_xy = calculate_geometry(target_wcs, shape=(256, 256))

    # Save target WCS header text
    hdr_txt_path = output_dir / "target_wcs_header.txt"
    hdr_txt_path.write_text(target_wcs.to_header().tostring(sep="\n"), encoding="utf-8")
    print(f"Saved target WCS header to: {hdr_txt_path}")

    # 3. Load occulter mask
    mask_fits = calib_dir / "cor2_mask.fts"
    if mask_fits.is_file():
        print(f"Loading occulter mask from: {mask_fits}")
        static_valid_mask = load_occulter_mask(mask_fits, target_wcs, shape=(256, 256))
    else:
        print("Warning: cor2_mask.fts not found. Using default FOV circular mask.")
        static_valid_mask = (radius_raster >= 22.0) & (radius_raster <= 122.5)

    # 4. Form 30-min pairs
    print(f"Forming ~{args.cadence_min}-minute observation pairs...")
    pairs = []
    target_sec = args.cadence_min * 60.0
    tolerance_sec = 600.0  # 10 min tolerance

    for i in range(len(file_records)):
        t_i = datetime.fromisoformat(file_records[i]["date_obs"])
        for j in range(i + 1, len(file_records)):
            t_j = datetime.fromisoformat(file_records[j]["date_obs"])
            dt = (t_j - t_i).total_seconds()
            if abs(dt - target_sec) <= tolerance_sec:
                pairs.append((file_records[i], file_records[j], dt / 60.0))
                break

    print(f"Constructed {len(pairs)} Running Difference pairs.")
    if not pairs:
        print("No valid pairs formed.")
        return

    # 5. Process pairs into arrays
    rd_list = []
    val_list = []
    manifest_rows = []

    for p_idx, (rec_start, rec_end, dt_min) in enumerate(pairs, start=1):
        # Calibrate start and end frames
        d_start, hdr_start, _, _ = calibrate_raw_fits(rec_start["path"])
        d_end, hdr_end, _, _ = calibrate_raw_fits(rec_end["path"])

        wcs_start = WCS(hdr_start).celestial
        wcs_end = WCS(hdr_end).celestial

        # Reproject to target grid
        r_start, fp_start = reproject_to_target(d_start, wcs_start, target_wcs, target_shape=(256, 256))
        r_end, fp_end = reproject_to_target(d_end, wcs_end, target_wcs, target_shape=(256, 256))

        pair_val = static_valid_mask & (fp_start > 0.95) & (fp_end > 0.95) & np.isfinite(r_start) & np.isfinite(r_end)
        rd = np.zeros((256, 256), dtype=np.float32)
        rd[pair_val] = (r_end[pair_val] - r_start[pair_val])

        rd_list.append(rd)
        val_list.append(pair_val)

        manifest_rows.append({
            "pair_index": p_idx,
            "rd_pair_id": f"{args.date_str}_{rec_end['date_obs'].replace(':', '').replace('-', '')}",
            "date_str": args.date_str,
            "start_time": rec_start["date_obs"],
            "end_time": rec_end["date_obs"],
            "delta_t_min": dt_min,
        })

    # Save science arrays
    rd_array = np.stack(rd_list, axis=0)
    val_array = np.stack(val_list, axis=0)
    df_man = pd.DataFrame(manifest_rows)

    np.save(output_dir / "north_up_RD_DNs.npy", rd_array)
    np.save(output_dir / "north_up_pair_valid_masks.npy", val_array)
    df_man.to_csv(output_dir / "01_rd_manifest.csv", index=False)

    print(f"\nPreprocessing successfully completed!")
    print(f"  Science RD Arrays : {output_dir / 'north_up_RD_DNs.npy'} {rd_array.shape}")
    print(f"  Valid Masks       : {output_dir / 'north_up_pair_valid_masks.npy'} {val_array.shape}")
    print(f"  Manifest CSV      : {output_dir / '01_rd_manifest.csv'} ({len(df_man)} pairs)")


if __name__ == "__main__":
    main()

