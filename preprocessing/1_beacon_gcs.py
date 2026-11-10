import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from common_utils import get_directories, scan_files, load_fits, gui_step1_diff_zscore, gui_step2_plot_params

# --- Configuration ---
DATE_STR = "20240510"
OUTPUT_DIR = "./RESULTS/1_beacon_gcs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_beacon_gcs():
    # 1. Fetch directories and scan files using common utilities
    dirs = get_directories(DATE_STR)
    b_single_files = scan_files(dirs['beacon_raw'])
    
    print(f"Starting Beacon-GCS Style (Z-score RD)")
    print(f"Target Date: {DATE_STR} | Files found: {len(b_single_files)}")

    if len(b_single_files) < 2:
        print("Error: At least two files are required to create a running difference.")
        return

    # 2. Set the first image as the initial previous image (prev_img)
    prev_file = b_single_files[0]
    prev_name = os.path.basename(prev_file)
    b_prev_raw = load_fits(prev_file)

    # 3. Iterate through files to calculate and save running differences
    # Start from index 1 (the second file) to compare with the previous one
    for curr_file in b_single_files[1:]:
        try:
            curr_name = os.path.basename(curr_file)
            b_curr_raw = load_fits(curr_file)

            # [Module 1] Calculate Z-score Difference
            b_rd_zscore = gui_step1_diff_zscore(b_prev_raw, b_curr_raw)

            # [Module 2] Calculate Visualization Range
            vmin_z, vmax_z = gui_step2_plot_params(b_rd_zscore, sigma_level=2)

            # --- Save Result ---
            save_name = curr_name.replace(".fts", "_RD_GCS.fts")
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            # Retrieve the current file's header to preserve original metadata
            with fits.open(curr_file) as hdul:
                hdr = hdul[0].header
            
            hdr['DATATYPE'] = 'Running Difference (Z-score)'
            hdr['HISTORY'] = f"GCS Style RD (Raw): {curr_name} - {prev_name}"
            hdr['VMIN'] = float(vmin_z)
            hdr['VMAX'] = float(vmax_z)
            
            fits.writeto(save_path, b_rd_zscore, hdr, overwrite=True)
            print(f"Saved: {save_name} (vs {prev_name})")

            # Update variables for the next iteration
            b_prev_raw = b_curr_raw
            prev_name = curr_name

        except Exception as e:
            print(f"Error processing {curr_name}: {e}")

if __name__ == "__main__":
    process_beacon_gcs()