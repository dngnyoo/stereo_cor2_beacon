import os
import numpy as np
from astropy.io import fits
from common_utils import (
    get_directories, scan_files, load_fits, 
    cor2_prep_level1, synthesize_pb, get_zscore_rd
)

# --- Configuration ---
DATE_STR = "20240510"
OUTPUT_DIR = "./RESULTS/4_science_cmad"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_science_cmad():
    # 1. Set directories and scan for files (Using Science Sequence data)
    dirs = get_directories(DATE_STR)
    # Apply filter_size=True for high-resolution science data (at least 1MB)
    s_seq_files = scan_files(dirs['science_seq'], filter_size=True)
    
    print(f"Starting Science-CMAD Pipeline (High-Precision pB Z-score RD)")
    print(f"Target Date: {DATE_STR} | Total Sequence Files: {len(s_seq_files)}")

    pb_history = []
    header_history = []

    # 2. Process triplets for high-precision L1 correction and pB synthesis (0, 120, 240 degree sets)
    # Science data sequences typically consist of 3-frame polarization sets.
    for i in range(0, len(s_seq_files) - 2, 3):
        triplet_paths = s_seq_files[i : i + 3]
        l1_imgs = []
        base_hdr = None

        print(f"Processing Sequence: {os.path.basename(triplet_paths[0])}...")

        for path in triplet_paths:
            with fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float32)
                hdr = hdul[0].header
                
                # [L1 High-Precision] 
                # Includes CMAD BKG subtraction, warping, unit conversion (MSB), and masking
                l1_data, l1_hdr = cor2_prep_level1(
                    data, hdr, dirs['cal_dir'], dirs['bkg_base'], silent=True
                )
                l1_imgs.append(l1_data)
                if base_hdr is None: 
                    base_hdr = l1_hdr

        # [L2 Synthesis] Synthesize Polarized Brightness (pB) from the 3-angle triplet
        pb = synthesize_pb(l1_imgs[0], l1_imgs[1], l1_imgs[2])
        pb_history.append(pb)
        header_history.append(base_hdr)

    # 3. Calculate Z-score Running Difference between adjacent pB sets
    # Extracts subtle plasma changes (CME) after physical background (F-corona) removal.
    for j in range(len(pb_history) - 1):
        prev_pb = pb_history[j]
        curr_pb = pb_history[j+1]
        
        # [Z-score RD] Calculate standardized difference after data distribution normalization
        rd_z = get_zscore_rd(prev_pb, curr_pb)
        
        # Save results
        save_name = f"Science_CMAD_pB_RD_seq_{j:02d}.fts"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        # Update FITS header metadata
        hdr = header_history[j+1]
        hdr['DATATYPE'] = 'Science pB Z-score RD'
        hdr['HISTORY'] = 'L1(Science CMAD) -> L2(pB) -> Z-score RD'
        
        # Record visualization parameters based on Sigma=2 range
        from common_utils import gui_step2_plot_params
        vmin, vmax = gui_step2_plot_params(rd_z, sigma_level=2)
        hdr['VMIN'] = float(vmin)
        hdr['VMAX'] = float(vmax)
        
        fits.writeto(save_path, rd_z, hdr, overwrite=True)
        print(f"Saved High-Res RD: {save_name}")

if __name__ == "__main__":
    process_science_cmad()