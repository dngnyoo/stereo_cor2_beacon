import os
import numpy as np
from astropy.io import fits
from common_utils import get_directories, scan_files, cor2_prep_level1, synthesize_pb, get_zscore_rd

# --- Configuration ---
DATE_STR = "20240510"
OUTPUT_DIR = "./RESULTS/2_beacon_cmad"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_beacon_cmad():
    # 1. Initialize directories and scan for sequence files
    dirs = get_directories(DATE_STR)
    b_seq_files = scan_files(dirs['beacon_seq'])
    
    print(f"Starting Beacon-CMAD Pipeline (idx_c vs idx_c+3 with Z-score RD)")
    
    pb_history = []
    header_history = []

    # 2. Group files in triplets to synthesize pB (0, 120, 240 degree sets)
    for i in range(0, len(b_seq_files) - 2, 3):
        triplet_paths = b_seq_files[i : i + 3]
        l1_imgs = []
        base_hdr = None

        for path in triplet_paths:
            with fits.open(path) as hdul:
                data, hdr = hdul[0].data, hdul[0].header
                # [L1 Prep] Physical correction including CMAD background subtraction
                l1_data, l1_hdr = cor2_prep_level1(data, hdr, dirs['cal_dir'], dirs['bkg_base'])
                l1_imgs.append(l1_data)
                if base_hdr is None: 
                    base_hdr = l1_hdr

        # [L2 Synthesis] Synthesize Polarized Brightness (pB)
        pb = synthesize_pb(l1_imgs[0], l1_imgs[1], l1_imgs[2])
        pb_history.append(pb)
        header_history.append(base_hdr)
        print(f"pB set {len(pb_history)} synthesized.")

    # 3. Calculate Z-score Running Difference between adjacent pB sets
    for j in range(len(pb_history) - 1):
        prev_pb = pb_history[j]
        curr_pb = pb_history[j+1]
        
        # [Z-score RD] Standardized difference calculation
        rd_z = get_zscore_rd(prev_pb, curr_pb)
        
        # Set output filename (e.g., difference between set j and j+1)
        save_name = f"Beacon_CMAD_pB_RD_seq_{j:02d}.fts"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        # Update FITS header metadata
        hdr = header_history[j+1]
        hdr['DATATYPE'] = 'pB Z-score Running Difference'
        hdr['HISTORY'] = 'L1(CMAD) -> L2(pB) -> Z-score RD'
        
        fits.writeto(save_path, rd_z, hdr, overwrite=True)
        print(f"Saved: {save_name}")

if __name__ == "__main__":
    process_beacon_cmad()