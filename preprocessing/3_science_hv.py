import os
import numpy as np
from astropy.io import fits
import functions as scc
from common_utils import get_directories, scan_files, load_fits

# --- Configuration ---
DATE_STR = "20240510"
OUTPUT_DIR = "./RESULTS/3_science_hv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_science_hv():
    # 1. Set directories and scan for files (Using Science Raw data)
    dirs = get_directories(DATE_STR)
    s_single_files = scan_files(dirs['science_raw'], filter_size=True) # Over 1MB only
    
    print(f"Starting Science-Helioviewer Style (Simple Prep + RD)")
    print(f"Target Date: {DATE_STR} | Files found: {len(s_single_files)}")

    if len(s_single_files) < 2:
        print("Error: At least two files are required to create a running difference.")
        return

    # Initialization for the first image
    prev_file = s_single_files[0]
    prev_name = os.path.basename(prev_file)
    
    with fits.open(prev_file) as hdul:
        prev_data = hdul[0].data.astype(np.float32)
        prev_hdr = hdul[0].header
        
        # [HV Prep] Trim & Exposure Normalization
        prev_im, _ = scc.scc_img_trim(prev_data, prev_hdr, silent=True)
        exptime = prev_hdr.get('EXPTIME', 1.0)
        if exptime > 0: 
            prev_im /= exptime

    # 2. Iterate through files to calculate simple running difference (Helioviewer style)
    for curr_file in s_single_files[1:]:
        try:
            curr_name = os.path.basename(curr_file)
            with fits.open(curr_file) as hdul:
                curr_data = hdul[0].data.astype(np.float32)
                curr_hdr = hdul[0].header

                # [HV Prep] Trim & Exposure Normalization
                curr_im, curr_hdr = scc.scc_img_trim(curr_data, curr_hdr, silent=True)
                exptime = curr_hdr.get('EXPTIME', 1.0)
                if exptime > 0: 
                    curr_im /= exptime

                # [Helioviewer Style Running Difference]
                # Calculate direct intensity difference without Z-score normalization.
                # Ensure array shapes match to prevent errors (handling potential trimming variations)
                if curr_im.shape != prev_im.shape:
                    from common_utils import rebin
                    prev_im = rebin(prev_im, curr_im.shape)

                rd_hv = curr_im - prev_im

                # --- Save Result ---
                save_name = curr_name.replace(".fts", "_RD_HV.fts")
                save_path = os.path.join(OUTPUT_DIR, save_name)
                
                # Set VMIN/VMAX for visualization using percentiles
                vmin, vmax = np.nanpercentile(rd_hv, [1.0, 99.0])
                
                curr_hdr['DATATYPE'] = 'Running Difference (Simple)'
                curr_hdr['HISTORY'] = f"HV Style RD (Exptime Normalized): {curr_name} - {prev_name}"
                curr_hdr['VMIN'] = float(vmin)
                curr_hdr['VMAX'] = float(vmax)
                
                fits.writeto(save_path, rd_hv, curr_hdr, overwrite=True)
                print(f"Saved: {save_name} (vs {prev_name})")

                # Update variables for next iteration
                prev_im = curr_im
                prev_name = curr_name

        except Exception as e:
            print(f"Error processing {curr_name}: {e}")

if __name__ == "__main__":
    process_science_hv()