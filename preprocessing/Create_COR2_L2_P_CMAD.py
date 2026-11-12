from pathlib import Path
import os
import numpy as np
from astropy.io import fits

# 🌟 Import the L1 correction module which includes updated Rebin and CMAD background subtraction logic
try:
    from Create_COR2_L1_P import cor2_prep_level1 
except ImportError:
    print("❌ Error: Create_COR2_L1_P.py not found. Please check the file path.")

def synthesize_pb(i0, i120, i240):
    """
    Calculates polarized brightness (pB) using three polarized images (0, 120, 240 degrees).
    Input data must be background-subtracted and calibrated to MSB units.
    """
    term1 = (i0 + i120 + i240)**2
    term2 = 3 * (i0*i120 + i120*i240 + i240*i0)
    diff = term1 - term2
    
    # Handle unphysical negative values (noise) by clipping to zero to prevent errors during sqrt
    diff[diff < 0] = 0
    pb = (4.0 / 3.0) * np.sqrt(diff)
    return pb

def process_l2_synthesis(input_dir, output_dir, cal_dir, bkg_dir):
    """Processes a directory of raw files to synthesize L2 pB images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Scan and sort observation files
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.fts')])
    print(f"📂 Scanning {len(files)} files to start pB (CMAD) synthesis...")
    
    triple_buffer = {}
    success_count = 0

    for f in files:
        f_path = os.path.join(input_dir, f)
        try:
            with fits.open(f_path) as hdul:
                hdr = hdul[0].header
                data = hdul[0].data.astype(np.float32)
                pol = float(hdr.get('POLAR', -1))
                
                # Store in buffer to form a polarization set (0, 120, 240 degrees)
                if pol == 0.0:
                    triple_buffer = {0.0: (data, hdr, f)}
                elif pol in [120.0, 240.0] and 0.0 in triple_buffer:
                    triple_buffer[pol] = (data, hdr, f)

                # Once all 3 images are collected, apply latest correction logic and synthesize
                if len(triple_buffer) == 3:
                    print(f"✅ Set found: {triple_buffer[0.0][2]} and two others")
                    
                    l1_processed_imgs = {}
                    final_hdr = None
                    
                    for p in [0.0, 120.0, 240.0]:
                        raw_data, raw_hdr, fname = triple_buffer[p]
                        # 🌟 [CRITICAL] Perform L1 correction with Rebin and CMAD sequence
                        # Science mode (1024) and Beacon mode (256) are handled automatically
                        l1_data, l1_hdr = cor2_prep_level1(raw_data, raw_hdr, cal_dir, bkg_dir, silent=True)
                        l1_processed_imgs[p] = l1_data
                        
                        if p == 0.0:
                            final_hdr = l1_hdr.copy()

                    # pB Synthesis (Unit: MSB)
                    pb_image = synthesize_pb(
                        l1_processed_imgs[0.0], 
                        l1_processed_imgs[120.0], 
                        l1_processed_imgs[240.0]
                    )
                    
                    # Convert filename and save (Resolution info is inherited from l1_hdr)
                    # Supports both Beacon (n7c2) and Science (n4c2) filename patterns
                    save_name = triple_buffer[0.0][2].replace('n7c2A.fts', 'pB_CMAD.fts').replace('n4c2A.fts', 'pB_CMAD.fts')
                    save_path = os.path.join(output_dir, save_name)
                    
                    final_hdr['DATATYPE'] = 'Polarized Brightness (pB)'
                    final_hdr['BUNIT'] = 'MSB' # Explicitly state the final unit
                    final_hdr['HISTORY'] = 'L1 Prep (Rebin & CMAD) -> L2 pB Synthesized'
                    
                    fits.writeto(save_path, pb_image, final_hdr, overwrite=True)
                    
                    print(f"   -> 💾 pB synthesis and save complete: {save_name}")
                    success_count += 1
                    triple_buffer = {} # Reset buffer for the next sequence

        except Exception as e:
            print(f"❌ Error processing {f}: {e}")

    print(f"\n🎯 Job complete! A total of {success_count} high-quality pB images were generated.")

if __name__ == "__main__":
    # User Environment Path Configuration
    RAW_INPUT_DIR = "/home/aababc/stereo_project/cor/STEREO_DATA/STA/science/20120723" 
    L2_OUTPUT_DIR = "/home/aababc/stereo_project/cor/L2_PB_CMAD_RESULTS"
    CAL_DIR = str(Path(__file__).resolve().parents[1] / "Calibration")
    BKG_DIR = str(Path(__file__).resolve().parents[1] / "Calibration" / "backgrounds")
    
    process_l2_synthesis(RAW_INPUT_DIR, L2_OUTPUT_DIR, CAL_DIR, BKG_DIR)