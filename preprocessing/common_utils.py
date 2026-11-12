import numpy as np
from scipy.ndimage import map_coordinates
import functions as scc

def rebin(a, new_shape):
    """Rebin a 2D array using block-averaging."""
    if a.shape == new_shape:
        return a
    factor_y = a.shape[0] // new_shape[0]
    factor_x = a.shape[1] // new_shape[1]
    factor_y = max(1, factor_y)
    factor_x = max(1, factor_x)
    try:
        reshaped = a.reshape(new_shape[0], factor_y, new_shape[1], factor_x)
        return reshaped.mean(axis=3).mean(axis=1)
    except ValueError:
        from scipy.ndimage import zoom
        zoom_factor = (new_shape[0] / a.shape[0], new_shape[1] / a.shape[1])
        return zoom(a, zoom_factor, order=1)

def get_running_diff(curr_img, prev_img):
    """Calculate the running difference between two images."""
    if curr_img.shape != prev_img.shape:
        prev_img = rebin(prev_img, curr_img.shape)
    return curr_img - prev_img

import os
import glob
import numpy as np
from astropy.io import fits

# --- 1. Path Configuration (Global Constants) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(PROJECT_ROOT, "DATA")

def get_directories(date_str):
    return {
        "beacon_raw": os.path.join(BASE_DIR, "STEREO_DATA/STA/beacon", date_str),
        "science_raw": os.path.join(BASE_DIR, "STEREO_DATA/STA/science", date_str),
        "beacon_seq": os.path.join(BASE_DIR, "STEREO_DATA_seq/STA/Beacon_seq", date_str),
        "science_seq": os.path.join(BASE_DIR, "STEREO_DATA_seq/STA/Science_seq", date_str),
        "l2_dir": os.path.join(BASE_DIR, f"L2_RESULTS_{date_str[2:]}"),
        "bkg_base": os.path.join(PROJECT_ROOT, "Calibration", "backgrounds"),
        "cal_dir": os.path.join(PROJECT_ROOT, "Calibration")
    }

# --- 2. Data Loading & Filtering ---
MIN_SIZE_BYTES = 1048576  # 1MB

def scan_files(directory, filter_size=False):
    """Scans for .fts files and optionally filters by minimum file size."""
    if not os.path.exists(directory):
        print(f"⚠️ Warning: Directory not found - {directory}")
        return []
        
    files = sorted(glob.glob(os.path.join(directory, "*.fts")))
    if not filter_size:
        return files
    return [f for f in files if os.path.getsize(f) >= MIN_SIZE_BYTES]

def load_fits(path):
    """Loads FITS file and converts NaNs to 0.0."""
    with fits.open(path) as h:
        return np.nan_to_num(h[0].data.astype(np.float32), nan=0.0)

# --- 3. Processing Algorithms ---
def gui_step1_diff_zscore(prev_data, curr_data):
    """Normalizes both frames using Z-score and calculates the difference."""
    a_norm = (prev_data - np.mean(prev_data)) / np.std(prev_data)
    b_norm = (curr_data - np.mean(curr_data)) / np.std(curr_data)
    return b_norm - a_norm

def gui_step2_plot_params(diff_data, sigma_level=2):
    """Calculates vmin and vmax based on N-sigma clipping."""
    mean_val = np.mean(diff_data)
    std_val = np.std(diff_data)
    vmin = mean_val - sigma_level * std_val
    vmax = mean_val + sigma_level * std_val
    return vmin, vmax

def cor2_prep_level1(data, header, cal_dir, bkg_base_dir, silent=True):
    """(Trim -> Exptime -> CMAD BKG -> Calibration -> Warp)"""
    im = data.copy().astype(np.float32)
    hdr = header.copy()
    obs = hdr.get('OBSRVTRY', 'STEREO_A').strip().upper()
    obs_char = 'A' if 'A' in obs else 'B'

    # 1. Trim & Exposure
    im, hdr = scc.scc_img_trim(im, hdr, silent=silent)
    exptime = hdr.get('EXPTIME', 1.0)
    if exptime > 0: im /= exptime

    # 2. Background Subtraction (CMAD logic)
    pol = float(hdr.get('POLAR', 0.0))
    date_obs = hdr.get('DATE-OBS', '')[:10].replace('-', '')
    yyyymm, yymmdd = date_obs[:6], date_obs[2:8]
    
    bkg_dir = os.path.join(bkg_base_dir, "monthly_min", obs_char.lower(), yyyymm)
    bkg_pattern = os.path.join(bkg_dir, f"mc2{obs_char}_p{int(pol):03}_*.fts")
    available_bkgs = sorted(glob.glob(bkg_pattern))

    if available_bkgs:
        valid_bkgs = [f for f in available_bkgs if os.path.basename(f).split('_')[-1][:6] <= yymmdd]
        bkg_file = valid_bkgs[-1] if valid_bkgs else available_bkgs[0]
        with fits.open(bkg_file) as b_hdul:
            b_data = b_hdul[0].data.astype(np.float32)
            im -= rebin(b_data, im.shape) # Rebin 적용하여 차감

    # 3. Calibration & Units
    calfac, hdr = scc.get_calfac(hdr, conv='MSB', silent=silent)
    im *= calfac
    
    # 4. Warp (Distortion Correction)
    # im, hdr = cor2_warp(im, hdr) 
    
    return im, hdr

# --- 3. L2 Core Functions (from Create_COR2_L2_P_CMAD.py) ---

def synthesize_pb(i0, i120, i240):
    term1 = (i0 + i120 + i240)**2
    term2 = 3 * (i0*i120 + i120*i240 + i240*i0)
    diff = np.clip(term1 - term2, 0, None)
    return (4.0 / 3.0) * np.sqrt(diff)

def get_zscore_rd(prev_pb, curr_pb):
    mu_p, std_p = np.nanmean(prev_pb), np.nanstd(prev_pb)
    mu_c, std_c = np.nanmean(curr_pb), np.nanstd(curr_pb)
    z_p = (prev_pb - mu_p) / (std_p if std_p > 0 else 1)
    z_c = (curr_pb - mu_c) / (std_c if std_c > 0 else 1)
    return z_c - z_p