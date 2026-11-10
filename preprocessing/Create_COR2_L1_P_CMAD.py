import os
import glob
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates
import functions as scc  

def rebin(a, new_shape):
    """
    Rebin a 2D array using block-averaging.
    Optimized for integer multiples (e.g., 2048x2048 -> 256x256).
    """
    if a.shape == new_shape:
        return a

    # 1. Calculate reduction ratios (e.g., 2048 // 256 = 8)
    # Must follow the order: (Original / Target)
    factor_y = a.shape[0] // new_shape[0]
    factor_x = a.shape[1] // new_shape[1]

    # 2. Force factor to at least 1 to prevent division by zero errors
    factor_y = max(1, factor_y)
    factor_x = max(1, factor_x)

    # 3. Reshape and calculate the mean for each block
    try:
        reshaped = a.reshape(new_shape[0], factor_y, new_shape[1], factor_x)
        return reshaped.mean(axis=3).mean(axis=1)
    except ValueError:
        # If reshape fails, fallback to scipy zoom for safe resizing
        from scipy.ndimage import zoom
        zoom_factor = (new_shape[0] / a.shape[0], new_shape[1] / a.shape[1])
        return zoom(a, zoom_factor, order=1)

def cor2_sebip(data, header, silent=False):
    """Function to restore onboard compressed SEBIP data."""
    d = data.copy()
    ip_raw = header.get('IP_00_19', '')
    if not ip_raw.strip(): return d, header
    
    # Ensure the string length for proper indexing
    while len(ip_raw) < 60: ip_raw = ' ' + ip_raw
    ip = [ip_raw[i:i+3] for i in range(0, 60, 3)]
    
    if '117' in ip:
        ind = ip.index('117')
        ip = ip[ind:] 
    
    # Apply bit-shift corrections based on compression parameters
    if ip.count('  1') > 0: d *= (2.0 ** ip.count('  1'))
    if ip.count(' 16') + ip.count(' 17') > 0: d *= (64.0 ** (ip.count(' 16') + ip.count(' 17')))
    if ip.count(' 53') > 0 and header.get('IPSUM', 0) > 0: d *= (4.0 ** ip.count(' 53'))
    
    header['HISTORY'] = 'Applied Python cor2_sebip correction'
    return d, header

def cor2_warp(im, hdr, silent=False):
    """Function to correct optical distortion (Warps the image)."""
    sc = hdr.get('OBSRVTRY', 'STEREO_A').strip().upper()
    sumxy = hdr.get('CDELT1', 14.7) / 14.7  
    xc_pix = hdr.get('CRPIX1', im.shape[1] / 2.0 + 0.5) - 1.0
    yc_pix = hdr.get('CRPIX2', im.shape[0] / 2.0 + 0.5) - 1.0

    # Coefficients for Distortion Model (A vs B)
    cf = [1.96029e-05, -0.0201616, 4.68841] if 'B' in sc else [1.04872e-05, -0.00801293, -0.243670]

    y_idx, x_idx = np.indices(im.shape)
    dx = x_idx - xc_pix; dy = y_idx - yc_pix
    r_target = np.sqrt(dx**2 + dy**2)
    r_full = r_target * sumxy 
    r_obs_full = r_full + (cf[2] + cf[1]*r_full + cf[0]*(r_full**2))
    r_obs = r_obs_full / sumxy 
    
    ratio = np.divide(r_obs, r_target, out=np.ones_like(r_target, dtype=float), where=r_target!=0)
    x_obs = xc_pix + dx * ratio; y_obs = yc_pix + dy * ratio
    
    # Re-sample the image using the calculated observation coordinates
    warped_im = map_coordinates(im, [y_obs, x_obs], order=1, mode='constant', cval=np.nan, prefilter=False)
    hdr['DISTCORR'] = 'T'
    return warped_im, hdr

def cor2_prep_level1(data, header, cal_dir, bkg_base_dir, silent=False):
    """Performs Level 1 preparation of COR2 data."""
    im = data.copy().astype(np.float32)
    hdr = header.copy()
    obs = hdr.get('OBSRVTRY', 'STEREO_A').strip().upper()
    obs_char = 'A' if 'A' in obs else 'B'

    # 1. Basic Corrections (Trimming, Exposure, SEBIP, Bias)
    im, hdr = scc.scc_img_trim(im, hdr, silent=silent)
    exptime = hdr.get('EXPTIME', 1.0)
    if exptime > 0: im = im / exptime 

    im, hdr = cor2_sebip(im, hdr, silent=silent)
    biasmean = scc.get_biasmean(hdr, silent=silent)
    im = im - (biasmean / exptime if exptime > 0 else biasmean)

    # 2. Background Subtraction (Includes auto-matching logic for nearest date)
    pol = float(hdr.get('POLAR', 0.0))
    date_obs_val = hdr.get('DATE-OBS', '')[:10].replace('-', '') # e.g., '20240510'
    yyyymm = date_obs_val[:6]   # '202405'
    yymmdd = date_obs_val[2:8] # '240510'
    
    bkg_dir = os.path.join(bkg_base_dir, "monthly_min", obs_char.lower(), yyyymm)
    
    # Get all background files in the monthly folder
    # e.g., mc2A_p000_240506.fts, mc2A_p000_240513.fts ...
    bkg_pattern = os.path.join(bkg_dir, f"mc2{obs_char}_p{int(pol):03}_*.fts")
    available_bkgs = sorted(glob.glob(bkg_pattern))

    bkg_file = None
    if available_bkgs:
        # Select the latest file that is older than or equal to the target date
        valid_bkgs = [f for f in available_bkgs if os.path.basename(f).split('_')[-1][:6] <= yymmdd]
        bkg_file = valid_bkgs[-1] if valid_bkgs else available_bkgs[0]

    if bkg_file and os.path.exists(bkg_file):
        with fits.open(bkg_file) as bkg_hdul:
            bkg_data = bkg_hdul[0].data.astype(np.float32)
            
            # Resolve resolution mismatch (e.g., 2048 <-> 256)
            if im.shape != bkg_data.shape:
                if not silent:
                    print(f"🔄 Resizing BKG {bkg_data.shape} to match IM {im.shape}")
                bkg_data = rebin(bkg_data, im.shape)
            
            im = im - bkg_data
            hdr['HISTORY'] = f'P-Image CMAD BKG Subtracted: {os.path.basename(bkg_file)}'
    else:
        if not silent:
            print(f"⚠️ Background file not found (Target Date: {yymmdd})")
    
    # --- Redundant path check (Re-check specific date) ---
    bkg_file_spec = os.path.join(bkg_dir, f"mc2{obs_char}_p{int(pol):03}_{yymmdd}.fts")
    if os.path.exists(bkg_file_spec):
        with fits.open(bkg_file_spec) as bkg_hdul:
            bkg_data = bkg_hdul[0].data.astype(np.float32)
            
            # Handling resolution matching for Science vs Beacon modes
            if im.shape[0] > bkg_data.shape[0]: # Science(2048) -> BKG(1024)
                im = rebin(im, bkg_data.shape)
                # Update header for resolution changes
                hdr['NAXIS1'] = hdr['NAXIS2'] = 1024
                hdr['CDELT1'] = hdr['CDELT2'] = hdr.get('CDELT1', 14.7) * 2.0
                hdr['CRPIX1'] = (hdr.get('CRPIX1', 1024.5) - 0.5) / 2.0 + 0.5
                hdr['CRPIX2'] = (hdr.get('CRPIX2', 1024.5) - 0.5) / 2.0 + 0.5
            elif im.shape[0] < bkg_data.shape[0]: # Beacon(256) <- BKG(1024)
                bkg_data = rebin(bkg_data, im.shape)
            
            # Direct subtraction
            im = im - bkg_data
            hdr['HISTORY'] = f'P-Image CMAD BKG Subtracted: {os.path.basename(bkg_file_spec)}'

    # 3. Unit Conversion (Update BUNIT to 'MSB')
    calfac, hdr = scc.get_calfac(hdr, conv='MSB', silent=silent)
    im = im * calfac
    hdr['BUNIT'] = 'MSB'
    hdr['DATATYPE'] = 'Total Brightness (tB)'

    # 4. Vignetting Correction (Using Rebin)
    v_fname = '20060929_vignet_vCc2A.fts' if 'A' in obs else '20140723_vignet_vCc2B.fts'
    v_path = os.path.join(cal_dir, v_fname)
    if os.path.exists(v_path):
        with fits.open(v_path) as hdul:
            v_data = hdul[0].data[0:2048, 50:2098].astype(np.float32)
            # Adjust rotation for specific date range (Instrumental rolls)
            if '2015-05-19' <= hdr.get('DATE-OBS', '') < '2023-08-12': v_data = np.rot90(v_data, 2)
            im = im / np.clip(rebin(v_data, im.shape), 0.05, None)

    # 5. Distortion Correction
    im, hdr = cor2_warp(im, hdr, silent=silent)

    # 6. Apply Occulter Mask
    m_fname = 'cor2A_mask.fts' if 'A' in obs else 'cor2B_mask.fts'
    m_path = os.path.join(cal_dir, m_fname)
    if os.path.exists(m_path):
        with fits.open(m_path) as hdul:
            mask_rebin = rebin(hdul[0].data.astype(np.float32), im.shape)
            im[mask_rebin < 0.99] = np.nan # Keep only clean pixels (99% threshold)

    if np.any(np.isfinite(im) & (im > 0)):
        hdr = scc.scc_update_hdr(im, hdr, silent=silent)

    return im, hdr

if __name__ == "__main__":
    # Execution logic remains the same
    pass