"""Physical calibration helpers for STEREO COR2 FITS data."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def get_biasmean(header: fits.Header, silent: bool = True) -> float:
    """Calculates mean bias value to be subtracted from image."""
    bias = float(header.get("BIASMEAN", 0.0))
    ipsum = int(header.get("IPSUM", 0))
    ip_raw = str(header.get("IP_00_19", ""))

    if ("103" in ip_raw) or (" 37" in ip_raw) or (" 38" in ip_raw):
        if not silent:
            print("Biasmean subtracted onboard in seb ip.")
        return 0.0

    detector = str(header.get("DETECTOR", ""))
    if detector.startswith("HI"):
        n_images = float(header.get("N_IMAGES", 1.0))
        bias = bias - (bias / max(1.0, n_images))

    if float(header.get("OFFSETCR", 0.0)) > 0:
        return 0.0

    if ipsum > 1:
        bias = bias * ((2 ** (ipsum - 1)) ** 2)

    return bias


def cor2_sebip(data: np.ndarray, header: fits.Header, silent: bool = True) -> Tuple[np.ndarray, fits.Header]:
    """Restores onboard compressed SEBIP bit-shift data."""
    d = data.copy()
    ip_raw = str(header.get("IP_00_19", ""))
    if not ip_raw.strip():
        return d, header

    while len(ip_raw) < 60:
        ip_raw = " " + ip_raw
    ip = [ip_raw[i : i + 3] for i in range(0, 60, 3)]

    if "117" in ip:
        ind = ip.index("117")
        ip = ip[ind:]

    if ip.count("  1") > 0:
        d *= (2.0 ** ip.count("  1"))
    if ip.count(" 16") + ip.count(" 17") > 0:
        d *= (64.0 ** (ip.count(" 16") + ip.count(" 17")))
    if ip.count(" 53") > 0 and float(header.get("IPSUM", 0)) > 0:
        d *= (4.0 ** ip.count(" 53"))

    return d, header


def calibrate_raw_fits(fits_path: Path | str) -> Tuple[np.ndarray, fits.Header, float, float]:
    """Reads raw FITS, normalizes by EXPTIME, applies SEBIP and bias correction."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header.copy()

    exptime = float(hdr.get("EXPTIME", 0.0))
    if not np.isfinite(exptime) or exptime <= 0:
        raise ValueError(f"Invalid EXPTIME={exptime} in {fits_path}")

    # 1. Normalize by exposure time
    data /= exptime

    # 2. SEBIP decompression
    data, hdr = cor2_sebip(data, hdr, silent=True)

    # 3. Bias subtraction
    bias = get_biasmean(hdr, silent=True)
    data -= (bias / exptime)

    return data, hdr, exptime, bias


def load_occulter_mask(mask_fits_path: Path | str, target_wcs: WCS, shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Loads cor2_mask.fts and projects to target North-Up WCS grid."""
    from reproject import reproject_interp
    with fits.open(mask_fits_path) as hdul:
        mask_raw = hdul[0].data.astype(np.float32)
        mask_wcs = WCS(hdul[0].header).celestial

    reproj_mask, _ = reproject_interp(
        (mask_raw, mask_wcs), target_wcs, shape_out=shape, order="nearest"
    )
    # Mask is 1 for valid pixels, 0 for occulted
    valid_mask = np.isfinite(reproj_mask) & (reproj_mask > 0.5)
    return valid_mask

