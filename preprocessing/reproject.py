"""WCS North-Up solar-centered reprojection and geometry utilities."""

from __future__ import annotations

import copy
from typing import List, Tuple
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp


def derive_target_wcs(fits_headers: List[fits.Header], target_shape: Tuple[int, int] = (256, 256)) -> WCS:
    """Derives a fixed North-Up target WCS grid from sequence median solar center."""
    centers, scales = [], []
    first_wcs = None
    for hdr in fits_headers:
        w = WCS(hdr).celestial
        if first_wcs is None:
            first_wcs = w
        centers.append(w.world_to_pixel_values(0.0, 0.0))
        scales.append(np.abs(w.wcs.cdelt))

    median_center = np.median(centers, axis=0)
    median_scale = np.median(scales, axis=0)

    target = WCS(naxis=2)
    target.wcs.ctype = list(first_wcs.wcs.ctype)
    target.wcs.cunit = list(first_wcs.wcs.cunit)
    target.wcs.crval = [0.0, 0.0]
    target.wcs.crpix = [float(median_center[0]) + 1.0, float(median_center[1]) + 1.0]
    target.wcs.cdelt = median_scale
    target.wcs.pc = np.eye(2)
    target.wcs.set()
    return target


def reproject_to_target(
    data: np.ndarray, source_wcs: WCS, target_wcs: WCS, target_shape: Tuple[int, int] = (256, 256), order: str = "bilinear"
) -> Tuple[np.ndarray, np.ndarray]:
    """Reprojects image array into target North-Up WCS grid."""
    frame_target = copy.deepcopy(target_wcs)
    for name in ("hgln_obs", "hglt_obs", "dsun_obs", "rsun_ref"):
        val = getattr(source_wcs.wcs.aux, name, None)
        if val is not None:
            setattr(frame_target.wcs.aux, name, val)
    for name in ("dateobs", "dateavg", "datebeg", "dateend", "mjdobs", "mjdavg", "mjdbeg", "mjdend"):
        try:
            val = getattr(source_wcs.wcs, name)
            if val not in (None, ""):
                setattr(frame_target.wcs, name, val)
        except (AttributeError, TypeError, ValueError):
            pass
    frame_target.wcs.set()

    reproj_data, footprint = reproject_interp(
        (np.asarray(data), source_wcs), frame_target, shape_out=target_shape, order=order, return_footprint=True
    )
    return reproj_data, footprint


def calculate_geometry(wcs: WCS, shape: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Computes solar position angle (PA 0-360) and radius rasters from WCS."""
    yy, xx = np.indices(shape, dtype=float)
    world = wcs.pixel_to_world_values(xx, yy)
    lon = (np.asarray(world[0], float) + 180.0) % 360.0 - 180.0
    lat = np.asarray(world[1], float)
    pa = np.degrees(np.arctan2(-lon, lat)) % 360.0
    sx, sy = wcs.world_to_pixel_values(0.0, 0.0)
    radius = np.hypot(xx - float(sx), yy - float(sy))
    return pa, radius, (float(sx), float(sy))

