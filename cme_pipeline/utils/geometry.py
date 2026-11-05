"""Geometry and coordinate utilities for STEREO COR2 CME processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def circular_diff_deg(
    a: Union[np.ndarray, float], b: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    """Signed shortest angular difference a - b on S^1 in range [-180, 180)."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def circular_mean_deg(
    angles: Union[np.ndarray, List[float]],
    weights: Optional[Union[np.ndarray, List[float]]] = None,
) -> float:
    """Computes exact circular mean direction on S^1 in range [0, 360)."""
    angles_arr = np.asarray(angles, dtype=float)
    if len(angles_arr) == 0:
        return np.nan
    rads = np.radians(angles_arr % 360.0)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        sin_sum = float(np.sum(w * np.sin(rads)))
        cos_sum = float(np.sum(w * np.cos(rads)))
    else:
        sin_sum = float(np.sum(np.sin(rads)))
        cos_sum = float(np.sum(np.cos(rads)))

    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return float(angles_arr[0] % 360.0)

    mean_rad = np.arctan2(sin_sum, cos_sum)
    mean_deg = float(np.degrees(mean_rad) % 360.0)
    if abs(mean_deg - 360.0) < 1e-9:
        mean_deg = 0.0
    return mean_deg


def minimal_circular_enclosing_arc(
    angles_deg: np.ndarray, margin_deg: float = 0.0
) -> Dict[str, float]:
    """Finds minimal enclosing circular arc on S^1.
    
    Returns:
        dict with arc_start, arc_end, arc_span, arc_center
    """
    if len(angles_deg) == 0:
        return {
            "arc_start": np.nan,
            "arc_end": np.nan,
            "arc_span": 0.0,
            "arc_center": np.nan,
        }

    unique_deg = np.unique(angles_deg % 360.0)
    if len(unique_deg) == 1:
        val = float(unique_deg[0])
        return {
            "arc_start": (val - margin_deg) % 360.0,
            "arc_end": (val + margin_deg) % 360.0,
            "arc_span": float(2.0 * margin_deg),
            "arc_center": val,
        }

    sorted_deg = np.sort(unique_deg)
    diffs = np.diff(sorted_deg)
    wrap_diff = (sorted_deg[0] + 360.0) - sorted_deg[-1]
    all_diffs = np.append(diffs, wrap_diff)

    max_gap_idx = int(np.argmax(all_diffs))
    max_gap = float(all_diffs[max_gap_idx])
    arc_span = 360.0 - max_gap

    if max_gap_idx == len(diffs):
        arc_start = float(sorted_deg[0])
        arc_end = float(sorted_deg[-1])
    else:
        arc_start = float(sorted_deg[max_gap_idx + 1])
        arc_end = float(sorted_deg[max_gap_idx])

    if margin_deg > 0.0:
        arc_start = (arc_start - margin_deg) % 360.0
        arc_end = (arc_end + margin_deg) % 360.0
        arc_span = min(360.0, arc_span + 2.0 * margin_deg)

    arc_center = (arc_start + arc_span / 2.0) % 360.0
    return {
        "arc_start": arc_start,
        "arc_end": arc_end,
        "arc_span": arc_span,
        "arc_center": arc_center,
    }


def circular_sector_overlap(
    c1: Optional[float],
    w1: float,
    c2: Optional[float],
    w2: float,
    n_bins: int = 36000,
    c1_is_interval: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Computes exact circular angular IoU, coverage, and CPA difference on S^1."""
    grid_deg = np.linspace(0.0, 360.0, n_bins, endpoint=False)

    if c1_is_interval is not None:
        th_start, th_end = c1_is_interval[0] % 360.0, c1_is_interval[1] % 360.0
        if th_start <= th_end:
            m1 = (grid_deg >= th_start) & (grid_deg <= th_end)
        else:
            m1 = (grid_deg >= th_start) | (grid_deg <= th_end)
    else:
        if c1 is None:
            m1 = np.zeros_like(grid_deg, dtype=bool)
        else:
            m1 = np.abs(circular_diff_deg(grid_deg, c1)) <= (w1 / 2.0)

    if c2 is None:
        m2 = np.zeros_like(grid_deg, dtype=bool)
    else:
        m2 = np.abs(circular_diff_deg(grid_deg, c2)) <= (w2 / 2.0)

    inter_cnt = int(np.count_nonzero(m1 & m2))
    union_cnt = int(np.count_nonzero(m1 | m2))
    s1_cnt = int(np.count_nonzero(m1))
    s2_cnt = int(np.count_nonzero(m2))

    iou = float(inter_cnt / union_cnt) if union_cnt > 0 else 0.0
    cov1 = float(inter_cnt / s1_cnt) if s1_cnt > 0 else 0.0
    cov2 = float(inter_cnt / s2_cnt) if s2_cnt > 0 else 0.0
    step = 360.0 / n_bins
    cpa_err = (
        float(abs(circular_diff_deg(c1, c2)))
        if (c1 is not None and c2 is not None)
        else np.nan
    )

    return {
        "iou": iou,
        "cov_cat": cov1,
        "cov_cand": cov2,
        "inter_deg": inter_cnt * step,
        "union_deg": union_cnt * step,
        "cpa_err": cpa_err,
    }


def parse_iso_dt(s: Any) -> datetime:
    """Parse various datetime representations into UTC datetime."""
    if isinstance(s, datetime):
        return s
    s_str = str(s).strip()
    if s_str.endswith("Z"):
        s_str = s_str[:-1]
    if "T" in s_str:
        try:
            return datetime.fromisoformat(s_str)
        except ValueError:
            return datetime.strptime(s_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
    if "/" in s_str:
        if len(s_str.split()) == 2:
            date_part, time_part = s_str.split()
            if time_part.count(":") == 1:
                return datetime.strptime(s_str, "%Y/%m/%d %H:%M")
            return datetime.strptime(s_str, "%Y/%m/%d %H:%M:%S")
    return datetime.strptime(s_str, "%Y-%m-%d %H:%M:%S")


def compute_wcs_rasters(
    wcs: WCS,
    image_shape: Tuple[int, int] = (256, 256),
    center_xy: Tuple[float, float] = (126.618, 127.503),
    patch_grid_shape: Tuple[int, int] = (64, 64),
    patch_size: int = 4,
) -> Dict[str, np.ndarray]:
    """Generates standard pixel-level and patch-level coordinate and PA rasters."""
    sx, sy = center_xy
    yy, xx = np.indices(image_shape, dtype=float)
    world = wcs.pixel_to_world_values(xx, yy)
    lon = (np.asarray(world[0], float) + 180.0) % 360.0 - 180.0
    lat = np.asarray(world[1], float)
    pa_raster = np.degrees(np.arctan2(-lon, lat)) % 360.0
    radius_raster = np.hypot(xx - sx, yy - sy)

    gy, gx = patch_grid_shape
    py_idx, px_idx = np.indices((gy, gx))
    patch_x = px_idx * patch_size + (patch_size - 1) / 2.0
    patch_y = py_idx * patch_size + (patch_size - 1) / 2.0
    patch_rad = np.hypot(patch_x - sx, patch_y - sy)
    patch_world = wcs.pixel_to_world_values(patch_x, patch_y)
    patch_lon = (np.asarray(patch_world[0], float) + 180.0) % 360.0 - 180.0
    patch_lat = np.asarray(patch_world[1], float)
    patch_pa = np.degrees(np.arctan2(-patch_lon, patch_lat)) % 360.0

    return {
        "pa_raster": pa_raster,
        "radius_raster": radius_raster,
        "patch_pa": patch_pa,
        "patch_rad": patch_rad,
        "patch_x": patch_x,
        "patch_y": patch_y,
    }

