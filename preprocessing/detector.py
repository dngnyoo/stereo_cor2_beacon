"""Causal rolling baseline and multi-scale patch activity detector."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter


def compute_frame_zscore(
    rd: np.ndarray, valid_mask: np.ndarray, gaussian_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Applies edge-preserving normalized Gaussian smoothing and Z-score standardization."""
    val = valid_mask.astype(bool)
    num = gaussian_filter(np.where(val, rd, 0.0), gaussian_sigma, mode="constant", cval=0.0)
    den = gaussian_filter(val.astype(float), gaussian_sigma, mode="constant", cval=0.0)
    usable = val & (den > 0)

    smoothed = np.full(rd.shape, np.nan, dtype=np.float32)
    smoothed[usable] = num[usable] / den[usable]

    vals = smoothed[usable & np.isfinite(smoothed)]
    if len(vals) < 2:
        return np.zeros_like(rd), usable, 0.0, 1.0

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    if not np.isfinite(std_val) or std_val <= 0:
        std_val = 1.0

    z_global = np.zeros_like(rd, dtype=np.float32)
    z_global[usable] = (smoothed[usable] - mean_val) / std_val
    return z_global, usable, mean_val, std_val


def causal_rolling_baseline(
    framewise_p95_history: List[float], k_mad: float = 3.0, mad_scale: float = 1.4826
) -> float:
    """Computes causal robust threshold from past 24-hour P95 history: median + k * 1.4826 * MAD."""
    hist = np.asarray(framewise_p95_history, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) < 3:
        return 3.0  # Safe default threshold

    med = float(np.median(hist))
    mad = float(np.median(np.abs(hist - med)))
    return med + k_mad * mad_scale * mad


def compute_patch_activities(
    z_image: np.ndarray, usable_mask: np.ndarray, patch_size: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes block-wise P95(|Z|) activity grid and fraction of usable pixels."""
    h, w = z_image.shape
    gh, gw = h // patch_size, w // patch_size

    # Reshape into blocks
    blocks_z = np.abs(z_image).reshape(gh, patch_size, gw, patch_size).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    blocks_u = usable_mask.reshape(gh, patch_size, gw, patch_size).transpose(0, 2, 1, 3).reshape(gh, gw, -1)

    frac_usable = np.mean(blocks_u, axis=2)
    p95_grid = np.zeros((gh, gw), dtype=np.float32)

    for r in range(gh):
        for c in range(gw):
            u = blocks_u[r, c]
            if np.any(u):
                p95_grid[r, c] = np.percentile(blocks_z[r, c][u], 95.0)

    return p95_grid, frac_usable

