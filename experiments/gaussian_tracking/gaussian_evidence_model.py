"""Spatial Gaussian Evidence Model for Standalone CME Tracking.

Implements:
1. Nonnegative pixel evidence calculation:
   - Positive standardized: E_t = max(0, z_t)
   - Compressed positive: E_t = log(1 + max(0, z_t))
2. Validity-mask-aware 2D spatial Gaussian convolution:
   - Normalizes by valid kernel weight sum (no border dimming, no occulter leakage)
3. Spatially coherent evidence region extraction on the 256x256 grid:
   - Replaces D4/T24 64x64 connected components with continuous pixel evidence regions
   - Computes evidence-weighted centroids, polar coordinates, radial fronts, and PA extents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import gaussian_filter, label, find_objects
from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg, minimal_circular_enclosing_arc


@dataclass
class PixelEvidenceConfig:
    """Configuration for pixel-level evidence and spatial Gaussian spreading."""
    formulation: str = "positive"  # 'positive' or 'compressed'
    sigma_s: float = 2.0  # Gaussian spreading scale in pixels (0.0 = raw pixel evidence)
    clip_z: Optional[float] = None  # Optional z-score ceiling
    r_min_px: float = 22.0  # Inner coronagraph occulter radius in pixels
    r_max_px: float = 122.5  # Outer coronagraph FOV boundary in pixels
    evidence_threshold: float = 0.5  # Minimum spread evidence to form a coherent region
    min_region_area_px: int = 8  # Minimum pixel count for a candidate region


@dataclass
class EvidenceRegion:
    """Spatially coherent high-evidence region extracted from spread evidence map."""
    region_id: int
    mask: np.ndarray  # (256, 256) boolean mask
    area_px: int
    total_evidence: float
    mean_evidence: float
    peak_evidence: float
    peak_x: float
    peak_y: float
    centroid_x: float
    centroid_y: float
    centroid_r: float
    centroid_pa: float
    r_front: float
    r_min: float
    pa_span: float
    pa_min: float
    pa_max: float
    bounding_box: Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)


def compute_pixel_evidence(
    z_map: np.ndarray,
    valid_mask: np.ndarray,
    formulation: str = "positive",
    clip_z: Optional[float] = None,
    r_min_px: float = 22.0,
    r_max_px: float = 122.5,
    radius_raster: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes nonnegative pixel-level detection evidence E_t(x, y).

    Args:
        z_map: Standardized z-score map (256, 256).
        valid_mask: Observation valid mask (256, 256).
        formulation: 'positive' (E = max(0, z)) or 'compressed' (E = log(1 + max(0, z))).
        clip_z: Optional ceiling for z-score clipping.
        r_min_px: Inner occulter radius in pixels.
        r_max_px: Outer FOV radius in pixels.
        radius_raster: Polar radial coordinate map (256, 256).

    Returns:
        ev: Nonnegative pixel evidence map (256, 256), zero outside valid coronagraph mask.
    """
    ev = np.zeros_like(z_map, dtype=np.float32)

    annular_mask = valid_mask.copy()
    if radius_raster is not None:
        annular_mask &= (radius_raster >= r_min_px) & (radius_raster <= r_max_px)

    valid_idx = np.where(annular_mask)
    if len(valid_idx[0]) == 0:
        return ev

    z_vals = z_map[valid_idx].astype(np.float64)
    if clip_z is not None:
        z_vals = np.minimum(z_vals, float(clip_z))

    z_pos = np.maximum(0.0, z_vals)

    if formulation == "positive":
        ev_vals = z_pos
    elif formulation == "compressed":
        ev_vals = np.log1p(z_pos)
    else:
        raise ValueError(f"Unknown evidence formulation: {formulation}. Choose 'positive' or 'compressed'.")

    ev[valid_idx] = ev_vals.astype(np.float32)
    return ev


def apply_spatial_gaussian_spreading(
    evidence: np.ndarray,
    valid_mask: np.ndarray,
    sigma_s: float,
    r_min_px: float = 22.0,
    r_max_px: float = 122.5,
    radius_raster: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Distributes evidence spatially using validity-mask-aware 2D Gaussian convolution.

    Uses normalized convolution:
        E_spread = (G * (E * M)) / (G * M)
    Strictly ensuring:
    1. Invalid/occulted pixels do NOT bleed artificial evidence into valid regions.
    2. Boundary pixels are not artificially dimmed.
    3. Invalid pixels are strictly zero.

    Args:
        evidence: Nonnegative pixel evidence (256, 256).
        valid_mask: Observation valid mask (256, 256).
        sigma_s: Gaussian spreading scale in pixels. If sigma_s <= 0, returns raw evidence.
        r_min_px: Inner occulter radius.
        r_max_px: Outer FOV boundary.
        radius_raster: Optional polar radius raster.

    Returns:
        spread_ev: Spatially spread evidence map (256, 256).
    """
    if sigma_s <= 0.0:
        # G0: Raw pixel evidence (no spatial spreading)
        return evidence.copy()

    annular_mask = valid_mask.copy()
    if radius_raster is not None:
        annular_mask &= (radius_raster >= r_min_px) & (radius_raster <= r_max_px)

    m_float = annular_mask.astype(np.float32)
    masked_ev = np.where(annular_mask, evidence, 0.0).astype(np.float32)

    # Numerator: convolution of masked evidence
    num = gaussian_filter(masked_ev, sigma=sigma_s, mode="constant", cval=0.0)
    # Denominator: convolution of mask (effective kernel weight sum over valid pixels)
    den = gaussian_filter(m_float, sigma=sigma_s, mode="constant", cval=0.0)

    spread_ev = np.zeros_like(evidence, dtype=np.float32)
    valid_den = annular_mask & (den > 1e-6)
    spread_ev[valid_den] = num[valid_den] / den[valid_den]

    return spread_ev


def extract_evidence_regions(
    spread_evidence: np.ndarray,
    valid_mask: np.ndarray,
    sx: float,
    sy: float,
    radius_raster: np.ndarray,
    pa_raster: np.ndarray,
    threshold: float = 0.5,
    min_area_px: int = 8,
    r_min_px: float = 22.0,
    r_max_px: float = 122.5,
) -> List[EvidenceRegion]:
    """Extracts spatially coherent high-evidence regions directly on 256x256 grid.

    Replaces patch-based D4 connected components with continuous pixel evidence regions.
    """
    annular_mask = valid_mask & (radius_raster >= r_min_px) & (radius_raster <= r_max_px)
    binary_active = annular_mask & (spread_evidence >= threshold)

    struct_8 = np.ones((3, 3), dtype=bool)
    labeled_img, n_features = label(binary_active, structure=struct_8)

    regions: List[EvidenceRegion] = []
    if n_features == 0:
        return regions

    slices = find_objects(labeled_img)
    yy, xx = np.indices(spread_evidence.shape)

    for feat_id in range(1, n_features + 1):
        sl = slices[feat_id - 1]
        if sl is None:
            continue

        region_submask = (labeled_img[sl] == feat_id)
        area = int(region_submask.sum())
        if area < min_area_px:
            continue

        # Global mask for region
        reg_mask = (labeled_img == feat_id)
        ev_vals = spread_evidence[reg_mask]
        tot_ev = float(np.sum(ev_vals))
        mean_ev = float(np.mean(ev_vals))
        pk_ev = float(np.max(ev_vals))

        # Peak location
        sub_y, sub_x = np.where(reg_mask)
        pk_idx = np.argmax(spread_evidence[sub_y, sub_x])
        pk_y = float(sub_y[pk_idx])
        pk_x = float(sub_x[pk_idx])

        # Evidence-weighted centroid
        ev_weights = ev_vals / tot_ev if tot_ev > 0 else np.full_like(ev_vals, 1.0 / area)
        cy = float(np.sum(sub_y * ev_weights))
        cx = float(np.sum(sub_x * ev_weights))

        # Centroid polar coordinates
        dx = cx - sx
        dy = cy - sy
        c_r = float(np.sqrt(dx**2 + dy**2))
        # COR2 standard north-up PA: PA = arctan2(-dx, dy) in degrees [0, 360)
        c_pa = float(np.degrees(np.arctan2(-dx, dy)) % 360.0)

        # Region radial and angular extent
        reg_radii = radius_raster[reg_mask]
        reg_pas = pa_raster[reg_mask]

        r_front = float(np.max(reg_radii))
        r_min = float(np.min(reg_radii))

        arc = minimal_circular_enclosing_arc(reg_pas)
        pa_span = float(arc["arc_span"])
        pa_min = float(arc["arc_start"])
        pa_max = float(arc["arc_end"])

        ymin = int(sl[0].start)
        ymax = int(sl[0].stop)
        xmin = int(sl[1].start)
        xmax = int(sl[1].stop)

        regions.append(EvidenceRegion(
            region_id=feat_id,
            mask=reg_mask,
            area_px=area,
            total_evidence=tot_ev,
            mean_evidence=mean_ev,
            peak_evidence=pk_ev,
            peak_x=pk_x,
            peak_y=pk_y,
            centroid_x=cx,
            centroid_y=cy,
            centroid_r=c_r,
            centroid_pa=c_pa,
            r_front=r_front,
            r_min=r_min,
            pa_span=pa_span,
            pa_min=pa_min,
            pa_max=pa_max,
            bounding_box=(ymin, xmin, ymax, xmax),
        ))

    # Sort regions descending by total evidence
    regions.sort(key=lambda r: r.total_evidence, reverse=True)
    return regions


def compute_gaussian_evidence_overlap(
    spread_ev_1: np.ndarray,
    mask_1: np.ndarray,
    spread_ev_2: np.ndarray,
    mask_2: np.ndarray,
) -> float:
    """Computes continuous Gaussian evidence overlap between two regions:

        S_overlap(1, 2) = sum min(E1, E2) / sum max(E1, E2)

    Evaluated over the union of their masks.
    """
    union_m = mask_1 | mask_2
    if not np.any(union_m):
        return 0.0

    e1 = np.where(mask_1, spread_ev_1, 0.0)[union_m]
    e2 = np.where(mask_2, spread_ev_2, 0.0)[union_m]

    intersection = np.sum(np.minimum(e1, e2))
    union = np.sum(np.maximum(e1, e2))

    if union <= 0.0:
        return 0.0
    return float(intersection / union)
