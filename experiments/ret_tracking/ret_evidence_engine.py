r"""Radial Evidence Engine for Standalone CME Identification and Front Detection.

Implements:
1. Nonnegative standardized pixel evidence:
   - Positive standardized: E_t = max(0, z_t)
   - Compressed positive: E_t = log(1 + max(0, z_t))
2. Validity-mask-normalized 2D Gaussian convolution:
   - E_tilde = (G_sigma * (E * M)) / (G_sigma * M)
3. Polar coordinate transformation:
   - E_tilde(r, theta) sampled using sub-pixel interpolation on (r, theta) grid
4. Radial ray accumulation:
   - A_t(theta) = \int_{r_min}^{r_max} E_tilde(r, theta) dr
   - Normalization by valid radial support length L_valid(theta)
   - Angular neighborhood smoothing (e.g., +/- 2 deg)
5. Circular angular peak & lobe detection:
   - Continuous peak detection on S^1
   - Lobe boundaries, angular support width, peak prominence
   - Representative CPA estimation: peak PA vs evidence-weighted circular mean
6. Radial peak and outermost supported front estimation:
   - r_peak: position of strongest evidence
   - r_front: outermost supported physical front with contiguous spatial support
   - Front profile as a function of PA: r_front(theta)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import find_peaks, peak_prominences, peak_widths

from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg


@dataclass
class PolarGridGeometry:
    """Precomputed coordinate grids for fast polar sampling."""
    sx: float
    sy: float
    r_min_px: float = 22.0
    r_max_px: float = 122.5
    n_r_bins: int = 101  # dr ~ 1.0 px
    n_pa_bins: int = 360  # d_theta = 1.0 deg
    r_bins: np.ndarray = field(init=False)
    pa_bins: np.ndarray = field(init=False)
    sampling_coords: np.ndarray = field(init=False)  # Shape (2, n_r, n_pa) for map_coordinates (y, x)

    def __post_init__(self):
        self.r_bins = np.linspace(self.r_min_px, self.r_max_px, self.n_r_bins, dtype=np.float32)
        self.pa_bins = np.linspace(0.0, 360.0, self.n_pa_bins, endpoint=False, dtype=np.float32)

        # 2D mesh of (r, pa)
        rr, ppa = np.meshgrid(self.r_bins, np.radians(self.pa_bins), indexing="ij")

        # In STEREO COR2 north-up:
        # PA measured from North (0 deg, +y) CCW through East (90 deg, -x), South (180 deg, -y), West (270 deg, +x)
        # x = sx - r * sin(PA)
        # y = sy + r * cos(PA)
        coords_x = self.sx - rr * np.sin(ppa)
        coords_y = self.sy + rr * np.cos(ppa)

        # map_coordinates requires (row, col) = (y, x)
        self.sampling_coords = np.array([coords_y, coords_x], dtype=np.float32)


@dataclass
class RadialCandidate:
    """Candidate CME propagation direction identified from radial evidence accumulation."""
    candidate_id: int
    peak_pa_deg: float
    weighted_cpa_deg: float
    pa_width_deg: float
    pa_start_deg: float
    pa_end_deg: float
    total_lobe_evidence: float
    peak_evidence: float
    prominence: float
    r_peak_px: float
    r_front_px: float
    r_min_px: float
    radial_profile: np.ndarray  # (n_r_bins,)
    front_profile: np.ndarray  # front radius per PA in lobe
    lobe_pa_indices: np.ndarray
    left_valley_pa_deg: Optional[float] = None
    right_valley_pa_deg: Optional[float] = None
    left_valley_dist_deg: Optional[float] = None
    right_valley_dist_deg: Optional[float] = None
    selected_half_width_deg: Optional[float] = None
    fallback_used: bool = False


@dataclass
class FrameRadialEvidenceResult:
    """Per-frame diagnostic data from radial evidence accumulation."""
    frame_idx: int
    raw_evidence: np.ndarray
    spread_evidence: np.ndarray
    polar_evidence: np.ndarray  # (n_r_bins, n_pa_bins)
    polar_valid: np.ndarray  # (n_r_bins, n_pa_bins) boolean
    angular_evidence_raw: np.ndarray  # (n_pa_bins,)
    angular_evidence_smoothed: np.ndarray  # (n_pa_bins,)
    valid_radial_support: np.ndarray  # (n_pa_bins,)
    candidates: List[RadialCandidate]


def compute_pixel_evidence(
    z_map: np.ndarray,
    valid_mask: np.ndarray,
    formulation: str = "positive",
    clip_z: Optional[float] = None,
) -> np.ndarray:
    """Computes nonnegative pixel detection evidence E_t(x, y)."""
    ev = np.zeros_like(z_map, dtype=np.float32)
    valid_idx = np.where(valid_mask)
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
        raise ValueError(f"Unknown formulation: {formulation}. Choose 'positive' or 'compressed'.")

    ev[valid_idx] = ev_vals.astype(np.float32)
    return ev


def apply_spatial_gaussian_spreading(
    evidence: np.ndarray,
    valid_mask: np.ndarray,
    sigma_s: float,
) -> np.ndarray:
    """Applies validity-mask-normalized 2D Gaussian convolution:

        E_tilde = (G_sigma * (E * M)) / (G_sigma * M)
    """
    if sigma_s <= 0.0:
        return evidence.copy()

    m_float = valid_mask.astype(np.float32)
    masked_ev = np.where(valid_mask, evidence, 0.0).astype(np.float32)

    num = gaussian_filter(masked_ev, sigma=sigma_s, mode="constant", cval=0.0)
    den = gaussian_filter(m_float, sigma=sigma_s, mode="constant", cval=0.0)

    spread_ev = np.zeros_like(evidence, dtype=np.float32)
    good = valid_mask & (den > 1e-6)
    spread_ev[good] = num[good] / den[good]
    return spread_ev


def sample_polar_evidence(
    spread_ev: np.ndarray,
    valid_mask: np.ndarray,
    grid: PolarGridGeometry,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms spread evidence and valid mask into polar coordinates (r, theta).

    Returns:
        polar_ev: (n_r_bins, n_pa_bins) float32
        polar_val: (n_r_bins, n_pa_bins) bool
    """
    polar_ev = map_coordinates(
        spread_ev, grid.sampling_coords, order=1, mode="constant", cval=0.0
    ).astype(np.float32)

    polar_val = map_coordinates(
        valid_mask.astype(np.float32), grid.sampling_coords, order=0, mode="constant", cval=0.0
    ) > 0.5

    # Zero out invalid pixels strictly
    polar_ev[~polar_val] = 0.0
    return polar_ev, polar_val


def accumulate_radial_evidence(
    polar_ev: np.ndarray,
    polar_val: np.ndarray,
    grid: PolarGridGeometry,
    angular_smooth_sigma_deg: float = 2.0,
    normalize_by_length: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates radial evidence accumulation A_t(theta).

    Args:
        polar_ev: Polar evidence map (n_r_bins, n_pa_bins).
        polar_val: Polar validity mask (n_r_bins, n_pa_bins).
        grid: Polar grid geometry.
        angular_smooth_sigma_deg: Gaussian smoothing sigma in degrees along PA.
        normalize_by_length: Whether to normalize by valid radial support length.

    Returns:
        a_raw: Raw radial accumulation (n_pa_bins,).
        a_smooth: Angularly smoothed radial accumulation (n_pa_bins,).
        valid_length: Valid radial support length per PA in pixels (n_pa_bins,).
    """
    dr = float(grid.r_bins[1] - grid.r_bins[0]) if len(grid.r_bins) > 1 else 1.0

    # Discrete radial integration: sum_r E(r, theta) * dr
    a_raw = np.sum(polar_ev, axis=0) * dr  # Shape (n_pa_bins,)
    valid_length = np.sum(polar_val, axis=0) * dr

    if normalize_by_length:
        denom = np.maximum(1.0, valid_length)
        a_acc = (a_raw / denom) * (grid.r_max_px - grid.r_min_px)
    else:
        a_acc = a_raw.copy()

    # Circular angular smoothing along PA
    if angular_smooth_sigma_deg > 0.0:
        # Wrap circular boundary for smoothing
        n_pad = int(math.ceil(4 * angular_smooth_sigma_deg))
        padded = np.concatenate([a_acc[-n_pad:], a_acc, a_acc[:n_pad]])
        smoothed = gaussian_filter(padded, sigma=angular_smooth_sigma_deg, mode="nearest")
        a_smooth = smoothed[n_pad:-n_pad]
    else:
        a_smooth = a_acc.copy()

    return a_raw, a_smooth.astype(np.float32), valid_length.astype(np.float32)


def detect_angular_candidates(
    a_smooth: np.ndarray,
    polar_ev: np.ndarray,
    grid: PolarGridGeometry,
    min_prominence: float = 3.0,
    min_separation_deg: float = 15.0,
    min_lobe_width_deg: float = 8.0,
    max_lobe_width_deg: float = 120.0,
    front_evidence_quantile: float = 0.20,
    front_min_threshold: float = 0.35,
    lobe_width_mode: str = "prominence",
) -> List[RadialCandidate]:
    """Identifies significant angular evidence lobes and extracts radial CME fronts.

    Supports:
    - 'prominence': Baseline prominence-based width with 1.5x expansion (capped at max_lobe_width_deg).
    - 'valley_symmetric': Symmetric nearest-side inter-peak valley-based width.
    """
    n_pa = len(a_smooth)
    # Replicate array across circular boundary for seamless peak detection
    pad_len = 60
    a_ext = np.concatenate([a_smooth[-pad_len:], a_smooth, a_smooth[:pad_len]])

    # Find local peaks on extended profile
    peaks, properties = find_peaks(
        a_ext,
        prominence=min_prominence,
        distance=int(round(min_separation_deg)),
    )

    if len(peaks) == 0:
        return []

    prominences = properties["prominences"]
    widths, width_heights, left_ips, right_ips = peak_widths(
        a_ext, peaks, rel_height=0.70
    )

    # Filter peaks that fall inside the primary [0, n_pa) range
    valid_peaks = []
    seen_primary_indices = set()

    for idx, pk in enumerate(peaks):
        orig_pa_idx = (pk - pad_len) % n_pa
        # Avoid duplicate peak from padding
        if orig_pa_idx in seen_primary_indices:
            continue
        # Check if peak is inside primary range in a_ext
        if pad_len <= pk < pad_len + n_pa:
            seen_primary_indices.add(orig_pa_idx)
            valid_peaks.append((
                pk, orig_pa_idx, prominences[idx], widths[idx],
                left_ips[idx], right_ips[idx]
            ))

    if not valid_peaks:
        return []

    # Precompute valley-based lobe widths if requested
    valley_info: Dict[int, Dict[str, Any]] = {}
    if lobe_width_mode == "valley_symmetric":
        peaks_by_angle = sorted(valid_peaks, key=lambda item: item[1])
        K = len(peaks_by_angle)
        for k, (pk_ext, orig_pa_idx, prom, width_deg, left_ip, right_ip) in enumerate(peaks_by_angle):
            theta_p = float(grid.pa_bins[orig_pa_idx])
            if K < 2:
                # Step 6: Single-peak fallback to prominence
                half_w_fallback = max(min_lobe_width_deg / 2.0, min(max_lobe_width_deg / 2.0, width_deg * 0.75))
                valley_info[orig_pa_idx] = {
                    "h": float(half_w_fallback),
                    "w": float(half_w_fallback * 2.0),
                    "theta_L": None,
                    "theta_R": None,
                    "d_L": None,
                    "d_R": None,
                    "fallback": True,
                    "valid": True,
                }
                continue

            prev_orig_idx = peaks_by_angle[(k - 1) % K][1]
            next_orig_idx = peaks_by_angle[(k + 1) % K][1]

            # Step 2 — Find Left and Right Valleys
            n_left = (orig_pa_idx - prev_orig_idx) % n_pa
            n_right = (next_orig_idx - orig_pa_idx) % n_pa

            if n_left <= 1 or n_right <= 1:
                # Missing valley interval on one side -> fallback
                half_w_fallback = max(min_lobe_width_deg / 2.0, min(max_lobe_width_deg / 2.0, width_deg * 0.75))
                valley_info[orig_pa_idx] = {
                    "h": float(half_w_fallback),
                    "w": float(half_w_fallback * 2.0),
                    "theta_L": None,
                    "theta_R": None,
                    "d_L": None,
                    "d_R": None,
                    "fallback": True,
                    "valid": True,
                }
                continue

            # Left interval strictly between adjacent peaks
            indices_left = [(prev_orig_idx + s) % n_pa for s in range(1, n_left)]
            vals_left = a_smooth[indices_left]
            min_left_val = float(np.min(vals_left))
            min_left_cands = [i for i in indices_left if a_smooth[i] == min_left_val]
            # Tie breaking: pick position closest to current peak
            v_left_idx = min(min_left_cands, key=lambda i: (orig_pa_idx - i) % n_pa)
            theta_L = float(grid.pa_bins[v_left_idx])
            d_L = float((theta_p - theta_L) % 360.0)

            # Right interval strictly between adjacent peaks
            indices_right = [(orig_pa_idx + s) % n_pa for s in range(1, n_right)]
            vals_right = a_smooth[indices_right]
            min_right_val = float(np.min(vals_right))
            min_right_cands = [i for i in indices_right if a_smooth[i] == min_right_val]
            # Tie breaking: pick position closest to current peak
            v_right_idx = min(min_right_cands, key=lambda i: (i - orig_pa_idx) % n_pa)
            theta_R = float(grid.pa_bins[v_right_idx])
            d_R = float((theta_R - theta_p) % 360.0)

            # Step 4 — Select the Smaller Distance
            h = min(d_L, d_R)
            W = 2.0 * h

            # Step 6 — Edge case: Narrow-lobe rejection
            if W < min_lobe_width_deg:
                valley_info[orig_pa_idx] = {"valid": False}
                continue

            valley_info[orig_pa_idx] = {
                "h": float(h),
                "w": float(W),
                "theta_L": theta_L,
                "theta_R": theta_R,
                "d_L": d_L,
                "d_R": d_R,
                "fallback": False,
                "valid": True,
            }

    # Sort descending by prominence (consistent ranking)
    valid_peaks.sort(key=lambda item: item[2], reverse=True)

    candidates: List[RadialCandidate] = []
    cand_id = 1

    for pk_ext, pk_orig, prom, width_deg, left_ip, right_ip in valid_peaks:
        theta_p = float(grid.pa_bins[pk_orig])

        if lobe_width_mode == "valley_symmetric":
            v_data = valley_info.get(pk_orig)
            if not v_data or not v_data["valid"]:
                continue
            half_w = v_data["h"]
            lobe_width = v_data["w"]
            theta_L = v_data["theta_L"]
            theta_R = v_data["theta_R"]
            d_L = v_data["d_L"]
            d_R = v_data["d_R"]
            fallback_used = v_data["fallback"]
        else:
            if width_deg < min_lobe_width_deg:
                continue
            half_w = max(min_lobe_width_deg / 2.0, min(max_lobe_width_deg / 2.0, width_deg * 0.75))
            lobe_width = float(half_w * 2.0)
            theta_L = None
            theta_R = None
            d_L = None
            d_R = None
            fallback_used = False

        # Determine symmetric lobe boundaries around original peak PA
        lobe_start_deg = (theta_p - half_w) % 360.0
        lobe_end_deg = (theta_p + half_w) % 360.0

        # Collect PA indices in this lobe
        d_pa_all = circular_diff_deg(grid.pa_bins, theta_p)
        lobe_mask = np.abs(d_pa_all) <= half_w
        lobe_indices = np.where(lobe_mask)[0]

        if len(lobe_indices) == 0:
            continue

        lobe_evidence = float(np.sum(a_smooth[lobe_indices]))
        peak_ev = float(a_smooth[pk_orig])

        # Evidence-weighted circular mean CPA:
        # CPA = atan2(sum A*sin(theta), sum A*cos(theta))
        theta_rad = np.radians(grid.pa_bins[lobe_indices])
        weights = a_smooth[lobe_indices]
        sin_sum = np.sum(weights * np.sin(theta_rad))
        cos_sum = np.sum(weights * np.cos(theta_rad))
        weighted_cpa = float(np.degrees(np.arctan2(sin_sum, cos_sum)) % 360.0)

        # Extract radial evidence profile for this lobe
        lobe_polar = polar_ev[:, lobe_indices]  # Shape (n_r, n_lobe)
        # Lobe mean radial profile
        rad_profile = np.mean(lobe_polar, axis=1)  # Shape (n_r,)

        # Strongest evidence radius
        pk_r_idx = int(np.argmax(rad_profile))
        r_peak = float(grid.r_bins[pk_r_idx])

        # Outermost supported radial front
        # Determine adaptive threshold lambda
        max_rad_val = float(np.max(rad_profile))
        lambda_thresh = max(front_min_threshold, max_rad_val * front_evidence_quantile)

        # Support check: outermost radius with significant contiguous evidence
        supported_r_indices = np.where(rad_profile >= lambda_thresh)[0]
        if len(supported_r_indices) > 0:
            # Find outermost supported index with at least 2 contiguous bins or robust support
            outermost_idx = supported_r_indices[-1]
            r_front = float(grid.r_bins[outermost_idx])
            r_min = float(grid.r_bins[supported_r_indices[0]])
        else:
            r_front = r_peak
            r_min = r_peak

        # Per-ray front profile r_front(theta) across lobe
        front_profile = np.zeros(len(lobe_indices), dtype=np.float32)
        for j, pa_idx in enumerate(lobe_indices):
            ray = polar_ev[:, pa_idx]
            supp = np.where(ray >= lambda_thresh)[0]
            front_profile[j] = float(grid.r_bins[supp[-1]]) if len(supp) > 0 else r_peak

        cand = RadialCandidate(
            candidate_id=cand_id,
            peak_pa_deg=theta_p,
            weighted_cpa_deg=weighted_cpa,
            pa_width_deg=lobe_width,
            pa_start_deg=lobe_start_deg,
            pa_end_deg=lobe_end_deg,
            total_lobe_evidence=lobe_evidence,
            peak_evidence=peak_ev,
            prominence=float(prom),
            r_peak_px=r_peak,
            r_front_px=r_front,
            r_min_px=r_min,
            radial_profile=rad_profile,
            front_profile=front_profile,
            lobe_pa_indices=lobe_indices,
            left_valley_pa_deg=theta_L,
            right_valley_pa_deg=theta_R,
            left_valley_dist_deg=d_L,
            right_valley_dist_deg=d_R,
            selected_half_width_deg=half_w,
            fallback_used=fallback_used,
        )
        candidates.append(cand)
        cand_id += 1

    return candidates
