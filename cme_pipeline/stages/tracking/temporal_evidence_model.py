"""Shared Pixel-Level Temporal Evidence Model for CME Tracking.

Provides continuous-timestamp causal temporal accumulation (H1) and outward-motion-compensated
accumulation (H2) on 256x256 COR2 running-difference grids, with circular PA-radial analysis
and exact 256x256-to-64x64 patch-grid component alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg


@dataclass
class DirectionalPeak:
    """Directional peak on S^1."""
    cpa_deg: float
    amplitude: float
    prominence: float
    rank: int


@dataclass
class TemporalEvidenceConfig:
    """Configuration for temporal evidence modeling."""
    variant: str = "H2"  # 'H0', 'H1', or 'H2'
    mode: str = "linear_zpos"  # 'linear_zpos', 'softmax_z', 'softmax_zpos', 'clipped_softmax'
    tau: float = 3.0
    half_life_min: float = 45.0
    radial_speed_px_30m: float = 8.0  # nominal ~400 km/s in COR2 pixel space
    n_pa_bins: int = 360
    n_r_bins: int = 100
    smooth_sigma_deg: float = 4.0
    peak_height_factor: float = 1.2
    peak_prominence_factor: float = 0.3
    min_peak_distance_deg: float = 15.0
    max_peaks: int = 4
    clip_z: Optional[float] = None


class PixelTemporalEvidenceModel:
    """Causal, timestamp-aware pixel-level temporal evidence estimator."""

    def __init__(
        self,
        sx: float,
        sy: float,
        pa_raster: np.ndarray,
        radius_raster: np.ndarray,
        r_min_px: float = 22.0,
        r_max_px: float = 122.5,
        config: Optional[TemporalEvidenceConfig] = None,
    ):
        self.sx = float(sx)
        self.sy = float(sy)
        self.pa_raster = pa_raster.astype(np.float32)
        self.radius_raster = radius_raster.astype(np.float32)
        self.r_min_px = float(r_min_px)
        self.r_max_px = float(r_max_px)
        self.cfg = config or TemporalEvidenceConfig()

        self.tau_decay_min = self.cfg.half_life_min / np.log(2.0)

        # Precompute coordinate grids for backward radial warp
        yy, xx = np.indices((256, 256), dtype=np.float32)
        self.dx = xx - self.sx
        self.dy = yy - self.sy

        # PA binning
        self.pa_bins = np.linspace(0.0, 360.0, self.cfg.n_pa_bins + 1, dtype=np.float32)
        self.pa_centers = (self.pa_bins[:-1] + self.pa_bins[1:]) / 2.0
        self.pixel_pa_bin = np.clip(
            np.digitize(self.pa_raster, self.pa_bins) - 1, 0, self.cfg.n_pa_bins - 1
        )

        # State memory
        self.accumulated_state: Optional[np.ndarray] = None
        self.last_predicted_state: Optional[np.ndarray] = None
        self.last_timestamp: Optional[datetime] = None

    def reset(self) -> None:
        """Resets temporal accumulation memory."""
        self.accumulated_state = None
        self.last_predicted_state = None
        self.last_timestamp = None

    def get_accumulated_evidence(self) -> np.ndarray:
        """Returns the current accumulated evidence state raster."""
        if self.accumulated_state is None:
            return np.zeros((256, 256), dtype=np.float32)
        return self.accumulated_state

    def compute_spatial_evidence(
        self,
        z_map: np.ndarray,
        analysis_mask: np.ndarray,
    ) -> np.ndarray:
        """Converts standardized z-score map into valid relative spatial evidence."""
        p = np.zeros_like(z_map, dtype=np.float32)
        valid_idx = np.where(analysis_mask)
        if len(valid_idx[0]) == 0:
            return p

        z_valid = z_map[valid_idx].astype(np.float64)

        if self.cfg.clip_z is not None:
            z_valid = np.minimum(z_valid, float(self.cfg.clip_z))

        if self.cfg.mode == "linear_zpos":
            z_pos = np.maximum(0.0, z_valid)
            tot = np.sum(z_pos)
            p_valid = z_pos / tot if tot > 0 else np.full_like(z_valid, 1.0 / len(z_valid))

        elif self.cfg.mode == "softmax_z":
            shift = (z_valid - np.max(z_valid)) / max(self.cfg.tau, 1e-6)
            exp_v = np.exp(shift)
            tot = np.sum(exp_v)
            p_valid = exp_v / tot if tot > 0 else np.full_like(z_valid, 1.0 / len(z_valid))

        elif self.cfg.mode == "softmax_zpos":
            z_pos = np.maximum(0.0, z_valid)
            shift = (z_pos - np.max(z_pos)) / max(self.cfg.tau, 1e-6)
            exp_v = np.exp(shift)
            tot = np.sum(exp_v)
            p_valid = exp_v / tot if tot > 0 else np.full_like(z_valid, 1.0 / len(z_valid))

        elif self.cfg.mode == "clipped_softmax":
            cap = self.cfg.clip_z if self.cfg.clip_z is not None else 6.0
            z_clipped = np.minimum(z_valid, cap)
            shift = (z_clipped - np.max(z_clipped)) / max(self.cfg.tau, 1e-6)
            exp_v = np.exp(shift)
            tot = np.sum(exp_v)
            p_valid = exp_v / tot if tot > 0 else np.full_like(z_valid, 1.0 / len(z_valid))

        else:
            raise ValueError(f"Unknown spatial evidence mode: {self.cfg.mode}")

        p[valid_idx] = p_valid.astype(np.float32)
        return p

    def _compute_warp_coords(self, dr_px: float) -> np.ndarray:
        """Computes backward warp coordinates for outward radial motion dr_px."""
        r_prev = np.maximum(self.radius_raster - dr_px, 0.0)
        scale = np.where(self.radius_raster > 0, r_prev / self.radius_raster, 0.0)
        xx_prev = self.sx + self.dx * scale
        yy_prev = self.sy + self.dy * scale
        return np.array([yy_prev, xx_prev], dtype=np.float32)

    def update(
        self,
        z_map: np.ndarray,
        val_mask: np.ndarray,
        current_time: datetime,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Updates temporal accumulation with current observation.

        Returns:
            (accumulated_evidence, spatial_evidence, dt_min, alpha)
        """
        analysis_mask = val_mask & (self.radius_raster >= self.r_min_px) & (self.radius_raster <= self.r_max_px)
        p_curr = self.compute_spatial_evidence(z_map, analysis_mask)

        variant = self.cfg.variant.upper()
        if variant == "H0":
            self.last_predicted_state = np.zeros_like(p_curr)
            self.accumulated_state = p_curr.copy()
            self.last_timestamp = current_time
            return p_curr, p_curr, 0.0, 0.0

        if self.last_timestamp is None:
            dt_min = 0.0
            alpha = 0.0
        else:
            dt_min = max(0.0, (current_time - self.last_timestamp).total_seconds() / 60.0)
            alpha = float(np.exp(-dt_min / self.tau_decay_min))

        if self.accumulated_state is None or alpha <= 1e-7:
            accum = p_curr.copy()
            pred = np.zeros_like(p_curr)
        else:
            if variant == "H1":
                pred = self.accumulated_state.copy()
                accum = alpha * pred + p_curr
            elif variant == "H2":
                dr_px = (self.cfg.radial_speed_px_30m / 30.0) * dt_min
                warp_coords = self._compute_warp_coords(dr_px)
                pred = map_coordinates(
                    self.accumulated_state,
                    warp_coords,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
                # Mask out behind occulter
                behind_occulter = (self.radius_raster - dr_px) < self.r_min_px
                pred[behind_occulter] = 0.0
                accum = alpha * pred + p_curr
            else:
                raise ValueError(f"Unknown temporal variant: {variant}")

        self.accumulated_state = accum.copy()
        self.last_predicted_state = pred.copy()
        self.last_timestamp = current_time
        return accum, p_curr, dt_min, alpha

    def compute_directional_profile(
        self,
        evidence_map: np.ndarray,
        analysis_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates raw, count-normalized, and circularly smoothed PA directional profiles.

        Returns:
            (raw_profile, norm_profile, smooth_profile)
        """
        mask_idx = np.where(analysis_mask)
        bin_idx = self.pixel_pa_bin[mask_idx]
        weights = evidence_map[mask_idx].astype(np.float64)

        raw_profile = np.bincount(bin_idx, weights=weights, minlength=self.cfg.n_pa_bins)
        valid_counts = np.bincount(bin_idx, minlength=self.cfg.n_pa_bins)

        norm_profile = np.zeros_like(raw_profile)
        positive_counts = valid_counts > 0
        norm_profile[positive_counts] = raw_profile[positive_counts] / valid_counts[positive_counts]

        smooth_profile = gaussian_filter1d(
            norm_profile,
            sigma=self.cfg.smooth_sigma_deg,
            mode="wrap",
        )
        return raw_profile, norm_profile, smooth_profile

    def find_directional_peaks(
        self,
        smooth_profile: Optional[np.ndarray] = None,
    ) -> List[DirectionalPeak]:
        """Identifies circular directional peaks on S^1."""
        if smooth_profile is None:
            if self.accumulated_state is None:
                return []
            analysis_mask = (self.radius_raster >= self.r_min_px) & (self.radius_raster <= self.r_max_px)
            _, _, smooth_profile = self.compute_directional_profile(self.accumulated_state, analysis_mask)

        mean_val = float(np.mean(smooth_profile))
        std_val = float(np.std(smooth_profile))

        tiled = np.tile(smooth_profile, 3)
        min_height = mean_val * self.cfg.peak_height_factor
        min_prominence = max(std_val * self.cfg.peak_prominence_factor, 1e-8)
        distance_bins = int(self.cfg.min_peak_distance_deg / (360.0 / self.cfg.n_pa_bins))

        peaks, props = find_peaks(
            tiled,
            height=min_height,
            prominence=min_prominence,
            distance=max(distance_bins, 1),
        )

        mid_start = self.cfg.n_pa_bins
        mid_end = 2 * self.cfg.n_pa_bins
        valid_peaks = []
        for p_idx, peak in enumerate(peaks):
            if mid_start <= peak < mid_end:
                orig_bin = peak - mid_start
                amp = float(smooth_profile[orig_bin])
                prom = float(props["prominences"][p_idx])
                valid_peaks.append((orig_bin, amp, prom))

        valid_peaks.sort(key=lambda x: x[1], reverse=True)

        peak_list = []
        for rank, (b_idx, amp, prom) in enumerate(valid_peaks[:self.cfg.max_peaks], start=1):
            cpa = float(self.pa_centers[b_idx])
            peak_list.append(DirectionalPeak(cpa_deg=cpa, amplitude=amp, prominence=prom, rank=rank))

        return peak_list

    def extract_component_support(
        self,
        comp_mask_patch: np.ndarray,
        val_mask_pixel: np.ndarray,
        evidence_map_pixel: np.ndarray,
    ) -> Dict[str, float]:
        """Aggregates 256x256 pixel temporal evidence onto a 64x64 component patch mask.

        Returns:
            dict containing:
            - 'total_evidence': sum of pixel evidence on valid component pixels
            - 'mean_evidence': average evidence per valid component pixel
            - 'num_valid_pixels': count of valid pixels underlying component
            - 'kin_pred_overlap': overlap ratio with outward-predicted evidence
        """
        # Expand 64x64 patch mask to 256x256 pixel mask (exact 4x4 block expansion)
        comp_pixel_mask = np.repeat(np.repeat(comp_mask_patch, 4, axis=0), 4, axis=1)
        valid_comp_mask = comp_pixel_mask & val_mask_pixel & (self.radius_raster >= self.r_min_px) & (self.radius_raster <= self.r_max_px)

        n_valid = int(valid_comp_mask.sum())
        if n_valid == 0:
            return {
                "total_evidence": 0.0,
                "mean_evidence": 0.0,
                "num_valid_pixels": 0,
                "kin_pred_overlap": 0.0,
            }

        tot_ev = float(np.sum(evidence_map_pixel[valid_comp_mask]))
        mean_ev = tot_ev / n_valid

        # Compute kinematic outward prediction overlap if available
        if self.last_predicted_state is not None:
            pred_ev = float(np.sum(self.last_predicted_state[valid_comp_mask]))
            kin_ratio = pred_ev / max(tot_ev, 1e-9)
        else:
            kin_ratio = 1.0

        return {
            "total_evidence": tot_ev,
            "mean_evidence": mean_ev,
            "num_valid_pixels": n_valid,
            "kin_pred_overlap": kin_ratio,
        }
