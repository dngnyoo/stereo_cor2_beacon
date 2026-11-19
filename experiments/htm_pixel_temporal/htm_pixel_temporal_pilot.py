#!/usr/bin/env python3
"""Stereo_COR2 - Pixel-Level HTM-Inspired Temporal Evidence Pilot.

Feasibility experiment investigating whether temporally accumulated pixel-level
evidence (H1 static, H2 outward-motion-compensated) identifies CME propagation
directions more reliably and robustly than single-frame evidence (H0).

Evaluated on: 2010-12-12 STEREO-A COR2 256x256 running difference sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

from cme_pipeline.live_tracking_input import load_live_tracking_input


# ==============================================================================
# 1. Spatial Evidence Computation (Phase 1)
# ==============================================================================

def compute_spatial_evidence(
    z_map: np.ndarray,
    analysis_mask: np.ndarray,
    mode: str = "softmax_z",
    tau: float = 3.0,
    clip_z: Optional[float] = None,
) -> np.ndarray:
    """Converts pixel-level z-scores into a spatial relative-probability distribution.

    Args:
        z_map: Standardized z-score map (256, 256), float32.
        analysis_mask: Boolean mask (256, 256) of valid, unocculted annular pixels.
        mode: Evidence mode:
            - 'softmax_z': Numerically stable Softmax on raw z-scores.
            - 'softmax_zpos': Softmax on positive z-scores max(0, z).
            - 'linear_zpos': Normalized positive intensity max(0, z) / sum.
            - 'clipped_softmax': Softmax after clipping z at clip_z.
        tau: Softmax temperature parameter (> 0).
        clip_z: Optional ceiling for z-score clipping.

    Returns:
        p: Spatial relative-probability distribution on (256, 256).
           Zero for invalid/occulted pixels; sums to 1.0 over analysis_mask.
    """
    p = np.zeros_like(z_map, dtype=np.float32)
    valid_indices = np.where(analysis_mask)
    if len(valid_indices[0]) == 0:
        return p

    z_valid = z_map[valid_indices].astype(np.float64)

    if clip_z is not None:
        z_valid = np.minimum(z_valid, float(clip_z))

    if mode == "softmax_z":
        shift = (z_valid - np.max(z_valid)) / max(tau, 1e-6)
        exp_vals = np.exp(shift)
        sum_exp = np.sum(exp_vals)
        p_valid = exp_vals / sum_exp if sum_exp > 0 else np.full_like(z_valid, 1.0 / len(z_valid))

    elif mode == "softmax_zpos":
        z_pos = np.maximum(0.0, z_valid)
        shift = (z_pos - np.max(z_pos)) / max(tau, 1e-6)
        exp_vals = np.exp(shift)
        sum_exp = np.sum(exp_vals)
        p_valid = exp_vals / sum_exp if sum_exp > 0 else np.full_like(z_valid, 1.0 / len(z_valid))

    elif mode == "linear_zpos":
        z_pos = np.maximum(0.0, z_valid)
        total = np.sum(z_pos)
        if total > 0:
            p_valid = z_pos / total
        else:
            p_valid = np.full_like(z_valid, 1.0 / len(z_valid))

    elif mode == "clipped_softmax":
        cap = clip_z if clip_z is not None else 6.0
        z_clipped = np.minimum(z_valid, cap)
        shift = (z_clipped - np.max(z_clipped)) / max(tau, 1e-6)
        exp_vals = np.exp(shift)
        sum_exp = np.sum(exp_vals)
        p_valid = exp_vals / sum_exp if sum_exp > 0 else np.full_like(z_valid, 1.0 / len(z_valid))

    else:
        raise ValueError(f"Unknown spatial evidence mode: {mode}")

    p[valid_indices] = p_valid.astype(np.float32)
    return p


# ==============================================================================
# 2. Temporal Accumulator (Phase 2: H0, H1, H2)
# ==============================================================================

class TemporalAccumulator:
    """Manages recursive temporal accumulation for H0, H1, and H2."""

    def __init__(
        self,
        sx: float,
        sy: float,
        r_raster: np.ndarray,
        r_min_px: float,
        r_max_px: float,
        half_life_min: float = 45.0,
        radial_speed_px_30m: float = 8.0,
        image_shape: Tuple[int, int] = (256, 256),
    ):
        self.sx = float(sx)
        self.sy = float(sy)
        self.r_raster = r_raster
        self.r_min_px = float(r_min_px)
        self.r_max_px = float(r_max_px)
        self.half_life_min = float(half_life_min)
        self.tau_decay_min = self.half_life_min / np.log(2.0)
        self.radial_speed_px_30m = float(radial_speed_px_30m)
        self.image_shape = image_shape

        # Precompute coordinate grids for radial warping
        yy, xx = np.indices(image_shape, dtype=np.float32)
        self.dx = xx - self.sx
        self.dy = yy - self.sy

        # State storage
        self.h1_state: Optional[np.ndarray] = None
        self.h2_state: Optional[np.ndarray] = None
        self.last_timestamp: Optional[datetime] = None

    def reset(self):
        """Reset accumulated states."""
        self.h1_state = None
        self.h2_state = None
        self.last_timestamp = None

    def _compute_warp_coords(self, dr_px: float) -> np.ndarray:
        """Computes backward warp coordinates for outward radial motion dr_px."""
        r_prev = np.maximum(self.r_raster - dr_px, 0.0)
        scale = np.where(self.r_raster > 0, r_prev / self.r_raster, 0.0)
        xx_prev = self.sx + self.dx * scale
        yy_prev = self.sy + self.dy * scale
        return np.array([yy_prev, xx_prev], dtype=np.float32)

    def step(
        self,
        p_current: np.ndarray,
        current_time: datetime,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Advances temporal accumulator with current observation.

        Returns:
            (h0, h1, h2, dt_min, alpha)
        """
        # H0: Memoryless current frame
        h0 = p_current.copy()

        if self.last_timestamp is None:
            dt_min = 0.0
            alpha = 0.0
        else:
            dt_min = (current_time - self.last_timestamp).total_seconds() / 60.0
            if dt_min < 0:
                dt_min = 0.0
            alpha = float(np.exp(-dt_min / self.tau_decay_min))

        # H1: Static exponential accumulation
        if self.h1_state is None or alpha <= 1e-7:
            h1 = p_current.copy()
        else:
            h1 = alpha * self.h1_state + p_current
        self.h1_state = h1.copy()

        # H2: Outward-motion-compensated accumulation
        if self.h2_state is None or alpha <= 1e-7:
            h2 = p_current.copy()
        else:
            dr_px = (self.radial_speed_px_30m / 30.0) * dt_min
            warp_coords = self._compute_warp_coords(dr_px)
            # Bilinear interpolation
            h2_pred = map_coordinates(
                self.h2_state,
                warp_coords,
                order=1,
                mode="constant",
                cval=0.0,
            )
            # Mask out region behind occulter
            behind_occulter = (self.r_raster - dr_px) < self.r_min_px
            h2_pred[behind_occulter] = 0.0
            h2 = alpha * h2_pred + p_current
        self.h2_state = h2.copy()

        self.last_timestamp = current_time
        return h0, h1, h2, dt_min, alpha


# ==============================================================================
# 3. PA-Radial Analyzer (Phase 3)
# ==============================================================================

@dataclass
class PeakInfo:
    cpa_deg: float
    amplitude: float
    prominence: float
    rank: int


class PARadialAnalyzer:
    """Transforms 2D Cartesian evidence to PA-radial maps and identifies directional peaks."""

    def __init__(
        self,
        sx: float,
        sy: float,
        pa_raster: np.ndarray,
        r_raster: np.ndarray,
        r_min_px: float = 22.0,
        r_max_px: float = 122.5,
        n_pa_bins: int = 360,
        n_r_bins: int = 100,
        smooth_sigma_deg: float = 4.0,
    ):
        self.sx = float(sx)
        self.sy = float(sy)
        self.pa_raster = pa_raster
        self.r_raster = r_raster
        self.r_min_px = float(r_min_px)
        self.r_max_px = float(r_max_px)
        self.n_pa_bins = n_pa_bins
        self.n_r_bins = n_r_bins
        self.smooth_sigma_deg = float(smooth_sigma_deg)

        # PA bin edges and centers
        self.pa_bins = np.linspace(0.0, 360.0, n_pa_bins + 1)
        self.pa_centers = (self.pa_bins[:-1] + self.pa_bins[1:]) / 2.0
        self.r_bins = np.linspace(self.r_min_px, self.r_max_px, n_r_bins)

        # Precompute digitized PA bin index for each pixel in 256x256
        self.pixel_pa_bin = np.clip(
            np.digitize(self.pa_raster, self.pa_bins) - 1, 0, n_pa_bins - 1
        )

        # Precompute sampling grid for unwrapped polar map (Radius x PA)
        pa_grid_2d, r_grid_2d = np.meshgrid(self.pa_centers, self.r_bins)
        rad_grid = np.radians(pa_grid_2d)
        # In FITS north-up: dx = -r*sin(PA), dy = r*cos(PA)
        self.unwarp_x = self.sx - r_grid_2d * np.sin(rad_grid)
        self.unwarp_y = self.sy + r_grid_2d * np.cos(rad_grid)
        self.unwarp_coords = np.array([self.unwarp_y, self.unwarp_x], dtype=np.float32)

    def unwrap_polar(self, image_2d: np.ndarray) -> np.ndarray:
        """Unwraps 2D Cartesian image to polar (Radius x PA) array of shape (n_r_bins, n_pa_bins)."""
        return map_coordinates(
            image_2d.astype(np.float32),
            self.unwarp_coords,
            order=1,
            mode="constant",
            cval=0.0,
        )

    def compute_pa_profile(
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

        # Total directional evidence per PA bin
        raw_profile = np.bincount(bin_idx, weights=weights, minlength=self.n_pa_bins)
        # Valid pixel counts per PA bin
        valid_counts = np.bincount(bin_idx, minlength=self.n_pa_bins)

        # Valid-pixel-count normalized directional evidence
        norm_profile = np.zeros_like(raw_profile)
        positive_counts = valid_counts > 0
        norm_profile[positive_counts] = raw_profile[positive_counts] / valid_counts[positive_counts]

        # Circular smoothing using Gaussian filter with wrap-around boundary condition
        smooth_profile = gaussian_filter1d(
            raw_profile,
            sigma=self.smooth_sigma_deg,
            mode="wrap",
        )
        return raw_profile, norm_profile, smooth_profile

    def find_directional_peaks(
        self,
        smooth_profile: np.ndarray,
        height_factor: float = 1.2,
        prominence_factor: float = 0.3,
        min_distance_deg: float = 15.0,
        max_peaks: int = 3,
    ) -> List[PeakInfo]:
        """Identifies directional peaks on S^1 preserving circular geometry."""
        mean_val = float(np.mean(smooth_profile))
        std_val = float(np.std(smooth_profile))

        # Enforce circular periodicity by replicating 3 periods: [-360, 720)
        tiled_profile = np.tile(smooth_profile, 3)

        min_height = mean_val * height_factor
        min_prominence = max(std_val * prominence_factor, 1e-8)
        distance_bins = int(min_distance_deg / (360.0 / self.n_pa_bins))

        peaks, props = find_peaks(
            tiled_profile,
            height=min_height,
            prominence=min_prominence,
            distance=max(distance_bins, 1),
        )

        # Select peaks falling strictly in the middle period [n_pa_bins, 2 * n_pa_bins)
        mid_start = self.n_pa_bins
        mid_end = 2 * self.n_pa_bins
        valid_peaks = []
        for p_idx, peak in enumerate(peaks):
            if mid_start <= peak < mid_end:
                orig_bin = peak - mid_start
                amp = float(smooth_profile[orig_bin])
                prom = float(props["prominences"][p_idx])
                valid_peaks.append((orig_bin, amp, prom))

        # Rank peaks by amplitude (descending)
        valid_peaks.sort(key=lambda item: item[1], reverse=True)

        peak_infos = []
        for rank, (b_idx, amp, prom) in enumerate(valid_peaks[:max_peaks], start=1):
            cpa = float(self.pa_centers[b_idx])
            peak_infos.append(PeakInfo(cpa_deg=cpa, amplitude=amp, prominence=prom, rank=rank))

        return peak_infos


# ==============================================================================
# 4. Visualization Engine (Phase 4)
# ==============================================================================

class Visualizer:
    """Generates 6-panel frame diagnostic visualizations and time-series Hovmoller plots."""

    def __init__(
        self,
        output_dir: Path,
        sx: float,
        sy: float,
        r_min_px: float,
        r_max_px: float,
    ):
        self.output_dir = output_dir
        self.frames_dir = output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.sx = sx
        self.sy = sy
        self.r_min_px = r_min_px
        self.r_max_px = r_max_px

    def plot_frame_diagnostic(
        self,
        frame_idx: int,
        pair_idx: int,
        timestamp: datetime,
        rd_image: np.ndarray,
        z_score: np.ndarray,
        p_spatial: np.ndarray,
        accum_evidence: np.ndarray,
        polar_evidence: np.ndarray,
        pa_centers: np.ndarray,
        r_bins: np.ndarray,
        peaks_h0: List[PeakInfo],
        peaks_h1: List[PeakInfo],
        peaks_h2: List[PeakInfo],
        accum_variant: str = "H2",
        analysis_mask: Optional[np.ndarray] = None,
    ):
        """Creates the 6-panel diagnostic visualization specified in Phase 4.

        Panels:
        1. Original north-up RD
        2. Pixel-level z-score
        3. Current-frame spatial relative probability (Softmax)
        4. Temporally accumulated evidence (H1/H2)
        5. PA-radial evidence map (polar unwrapped)
        6. Detected CPA candidate directions overlaid on RD
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#111118")
        axes = axes.flatten()

        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        fig.suptitle(
            f"Stereo_COR2 HTM Temporal Evidence Pilot - Frame {frame_idx:02d} (Pair {pair_idx}) | {time_str}",
            color="white",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Style all axes
        for ax in axes:
            ax.set_facecolor("black")
            ax.tick_params(colors="lightgray")
            for spine in ax.spines.values():
                spine.set_color("#444455")

        # 1. Original north-up RD
        ax0 = axes[0]
        rd_display = np.nan_to_num(rd_image, nan=0.0)
        vlim = np.percentile(np.abs(rd_display[analysis_mask]), 98) if analysis_mask is not None else 50.0
        im0 = ax0.imshow(rd_display, origin="lower", cmap="gray", vmin=-vlim, vmax=vlim)
        self._add_occulter_rings(ax0)
        ax0.set_title("1. Original North-Up Running Difference", color="white", fontsize=11)
        cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        cb0.set_label("DN / s", color="lightgray", fontsize=9)
        cb0.ax.tick_params(colors="lightgray", labelsize=8)

        # 2. Pixel-level z-score
        ax1 = axes[1]
        z_display = np.nan_to_num(z_score, nan=0.0)
        if analysis_mask is not None:
            z_display = np.where(analysis_mask, z_display, 0.0)
        im1 = ax1.imshow(z_display, origin="lower", cmap="coolwarm", vmin=-3.0, vmax=6.0)
        self._add_occulter_rings(ax1)
        ax1.set_title("2. Pixel-Level Standardized z-score", color="white", fontsize=11)
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_label("z-score (sigma)", color="lightgray", fontsize=9)
        cb1.ax.tick_params(colors="lightgray", labelsize=8)

        # 3. Current-frame spatial probability
        ax2 = axes[2]
        p_disp = p_spatial.copy()
        if analysis_mask is not None:
            p_disp = np.where(analysis_mask, p_disp, np.nan)
        vmax_p = np.nanpercentile(p_disp[analysis_mask], 99.8) if analysis_mask is not None else 1e-3
        im2 = ax2.imshow(p_disp, origin="lower", cmap="inferno", vmin=0, vmax=max(vmax_p, 1e-5))
        self._add_occulter_rings(ax2)
        ax2.set_title("3. Current-Frame Spatial Probability (Softmax)", color="white", fontsize=11)
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("p(x, y)", color="lightgray", fontsize=9)
        cb2.ax.tick_params(colors="lightgray", labelsize=8)

        # 4. Temporally accumulated evidence (H1/H2)
        ax3 = axes[3]
        ev_disp = accum_evidence.copy()
        if analysis_mask is not None:
            ev_disp = np.where(analysis_mask, ev_disp, np.nan)
        vmax_ev = np.nanpercentile(ev_disp[analysis_mask], 99.8) if analysis_mask is not None else 1e-3
        im3 = ax3.imshow(ev_disp, origin="lower", cmap="viridis", vmin=0, vmax=max(vmax_ev, 1e-5))
        self._add_occulter_rings(ax3)
        ax3.set_title(f"4. Temporally Accumulated Evidence ({accum_variant})", color="white", fontsize=11)
        cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb3.set_label("Accumulated Evidence", color="lightgray", fontsize=9)
        cb3.ax.tick_params(colors="lightgray", labelsize=8)

        # 5. PA-radial evidence map (polar unrolled: Radius vs PA)
        ax4 = axes[4]
        # polar_evidence shape: (n_r_bins, n_pa_bins)
        # extent: [pa_min, pa_max, r_min, r_max]
        im4 = ax4.imshow(
            polar_evidence,
            origin="lower",
            extent=[0.0, 360.0, self.r_min_px, self.r_max_px],
            aspect="auto",
            cmap="magma",
        )
        ax4.set_title(f"5. PA-Radial Evidence Map ({accum_variant})", color="white", fontsize=11)
        ax4.set_xlabel("Position Angle (deg, North=0, East=90)", color="lightgray", fontsize=9)
        ax4.set_ylabel("Radius (pixels)", color="lightgray", fontsize=9)
        ax4.set_xticks(np.arange(0, 361, 45))
        cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cb4.set_label("Polar Evidence Density", color="lightgray", fontsize=9)
        cb4.ax.tick_params(colors="lightgray", labelsize=8)

        # Overlay detected peaks on polar map
        for pk in peaks_h2:
            ax4.axvline(pk.cpa_deg, color="cyan" if pk.rank == 1 else "yellow", linestyle="--", alpha=0.7)

        # 6. Detected CPA candidate directions overlaid on original RD
        ax5 = axes[5]
        ax5.imshow(rd_display, origin="lower", cmap="gray", vmin=-vlim, vmax=vlim)
        self._add_occulter_rings(ax5)
        ax5.set_title("6. Detected CPA Candidates Overlaid on RD", color="white", fontsize=11)

        # Plot candidate CPA rays from solar center
        r_ray = np.linspace(self.r_min_px, self.r_max_px, 50)
        legend_handles = []

        # Overlay H0 primary peak
        if peaks_h0 and np.isfinite(peaks_h0[0].cpa_deg):
            cpa0 = peaks_h0[0].cpa_deg
            rad0 = np.radians(cpa0)
            x_ray0 = self.sx - r_ray * np.sin(rad0)
            y_ray0 = self.sy + r_ray * np.cos(rad0)
            (line_h0,) = ax5.plot(x_ray0, y_ray0, color="#FF5555", linestyle=":", linewidth=2, label=f"H0 CPA: {cpa0:.1f}°")
            legend_handles.append(line_h0)

        # Overlay H1 primary peak
        if peaks_h1 and np.isfinite(peaks_h1[0].cpa_deg):
            cpa1 = peaks_h1[0].cpa_deg
            rad1 = np.radians(cpa1)
            x_ray1 = self.sx - r_ray * np.sin(rad1)
            y_ray1 = self.sy + r_ray * np.cos(rad1)
            (line_h1,) = ax5.plot(x_ray1, y_ray1, color="#FFB800", linestyle="-.", linewidth=2, label=f"H1 CPA: {cpa1:.1f}°")
            legend_handles.append(line_h1)

        # Overlay H2 peaks
        colors_h2 = ["#00FFFF", "#39FF14", "#FF00FF"]
        for pk in peaks_h2:
            rad2 = np.radians(pk.cpa_deg)
            x_ray2 = self.sx - r_ray * np.sin(rad2)
            y_ray2 = self.sy + r_ray * np.cos(rad2)
            style = "-" if pk.rank == 1 else "--"
            width = 2.5 if pk.rank == 1 else 1.8
            color = colors_h2[(pk.rank - 1) % len(colors_h2)]
            (line_h2,) = ax5.plot(
                x_ray2,
                y_ray2,
                color=color,
                linestyle=style,
                linewidth=width,
                label=f"H2 Peak {pk.rank}: {pk.cpa_deg:.1f}°",
            )
            legend_handles.append(line_h2)

            # Annotate peak CPA near outer edge
            x_label = self.sx - (self.r_max_px + 8) * np.sin(rad2)
            y_label = self.sy + (self.r_max_px + 8) * np.cos(rad2)
            ax5.text(
                x_label,
                y_label,
                f"{pk.cpa_deg:.0f}°",
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

        if legend_handles:
            ax5.legend(handles=legend_handles, loc="upper right", fontsize=8, facecolor="#1e1e28", edgecolor="#555566", labelcolor="white")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = self.frames_dir / f"diag_frame_{frame_idx:02d}_pair_{pair_idx}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path

    def _add_occulter_rings(self, ax: plt.Axes):
        """Draws occulting disk and outer FOV boundaries."""
        occulter = plt.Circle(
            (self.sx, self.sy),
            self.r_min_px,
            fill=False,
            color="#00FFAA",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        fov_outer = plt.Circle(
            (self.sx, self.sy),
            self.r_max_px,
            fill=False,
            color="#FF0055",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        sun_center = plt.Circle(
            (self.sx, self.sy),
            1.5,
            fill=True,
            color="yellow",
            alpha=0.9,
        )
        ax.add_patch(occulter)
        ax.add_patch(fov_outer)
        ax.add_patch(sun_center)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)

    def plot_hovmoller_comparison(
        self,
        timestamps: List[datetime],
        pa_profiles_h0: np.ndarray,
        pa_profiles_h1: np.ndarray,
        pa_profiles_h2: np.ndarray,
        cpa_candidates_h0: List[float],
        cpa_candidates_h1: List[float],
        cpa_candidates_h2: List[float],
        catalog_events: List[Dict[str, Any]],
        output_path: Path,
    ):
        """Creates the full-day directional evolution Hovmoller diagram (PA vs Time)."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, sharey=True, facecolor="#111118")

        time_labels = [t.strftime("%H:%M") for t in timestamps]
        n_frames = len(timestamps)
        pa_bins = np.linspace(0, 360, pa_profiles_h0.shape[1])

        variants = [
            ("H0: Single-Frame Evidence (Memoryless)", pa_profiles_h0, cpa_candidates_h0, "#FF5555"),
            ("H1: Exponentially Accumulated Evidence (Static)", pa_profiles_h1, cpa_candidates_h1, "#FFB800"),
            ("H2: Outward-Motion-Compensated Accumulation (v_r=8px/30m)", pa_profiles_h2, cpa_candidates_h2, "#00FFFF"),
        ]

        for idx, (title, profile_data, cpa_list, cpa_color) in enumerate(variants):
            ax = axes[idx]
            ax.set_facecolor("black")
            ax.tick_params(colors="lightgray")
            for spine in ax.spines.values():
                spine.set_color("#444455")

            # profile_data shape: (n_frames, 360) -> transpose for (PA, Time)
            data_T = profile_data.T
            # Normalize each column for visualization contrast
            col_max = np.maximum(np.max(data_T, axis=0, keepdims=True), 1e-9)
            norm_data = data_T / col_max

            im = ax.imshow(
                norm_data,
                origin="lower",
                extent=[0, n_frames - 1, 0, 360],
                aspect="auto",
                cmap="inferno",
            )

            # Overlay detected CPA track
            valid_x = []
            valid_y = []
            for f_i, cpa in enumerate(cpa_list):
                if np.isfinite(cpa):
                    valid_x.append(f_i)
                    valid_y.append(cpa)

            ax.plot(valid_x, valid_y, color=cpa_color, marker="o", markersize=4, linestyle="-", linewidth=1.5, label=f"Detected CPA Track ({title.split(':')[0]})", alpha=0.9)

            # Overlay external catalog reference markers
            for cat in catalog_events:
                # Find closest frame index
                cat_t = cat["time"]
                diffs = [abs((t - cat_t).total_seconds()) for t in timestamps]
                nearest_f = int(np.argmin(diffs))
                cat_cpa = cat["cpa"]
                width = cat.get("width", 0.0)
                src = cat["source"]

                ax.scatter(
                    nearest_f,
                    cat_cpa,
                    marker="*",
                    s=140,
                    color="#00FFAA" if src == "SEEDS" else "#FF00FF",
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=10,
                    label=f"{src} Catalog Event (CPA={cat_cpa}°)" if idx == 0 else "",
                )
                if width > 0:
                    pa_low = (cat_cpa - width / 2.0) % 360.0
                    pa_high = (cat_cpa + width / 2.0) % 360.0
                    if pa_low <= pa_high:
                        ax.vlines(nearest_f, pa_low, pa_high, color="#00FFAA" if src == "SEEDS" else "#FF00FF", alpha=0.5, linewidth=2.5)

            ax.set_title(title, color="white", fontsize=12, fontweight="bold")
            ax.set_ylabel("Position Angle (deg)", color="lightgray", fontsize=10)
            ax.set_yticks(np.arange(0, 361, 45))
            ax.set_ylim(0, 360)
            ax.legend(loc="upper left", fontsize=8, facecolor="#1e1e28", edgecolor="#555566", labelcolor="white")

            cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
            cb.set_label("Relative PA Density", color="lightgray", fontsize=8)
            cb.ax.tick_params(colors="lightgray", labelsize=8)

        # Set shared x ticks
        axes[-1].set_xticks(np.arange(n_frames))
        axes[-1].set_xticklabels(time_labels, rotation=45, ha="right", fontsize=9, color="lightgray")
        axes[-1].set_xlabel("Observation Timestamp (UTC, 2010-12-12)", color="lightgray", fontsize=10)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved Hovmoller time-series plot to: {output_path}")


# ==============================================================================
# 5. Core Experiment Runner & Evaluator (Phase 5)
# ==============================================================================

class ExperimentRunner:
    """Executes the complete experimental workflow across all frames and parameter sweeps."""

    def __init__(
        self,
        detection_dir: str = "runs/clean_cor2a_verify/detection",
        output_dir: str = "experiments/htm_pixel_temporal",
    ):
        self.output_dir = Path(output_dir).resolve()
        self.vis_dir = self.output_dir / "visualizations"
        self.res_dir = self.output_dir / "results"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.res_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading live tracking input from: {detection_dir}")
        self.det = load_live_tracking_input(detection_dir)
        self.geom = self.det.geometry
        self.sx = self.geom.sx
        self.sy = self.geom.sy
        self.r_min = self.geom.occulter_r_min_px
        self.r_max = self.geom.fov_r_max_px

        # Filter evaluation frames: VALID and usable_detection
        self.eval_frames = [
            (idx, f) for idx, f in enumerate(self.det.frame_cache)
            if f["observation_status"] == "VALID" and f["usable_detection"]
        ]
        print(f"Identified {len(self.eval_frames)} valid evaluation frames for analysis.")

        # Initialize analyzer and visualizer
        self.analyzer = PARadialAnalyzer(
            sx=self.sx,
            sy=self.sy,
            pa_raster=self.geom.pa_raster,
            r_raster=self.geom.radius_raster,
            r_min_px=self.r_min,
            r_max_px=self.r_max,
        )
        self.visualizer = Visualizer(
            output_dir=self.vis_dir,
            sx=self.sx,
            sy=self.sy,
            r_min_px=self.r_min,
            r_max_px=self.r_max,
        )

        # External catalog reference events for 2010-12-12
        self.catalog_events = [
            {"source": "SEEDS", "time": datetime(2010, 12, 12, 4, 54), "cpa": 91.0, "width": 138.0, "speed": 499.0},
            {"source": "SEEDS", "time": datetime(2010, 12, 12, 9, 24), "cpa": 76.0, "width": 113.0, "speed": 401.0},
            {"source": "SEEDS", "time": datetime(2010, 12, 12, 11, 24), "cpa": 113.0, "width": 32.0, "speed": 2901.0},
            {"source": "SEEDS", "time": datetime(2010, 12, 12, 12, 54), "cpa": 154.0, "width": 111.0, "speed": 270.0},
            {"source": "SEEDS", "time": datetime(2010, 12, 12, 13, 54), "cpa": 40.0, "width": 39.0, "speed": 248.0},
            {"source": "SEEDS", "time": datetime(2010, 12, 12, 17, 24), "cpa": 117.0, "width": 34.0, "speed": 373.0},
            {"source": "Gottingen", "time": datetime(2010, 12, 12, 8, 8, 15), "cpa": 90.0, "width": 90.0, "speed": 451.0},
        ]

    def run_baseline_pipeline(
        self,
        tau: float = 3.0,
        mode: str = "softmax_z",
        half_life_min: float = 45.0,
        radial_speed_px_30m: float = 8.0,
        generate_all_frame_plots: bool = True,
    ) -> pd.DataFrame:
        """Executes the standard baseline run across all evaluation frames."""
        print(f"\n--- Running Baseline Pilot (tau={tau}, mode={mode}, half_life={half_life_min}m, v_r={radial_speed_px_30m}px/30m) ---")

        accumulator = TemporalAccumulator(
            sx=self.sx,
            sy=self.sy,
            r_raster=self.geom.radius_raster,
            r_min_px=self.r_min,
            r_max_px=self.r_max,
            half_life_min=half_life_min,
            radial_speed_px_30m=radial_speed_px_30m,
        )

        rows = []
        timestamps = []
        h0_profiles = []
        h1_profiles = []
        h2_profiles = []
        cpa_h0 = []
        cpa_h1 = []
        cpa_h2 = []

        for seq_idx, (cache_idx, f) in enumerate(self.eval_frames):
            timestamp = f["timestamp"]
            pair_idx = f["pair_idx"]
            z_map = f["z_global"]
            val_mask = f["val"]
            r_map = self.geom.radius_raster
            analysis_mask = val_mask & (r_map >= self.r_min) & (r_map <= self.r_max)

            # Phase 1: Spatial Evidence
            p_spatial = compute_spatial_evidence(
                z_map=z_map,
                analysis_mask=analysis_mask,
                mode=mode,
                tau=tau,
            )

            # Phase 2: Temporal Accumulation
            h0, h1, h2, dt_min, alpha = accumulator.step(p_spatial, timestamp)

            # Phase 3: PA-Radial Analysis
            raw0, norm0, sm0 = self.analyzer.compute_pa_profile(h0, analysis_mask)
            raw1, norm1, sm1 = self.analyzer.compute_pa_profile(h1, analysis_mask)
            raw2, norm2, sm2 = self.analyzer.compute_pa_profile(h2, analysis_mask)

            # Polar unrolled map for H2
            polar_h2 = self.analyzer.unwrap_polar(h2)

            # Peak finding
            peaks0 = self.analyzer.find_directional_peaks(sm0)
            peaks1 = self.analyzer.find_directional_peaks(sm1)
            peaks2 = self.analyzer.find_directional_peaks(sm2)

            timestamps.append(timestamp)
            h0_profiles.append(sm0)
            h1_profiles.append(sm1)
            h2_profiles.append(sm2)

            p0_cpa = peaks0[0].cpa_deg if peaks0 else np.nan
            p1_cpa = peaks1[0].cpa_deg if peaks1 else np.nan
            p2_cpa = peaks2[0].cpa_deg if peaks2 else np.nan

            cpa_h0.append(p0_cpa)
            cpa_h1.append(p1_cpa)
            cpa_h2.append(p2_cpa)

            # Diagnostic plot
            if generate_all_frame_plots or seq_idx in [0, 1, 3, 4, 6, 8, 12, 13, 14, 18, 19, 20, 25, 30]:
                self.visualizer.plot_frame_diagnostic(
                    frame_idx=seq_idx,
                    pair_idx=pair_idx,
                    timestamp=timestamp,
                    rd_image=f["rd"],
                    z_score=z_map,
                    p_spatial=p_spatial,
                    accum_evidence=h2,
                    polar_evidence=polar_h2,
                    pa_centers=self.analyzer.pa_centers,
                    r_bins=self.analyzer.r_bins,
                    peaks_h0=peaks0,
                    peaks_h1=peaks1,
                    peaks_h2=peaks2,
                    accum_variant="H2",
                    analysis_mask=analysis_mask,
                )

            # Record row
            row = {
                "sequence_index": seq_idx,
                "pair_index": pair_idx,
                "timestamp": timestamp.isoformat(),
                "time_utc": timestamp.strftime("%H:%M:%S"),
                "dt_from_prev_min": dt_min,
                "temporal_decay_alpha": alpha,
                # H0 candidates
                "h0_cpa_primary": p0_cpa,
                "h0_amp_primary": peaks0[0].amplitude if peaks0 else 0.0,
                "h0_cpa_secondary": peaks0[1].cpa_deg if len(peaks0) > 1 else np.nan,
                "h0_n_peaks": len(peaks0),
                # H1 candidates
                "h1_cpa_primary": p1_cpa,
                "h1_amp_primary": peaks1[0].amplitude if peaks1 else 0.0,
                "h1_cpa_secondary": peaks1[1].cpa_deg if len(peaks1) > 1 else np.nan,
                "h1_n_peaks": len(peaks1),
                # H2 candidates
                "h2_cpa_primary": p2_cpa,
                "h2_amp_primary": peaks2[0].amplitude if peaks2 else 0.0,
                "h2_cpa_secondary": peaks2[1].cpa_deg if len(peaks2) > 1 else np.nan,
                "h2_n_peaks": len(peaks2),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.res_dir / "frame_cpa_candidates.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported frame-level CPA candidates CSV to: {csv_path}")

        # Generate full-day Hovmoller diagram
        hovmoller_path = self.vis_dir / "hovmoller_time_series_comparison.png"
        self.visualizer.plot_hovmoller_comparison(
            timestamps=timestamps,
            pa_profiles_h0=np.array(h0_profiles),
            pa_profiles_h1=np.array(h1_profiles),
            pa_profiles_h2=np.array(h2_profiles),
            cpa_candidates_h0=cpa_h0,
            cpa_candidates_h1=cpa_h1,
            cpa_candidates_h2=cpa_h2,
            catalog_events=self.catalog_events,
            output_path=hovmoller_path,
        )

        return df

    def run_parameter_sensitivity(self) -> Dict[str, Any]:
        """Runs systematic sensitivity sweeps over temperature, decay, and motion."""
        print("\n========================================================")
        print("Starting Systematic Parameter Sensitivity Analysis...")
        print("========================================================")

        results = {}

        # 1. Softmax Temperature Sweep
        temps = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        temp_metrics = []
        for t in temps:
            df = self._run_pipeline_silent(tau=t, mode="softmax_z", half_life_min=45.0, radial_speed=8.0)
            metrics = self._compute_run_metrics(df)
            metrics["tau"] = t
            temp_metrics.append(metrics)
            print(f"Temperature tau={t:4.1f} | H0 Jitter: {metrics['h0_mean_jump']:.2f}° | H1 Jitter: {metrics['h1_mean_jump']:.2f}° | H2 Jitter: {metrics['h2_mean_jump']:.2f}° | Star Rejection H2: {metrics['star_rejection_h2']:.2f}")

        results["temperature_sweep"] = pd.DataFrame(temp_metrics)
        results["temperature_sweep"].to_csv(self.res_dir / "sensitivity_temperature.csv", index=False)

        # 2. Evidence Mode Sweep
        modes = ["softmax_z", "softmax_zpos", "linear_zpos", "clipped_softmax"]
        mode_metrics = []
        for m in modes:
            df = self._run_pipeline_silent(tau=3.0, mode=m, half_life_min=45.0, radial_speed=8.0)
            metrics = self._compute_run_metrics(df)
            metrics["mode"] = m
            mode_metrics.append(metrics)
            print(f"Mode: {m:16s} | H0 Jitter: {metrics['h0_mean_jump']:.2f}° | H1 Jitter: {metrics['h1_mean_jump']:.2f}° | H2 Jitter: {metrics['h2_mean_jump']:.2f}° | Star Rejection H2: {metrics['star_rejection_h2']:.2f}")

        results["mode_sweep"] = pd.DataFrame(mode_metrics)
        results["mode_sweep"].to_csv(self.res_dir / "sensitivity_evidence_mode.csv", index=False)

        # 3. Decay Half-Life Sweep
        half_lives = [15.0, 30.0, 45.0, 90.0, 180.0]
        decay_metrics = []
        for hl in half_lives:
            df = self._run_pipeline_silent(tau=3.0, mode="softmax_z", half_life_min=hl, radial_speed=8.0)
            metrics = self._compute_run_metrics(df)
            metrics["half_life_min"] = hl
            decay_metrics.append(metrics)
            print(f"Half-Life hl={hl:5.1f}m | H1 Jitter: {metrics['h1_mean_jump']:.2f}° | H2 Jitter: {metrics['h2_mean_jump']:.2f}° | H2 Max Jump: {metrics['h2_max_jump']:.2f}°")

        results["decay_sweep"] = pd.DataFrame(decay_metrics)
        results["decay_sweep"].to_csv(self.res_dir / "sensitivity_temporal_decay.csv", index=False)

        # 4. Outward Radial Speed Sweep (v_r)
        speeds = [0.0, 4.0, 8.0, 12.0, 16.0]
        speed_metrics = []
        for sp in speeds:
            df = self._run_pipeline_silent(tau=3.0, mode="softmax_z", half_life_min=45.0, radial_speed=sp)
            metrics = self._compute_run_metrics(df)
            metrics["radial_speed_px_30m"] = sp
            speed_metrics.append(metrics)
            print(f"Speed v_r={sp:4.1f} px/30m | H2 Jitter: {metrics['h2_mean_jump']:.2f}° | H2 Max Jump: {metrics['h2_max_jump']:.2f}° | H2 Persistence: {metrics['h2_mean_persistence']:.2f} frames")

        results["speed_sweep"] = pd.DataFrame(speed_metrics)
        results["speed_sweep"].to_csv(self.res_dir / "sensitivity_radial_speed.csv", index=False)

        # Generate summary plot for sensitivity
        self._plot_sensitivity_summary(results)

        return results

    def _run_pipeline_silent(
        self,
        tau: float,
        mode: str,
        half_life_min: float,
        radial_speed: float,
    ) -> pd.DataFrame:
        """Runs pipeline silently without producing image files."""
        accumulator = TemporalAccumulator(
            sx=self.sx,
            sy=self.sy,
            r_raster=self.geom.radius_raster,
            r_min_px=self.r_min,
            r_max_px=self.r_max,
            half_life_min=half_life_min,
            radial_speed_px_30m=radial_speed,
        )
        rows = []
        for seq_idx, (cache_idx, f) in enumerate(self.eval_frames):
            timestamp = f["timestamp"]
            z_map = f["z_global"]
            val_mask = f["val"]
            analysis_mask = val_mask & (self.geom.radius_raster >= self.r_min) & (self.geom.radius_raster <= self.r_max)

            p_spatial = compute_spatial_evidence(z_map, analysis_mask, mode=mode, tau=tau)
            h0, h1, h2, dt_min, alpha = accumulator.step(p_spatial, timestamp)

            _, _, sm0 = self.analyzer.compute_pa_profile(h0, analysis_mask)
            _, _, sm1 = self.analyzer.compute_pa_profile(h1, analysis_mask)
            _, _, sm2 = self.analyzer.compute_pa_profile(h2, analysis_mask)

            pks0 = self.analyzer.find_directional_peaks(sm0)
            pks1 = self.analyzer.find_directional_peaks(sm1)
            pks2 = self.analyzer.find_directional_peaks(sm2)

            rows.append({
                "seq_idx": seq_idx,
                "timestamp": timestamp,
                "h0_cpa": pks0[0].cpa_deg if pks0 else np.nan,
                "h1_cpa": pks1[0].cpa_deg if pks1 else np.nan,
                "h2_cpa": pks2[0].cpa_deg if pks2 else np.nan,
                "sm2_east": float(np.sum(sm2[30:140])),
                "sm2_west": float(np.sum(sm2[250:275])),
            })
        return pd.DataFrame(rows)

    def _compute_run_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculates quantitative performance metrics for a pipeline run."""
        metrics = {}
        for var in ["h0", "h1", "h2"]:
            cpas = df[f"{var}_cpa"].to_numpy()
            diffs = []
            for i in range(len(cpas) - 1):
                c1, c2 = cpas[i], cpas[i + 1]
                if np.isfinite(c1) and np.isfinite(c2):
                    # Circular angular difference on S^1
                    d = abs((c2 - c1 + 180.0) % 360.0 - 180.0)
                    diffs.append(d)
                else:
                    diffs.append(180.0)  # Penalty for missed detection

            metrics[f"{var}_mean_jump"] = float(np.mean(diffs)) if diffs else np.nan
            metrics[f"{var}_median_jump"] = float(np.median(diffs)) if diffs else np.nan
            metrics[f"{var}_max_jump"] = float(np.max(diffs)) if diffs else np.nan
            metrics[f"{var}_valid_fraction"] = float(np.mean(np.isfinite(cpas)))

            # Persistence: average streak length of CPA staying within 15 deg
            streaks = []
            current_streak = 1
            for d in diffs:
                if d <= 15.0:
                    current_streak += 1
                else:
                    streaks.append(current_streak)
                    current_streak = 1
            streaks.append(current_streak)
            metrics[f"{var}_mean_persistence"] = float(np.mean(streaks))

        # Star rejection ratio at frame index 13 (approx 11:54 UT)
        if "sm2_east" in df.columns and len(df) > 13:
            east_val = df.iloc[13]["sm2_east"]
            west_val = df.iloc[13]["sm2_west"]
            metrics["star_rejection_h2"] = float(east_val / max(west_val, 1e-9))
        else:
            metrics["star_rejection_h2"] = 1.0

        return metrics

    def _plot_sensitivity_summary(self, results: Dict[str, pd.DataFrame]):
        """Plots summary curves for sensitivity analyses."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#111118")

        for ax in axes.flatten():
            ax.set_facecolor("#1a1a24")
            ax.tick_params(colors="lightgray")
            for spine in ax.spines.values():
                spine.set_color("#444455")

        # 1. Temperature vs Jitter
        ax1 = axes[0, 0]
        df_temp = results["temperature_sweep"]
        ax1.plot(df_temp["tau"], df_temp["h0_mean_jump"], "o--", color="#FF5555", label="H0 (Current)")
        ax1.plot(df_temp["tau"], df_temp["h1_mean_jump"], "s-.", color="#FFB800", label="H1 (Static)")
        ax1.plot(df_temp["tau"], df_temp["h2_mean_jump"], "^-", color="#00FFFF", linewidth=2, label="H2 (Motion-Comp)")
        ax1.set_title("CPA Frame-to-Frame Jitter vs Softmax Temperature", color="white", fontsize=11)
        ax1.set_xlabel("Temperature tau", color="lightgray")
        ax1.set_ylabel("Mean Consecutive Jump (deg)", color="lightgray")
        ax1.legend(facecolor="#252535", edgecolor="#555566", labelcolor="white")
        ax1.grid(True, linestyle=":", alpha=0.3, color="gray")

        # 2. Mode vs Jitter
        ax2 = axes[0, 1]
        df_mode = results["mode_sweep"]
        x_pos = np.arange(len(df_mode))
        width = 0.25
        ax2.bar(x_pos - width, df_mode["h0_mean_jump"], width, label="H0", color="#FF5555")
        ax2.bar(x_pos, df_mode["h1_mean_jump"], width, label="H1", color="#FFB800")
        ax2.bar(x_pos + width, df_mode["h2_mean_jump"], width, label="H2", color="#00FFFF")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df_mode["mode"], rotation=20, ha="right", color="lightgray")
        ax2.set_title("Directional Jitter across Spatial Evidence Formulations", color="white", fontsize=11)
        ax2.set_ylabel("Mean Consecutive Jump (deg)", color="lightgray")
        ax2.legend(facecolor="#252535", edgecolor="#555566", labelcolor="white")
        ax2.grid(True, linestyle=":", alpha=0.3, color="gray")

        # 3. Decay Half-Life vs Jitter
        ax3 = axes[1, 0]
        df_decay = results["decay_sweep"]
        ax3.plot(df_decay["half_life_min"], df_decay["h1_mean_jump"], "s-.", color="#FFB800", label="H1 (Static)")
        ax3.plot(df_decay["half_life_min"], df_decay["h2_mean_jump"], "^-", color="#00FFFF", linewidth=2, label="H2 (Motion-Comp)")
        ax3.set_title("Directional Stability vs Temporal Half-Life", color="white", fontsize=11)
        ax3.set_xlabel("Evidence Half-Life (minutes)", color="lightgray")
        ax3.set_ylabel("Mean Consecutive Jump (deg)", color="lightgray")
        ax3.legend(facecolor="#252535", edgecolor="#555566", labelcolor="white")
        ax3.grid(True, linestyle=":", alpha=0.3, color="gray")

        # 4. Outward Speed vs Jitter & Persistence
        ax4 = axes[1, 1]
        df_speed = results["speed_sweep"]
        ax4_twin = ax4.twinx()
        l1 = ax4.plot(df_speed["radial_speed_px_30m"], df_speed["h2_mean_jump"], "^-", color="#00FFFF", linewidth=2, label="H2 Jitter (deg)")
        l2 = ax4_twin.plot(df_speed["radial_speed_px_30m"], df_speed["h2_mean_persistence"], "o-", color="#39FF14", linewidth=2, label="H2 Persistence (frames)")
        ax4.set_title("H2 Performance vs Outward Radial Speed Parameter", color="white", fontsize=11)
        ax4.set_xlabel("Radial Speed (pixels / 30 min)", color="lightgray")
        ax4.set_ylabel("Mean Consecutive Jump (deg)", color="#00FFFF")
        ax4_twin.set_ylabel("Mean Persistence (frames)", color="#39FF14")
        ax4.tick_params(colors="#00FFFF")
        ax4_twin.tick_params(colors="#39FF14")
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="upper right", facecolor="#252535", edgecolor="#555566", labelcolor="white")
        ax4.grid(True, linestyle=":", alpha=0.3, color="gray")

        plt.tight_layout()
        out_fig = self.vis_dir / "parameter_sensitivity_summary.png"
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"Saved parameter sensitivity figure to: {out_fig}")


# ==============================================================================
# 6. CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stereo_COR2 HTM Temporal Evidence Pilot")
    parser.add_argument("--detection-dir", default="runs/clean_cor2a_verify/detection", help="Path to detection directory")
    parser.add_argument("--output-dir", default="experiments/htm_pixel_temporal", help="Output directory")
    parser.add_argument("--tau", type=float, default=3.0, help="Softmax temperature parameter")
    parser.add_argument("--mode", default="softmax_z", choices=["softmax_z", "softmax_zpos", "linear_zpos", "clipped_softmax"])
    parser.add_argument("--half-life", type=float, default=45.0, help="Evidence half-life in minutes")
    parser.add_argument("--radial-speed", type=float, default=8.0, help="Radial propagation speed (px / 30 min)")
    parser.add_argument("--all-frames", action="store_true", default=True, help="Generate diagnostic panels for all evaluation frames")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip parameter sensitivity sweeps")

    args = parser.parse_args()

    runner = ExperimentRunner(
        detection_dir=args.detection_dir,
        output_dir=args.output_dir,
    )

    # 1. Run baseline
    df_results = runner.run_baseline_pipeline(
        tau=args.tau,
        mode=args.mode,
        half_life_min=args.half_life,
        radial_speed_px_30m=args.radial_speed,
        generate_all_frame_plots=args.all_frames,
    )

    # 2. Run sensitivity sweeps
    if not args.skip_sensitivity:
        sensitivity_results = runner.run_parameter_sensitivity()

    print("\n========================================================")
    print("Stereo_COR2 HTM Temporal Evidence Pilot Completed Successfully.")
    print(f"Results located in: {runner.output_dir}")
    print("========================================================")


if __name__ == "__main__":
    main()
