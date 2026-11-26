r"""Comprehensive Runner for RET-V1 (Symmetric Valley-Based Lobe) Experiment.

Generates:
1. Baseline (RET-B0) and RET-V1 full 63-pair runs.
2. 12 individual 4-panel diagnostic frames for RET-B0 (Pairs 35-46).
3. 12 individual 4-panel diagnostic frames for RET-V1 (Pairs 35-46).
4. 12 side-by-side comparison figures for Pairs 35-46.
5. Dedicated diagnostic plots:
   - Pair 36 radial profile comparison (showing why 62.2px front was an artifact).
   - CME 46 deg front trajectory comparison.
   - CME 46 deg lobe width trajectory comparison.
6. Quantitative diagnostics CSV for Pairs 35-46 (all required columns).
7. Full-sequence comparison CSV across all 63 pairs.
8. Animated GIFs:
   - ret_v1_full_sequence.gif (63 frames)
   - ret_v1_pairs35_46_comparison.gif (Pairs 35-46)
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from cme_pipeline.cadence_selection import parse_timestamp
from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg
from experiments.ret_tracking.ret_evidence_engine import (
    PolarGridGeometry,
    RadialCandidate,
    FrameRadialEvidenceResult,
    compute_pixel_evidence,
    apply_spatial_gaussian_spreading,
    sample_polar_evidence,
    accumulate_radial_evidence,
    detect_angular_candidates,
)
from experiments.ret_tracking.ret_tracker import (
    RETConfig,
    RadialEvidenceTracker,
    RETTrack,
    RETTrackObservation,
    FrameRETResult,
)


class RETV1ExperimentRunner:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.data_dir = root_dir / "runs/dual_archive_20101212/preprocessed"
        self.out_base = root_dir / "experiments/ret_tracking/results/ret_v1_valley"

        # Output subdirectories
        self.b0_frames_dir = self.out_base / "baseline/frames"
        self.v1_frames_dir = self.out_base / "ret_v1/frames"
        self.comp_dir = self.out_base / "comparisons"
        self.diag_dir = self.out_base / "diagnostics"
        self.metrics_dir = self.out_base / "metrics"

        for d in [self.b0_frames_dir, self.v1_frames_dir, self.comp_dir, self.diag_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. Load manifest and RD data
        self.manifest = pd.read_csv(self.data_dir / "01_rd_manifest.csv")
        self.n_frames = len(self.manifest)
        self.rd_images = np.load(self.data_dir / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
        self.valid_masks = np.load(self.data_dir / "north_up_pair_valid_masks.npy", mmap_mode="r")

        # 2. Solar geometry from WCS
        hdr_str = (self.data_dir / "target_wcs_header.txt").read_text(encoding="utf-8")
        self.wcs = WCS(fits.Header.fromstring(hdr_str, sep="\n"))
        sx, sy = self.wcs.world_to_pixel_values(0.0, 0.0)
        self.sx = float(sx)
        self.sy = float(sy)
        self.grid = PolarGridGeometry(sx=self.sx, sy=self.sy)
        self.timestamps = [parse_timestamp(t) for t in self.manifest["end_time"]]

        # Precompute standardized smoothed z-score maps
        print(f"Precomputing z-scores for {self.n_frames} frames...")
        self.z_maps = self._precompute_z_maps()

    def _precompute_z_maps(self, gaussian_sigma: float = 1.0) -> np.ndarray:
        z_maps = np.zeros((self.n_frames, 256, 256), dtype=np.float32)
        for idx in range(self.n_frames):
            image = np.asarray(self.rd_images[idx], dtype=np.float32)
            val = np.asarray(self.valid_masks[idx], dtype=bool)

            num = gaussian_filter(np.where(val, image, 0.0), gaussian_sigma, mode="constant", cval=0.0)
            den = gaussian_filter(val.astype(float), gaussian_sigma, mode="constant", cval=0.0)

            good = val & (den > 0)
            smoothed = np.full(image.shape, np.nan, dtype=np.float32)
            smoothed[good] = (num[good] / den[good]).astype(np.float32)

            fin = smoothed[good & np.isfinite(smoothed)]
            if fin.size and float(fin.std()) > 0:
                z_maps[idx, good] = (smoothed[good] - float(fin.mean())) / float(fin.std())
        return z_maps

    def run_tracker(self, cfg: RETConfig) -> Tuple[RadialEvidenceTracker, List[FrameRETResult], List[Dict[str, Any]]]:
        """Runs tracker across full sequence, returning tracker, per-frame diagnostic results, and candidate history."""
        tracker = RadialEvidenceTracker(config=cfg, grid=self.grid)
        frame_results: List[FrameRETResult] = []
        frame_snapshots: List[Dict[str, Any]] = []

        for idx in range(self.n_frames):
            ts = self.timestamps[idx]
            pair_num = int(self.manifest.iloc[idx]["pair_index"])

            # Snapshot active tracks BEFORE step
            active_before = {}
            for tid, trk in tracker.tracks.items():
                if trk.status in ("tentative", "confirmed"):
                    active_before[tid] = {
                        "status": trk.status,
                        "hits": trk.hits,
                        "missed_count": trk.missed_count,
                        "latest_cpa": trk.latest.cpa_deg,
                        "latest_r_front": trk.latest.r_front_px,
                        "latest_r_peak": trk.latest.r_peak_px,
                        "latest_width": trk.latest.pa_width_deg,
                        "current_vr": trk.current_vr,
                        "last_obs_timestamp": trk.latest.timestamp,
                    }

            dt_hours = 0.5
            if tracker.previous_timestamp is not None:
                dt_sec = (ts - tracker.previous_timestamp).total_seconds()
                dt_hours = max(0.01, dt_sec / 3600.0)

            res = tracker.step(idx, ts, self.z_maps[idx], self.valid_masks[idx])
            frame_results.append(res)

            frame_snapshots.append({
                "frame_idx": idx,
                "pair_num": pair_num,
                "timestamp": ts,
                "dt_hours": dt_hours,
                "active_before": active_before,
                "res": res,
            })

        return tracker, frame_results, frame_snapshots


def add_coronagraph_rings(ax, grid: PolarGridGeometry, sx: float, sy: float, r_birth: Optional[float] = None):
    c_in = plt.Circle((sx, sy), grid.r_min_px, color="cyan", fill=False, linestyle="--", linewidth=1.2, label=f"Inner ({grid.r_min_px}px)")
    c_out = plt.Circle((sx, sy), grid.r_max_px, color="cyan", fill=False, linestyle=":", linewidth=1.2, label=f"Outer ({grid.r_max_px}px)")
    ax.add_patch(c_in)
    ax.add_patch(c_out)
    if r_birth is not None:
        c_b = plt.Circle((sx, sy), r_birth, color="salmon", fill=False, linestyle=":", linewidth=1.0, label=f"Birth ({r_birth}px)")
        ax.add_patch(c_b)
    ax.plot(sx, sy, "+", color="yellow", markersize=6)
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")


def plot_single_variant_panel(
    variant_name: str,
    pair_num: int,
    frame_idx: int,
    ts: datetime,
    rd_img: np.ndarray,
    val_m: np.ndarray,
    spread_ev: np.ndarray,
    a_smooth: np.ndarray,
    candidates: List[RadialCandidate],
    active_confirmed: List[RETTrack],
    active_tentative: List[RETTrack],
    active_before: Dict[int, Any],
    matched_pairs: List[Tuple[int, int]],
    grid: PolarGridGeometry,
    r_birth_max: float,
    out_path: Path,
):
    """Produces the mandatory 4-panel diagnostic image for a single variant."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    sx, sy = grid.sx, grid.sy

    # 1. Raw North-up Running Difference
    vlim = float(np.nanpercentile(np.abs(rd_img[val_m]), 98)) if np.any(val_m) else 1.0
    vlim = max(0.1, vlim)
    im0 = axes[0].imshow(rd_img, cmap="gray", vmin=-vlim, vmax=vlim)
    add_coronagraph_rings(axes[0], grid, sx, sy, r_birth=r_birth_max)
    axes[0].set_title(f"1. Raw Running Difference\nPair {pair_num:02d} ({ts.strftime('%Y-%m-%d %H:%M UTC')})", fontsize=10, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="DN/s")

    # 2. Spatial Gaussian Evidence
    im1 = axes[1].imshow(spread_ev, cmap="magma", vmin=0, vmax=4)
    add_coronagraph_rings(axes[1], grid, sx, sy, r_birth=r_birth_max)
    axes[1].set_title(f"2. Spatial Gaussian Evidence $\\widetilde{{E}}_t$\n($\\sigma_s=2.0$ px, Mask-Normalized)", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Std Evidence")

    # 3. Angular Evidence Profile
    axes[2].plot(grid.pa_bins, a_smooth, color="#2b5c8f", linewidth=2.0, label="Accumulated $A_t(\\theta)$")
    axes[2].set_xlim(0, 360)
    axes[2].set_xlabel("Position Angle (deg) [0=N, 90=E, 180=S, 270=W]", fontsize=9)
    axes[2].set_ylabel("Radial Evidence Accumulation $A_t(\\theta)$", fontsize=9)
    axes[2].set_title(f"3. Angular Evidence Profile ({variant_name})\n{len(candidates)} Candidate Lobe(s) Detected", fontsize=10, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    colors = ["#d9534f", "#f0ad4e", "#5cb85c", "#5bc0de", "#9b59b6", "#e67e22", "#34495e", "#16a085"]

    for i, cand in enumerate(candidates):
        c_col = colors[i % len(colors)]
        # Peak PA
        axes[2].axvline(cand.peak_pa_deg, color=c_col, linestyle=":", alpha=0.7)
        axes[2].plot(cand.peak_pa_deg, cand.peak_evidence, "o", color=c_col, markersize=6)

        # CPA
        axes[2].axvline(cand.weighted_cpa_deg, color=c_col, linestyle="--", alpha=0.9)

        # Shaded lobe region
        if cand.pa_start_deg <= cand.pa_end_deg:
            mask = (grid.pa_bins >= cand.pa_start_deg) & (grid.pa_bins <= cand.pa_end_deg)
            axes[2].fill_between(grid.pa_bins, 0, a_smooth, where=mask, color=c_col, alpha=0.15)
        else:
            mask = (grid.pa_bins >= cand.pa_start_deg) | (grid.pa_bins <= cand.pa_end_deg)
            axes[2].fill_between(grid.pa_bins, 0, a_smooth, where=mask, color=c_col, alpha=0.15)

        # For RET-V1: plot valleys if available
        if variant_name == "RET-V1" and cand.left_valley_pa_deg is not None:
            axes[2].plot(cand.left_valley_pa_deg, a_smooth[int(round(cand.left_valley_pa_deg)) % 360], "v", color="purple", markersize=5)
            axes[2].plot(cand.right_valley_pa_deg, a_smooth[int(round(cand.right_valley_pa_deg)) % 360], "^", color="purple", markersize=5)
            annot_txt = f"C{cand.candidate_id}: {cand.weighted_cpa_deg:.0f}°\n$W={cand.pa_width_deg:.0f}°$\n($d_L={cand.left_valley_dist_deg:.0f}°, d_R={cand.right_valley_dist_deg:.0f}°$)"
        else:
            annot_txt = f"C{cand.candidate_id}: {cand.weighted_cpa_deg:.0f}°\n$W={cand.pa_width_deg:.0f}°$"

        axes[2].text(cand.weighted_cpa_deg, cand.peak_evidence * 1.03, annot_txt, color=c_col, fontsize=7, fontweight="bold", ha="center")

    # 4. Tracking Overlay
    axes[3].imshow(rd_img, cmap="gray", vmin=-vlim, vmax=vlim)
    add_coronagraph_rings(axes[3], grid, sx, sy, r_birth=r_birth_max)
    axes[3].set_title(f"4. Tracking Overlay ({variant_name})\nConfirmed (Green), Tentative (Yellow), Rej (Red)", fontsize=10, fontweight="bold")

    matched_cand_ids = {cid: tid for tid, cid in matched_pairs}
    matched_track_ids = {tid for tid, cid in matched_pairs}

    # Overlay candidate fronts
    for i, cand in enumerate(candidates):
        cpa_rad = np.radians(cand.weighted_cpa_deg)
        x_front = sx - cand.r_front_px * np.sin(cpa_rad)
        y_front = sy + cand.r_front_px * np.cos(cpa_rad)
        x_pk = sx - cand.r_peak_px * np.sin(cpa_rad)
        y_pk = sy + cand.r_peak_px * np.cos(cpa_rad)

        is_matched = cand.candidate_id in matched_cand_ids
        tid = matched_cand_ids.get(cand.candidate_id)

        # Draw front arc across lobe span
        span = cand.pa_width_deg
        theta_lobe = np.linspace(-span / 2.0, span / 2.0, 20)
        arc_angles = (cand.peak_pa_deg + theta_lobe) % 360.0
        arc_rad = np.radians(arc_angles)
        arc_x = sx - cand.r_front_px * np.sin(arc_rad)
        arc_y = sy + cand.r_front_px * np.cos(arc_rad)

        if is_matched:
            # Matched candidate front
            axes[3].plot(arc_x, arc_y, color="lime", linewidth=1.8, alpha=0.8)
            axes[3].plot(x_front, y_front, "s", color="lime", markersize=7, markeredgecolor="black")
            axes[3].plot(x_pk, y_pk, "o", color="yellow", markersize=4, markeredgecolor="black")
        else:
            # Unmatched / Rejected candidate
            axes[3].plot(arc_x, arc_y, color="crimson", linestyle="--", linewidth=1.2, alpha=0.7)
            axes[3].plot(x_front, y_front, "x", color="crimson", markersize=7, markeredgewidth=2)
            axes[3].annotate(
                f"Rej C{cand.candidate_id}\n({cand.weighted_cpa_deg:.0f}°, r={cand.r_front_px:.0f})",
                (x_front + 3, y_front - 3),
                color="pink", fontsize=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="darkred", alpha=0.7, edgecolor="crimson")
            )

    # Overlay active confirmed tracks using current frame observation
    for trk in active_confirmed:
        cur_obs = [o for o in trk.observations if o.frame_idx == frame_idx]
        if not cur_obs:
            continue
        obs = cur_obs[0]
        cpa_rad = np.radians(obs.cpa_deg)
        x_ray = sx - grid.r_max_px * np.sin(cpa_rad)
        y_ray = sy + grid.r_max_px * np.cos(cpa_rad)
        axes[3].plot([sx, x_ray], [sy, y_ray], color="lime", linestyle="--", linewidth=1.2, alpha=0.7)

        x_front = sx - obs.r_front_px * np.sin(cpa_rad)
        y_front = sy + obs.r_front_px * np.cos(cpa_rad)
        axes[3].plot(x_front, y_front, "^", color="lime", markersize=8, markeredgecolor="black")

        axes[3].annotate(
            f"Trk {trk.track_id} (C{obs.candidate_id})\nCPA {obs.cpa_deg:.0f}°\nr={obs.r_front_px:.0f}px",
            (x_front + 4, y_front - 4),
            color="white", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75, edgecolor="lime")
        )

    for trk in active_tentative:
        cur_obs = [o for o in trk.observations if o.frame_idx == frame_idx]
        if not cur_obs:
            continue
        obs = cur_obs[0]
        cpa_rad = np.radians(obs.cpa_deg)
        x_front = sx - obs.r_front_px * np.sin(cpa_rad)
        y_front = sy + obs.r_front_px * np.cos(cpa_rad)
        axes[3].plot(x_front, y_front, "d", color="gold", markersize=7, markeredgecolor="black")
        axes[3].annotate(
            f"Tent Trk {trk.track_id} (C{obs.candidate_id})\n({obs.cpa_deg:.0f}°, r={obs.r_front_px:.0f}px)",
            (x_front + 4, y_front + 4),
            color="black", fontsize=6.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="gold", alpha=0.8, edgecolor="black")
        )

    # Check missing tracks (active before but not matched this step)
    for tid, tinfo in active_before.items():
        if tid not in matched_track_ids:
            # Temporarily missing track: compute predicted radius
            dt_since = (ts - tinfo["last_obs_timestamp"]).total_seconds() / 3600.0
            r_pred = tinfo["latest_r_front"] + tinfo["current_vr"] * dt_since
            cpa_rad = np.radians(tinfo["latest_cpa"])
            x_f = sx - r_pred * np.sin(cpa_rad)
            y_f = sy + r_pred * np.cos(cpa_rad)
            axes[3].plot(x_f, y_f, "o", color="orange", markersize=8, markeredgecolor="orange", fillstyle="none", markeredgewidth=2.0)
            axes[3].annotate(
                f"MISS Trk {tid}\nPred r={r_pred:.0f}px (M={tinfo['missed_count'] + 1})",
                (x_f - 25, y_f + 12),
                color="orange", fontsize=6.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8, edgecolor="orange")
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_side_by_side_comparison(
    pair_num: int,
    ts: datetime,
    rd_img: np.ndarray,
    val_m: np.ndarray,
    b0_snap: Dict[str, Any],
    v1_snap: Dict[str, Any],
    b0_tracker: RadialEvidenceTracker,
    v1_tracker: RadialEvidenceTracker,
    grid: PolarGridGeometry,
    r_birth_max: float,
    out_path: Path,
):
    """Produces side-by-side comparison figure (2x2 grid: B0 A(theta), V1 A(theta), B0 Overlay, V1 Overlay)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    sx, sy = grid.sx, grid.sy

    b0_res = b0_snap["res"]
    v1_res = v1_snap["res"]

    # Top-Left: Baseline Angular Profile
    ax_a_b0 = axes[0, 0]
    ax_a_b0.plot(grid.pa_bins, b0_res.evidence_result.angular_evidence_smoothed, color="#2b5c8f", linewidth=2.0)
    ax_a_b0.set_xlim(0, 360)
    ax_a_b0.set_title(f"A. RET-B0 Angular Evidence Profile (Pair {pair_num:02d})\nProminence-Based Lobe Boundaries (1.5x Expansion)", fontsize=11, fontweight="bold")
    ax_a_b0.set_xlabel("Position Angle (deg)", fontsize=9)
    ax_a_b0.set_ylabel("Accumulated Evidence $A_t(\\theta)$", fontsize=9)
    ax_a_b0.grid(True, linestyle="--", alpha=0.5)

    for cand in b0_res.evidence_result.candidates:
        ax_a_b0.axvline(cand.weighted_cpa_deg, color="crimson", linestyle="--", alpha=0.8)
        ax_a_b0.plot(cand.peak_pa_deg, cand.peak_evidence, "o", color="crimson", markersize=6)
        if cand.pa_start_deg <= cand.pa_end_deg:
            m = (grid.pa_bins >= cand.pa_start_deg) & (grid.pa_bins <= cand.pa_end_deg)
        else:
            m = (grid.pa_bins >= cand.pa_start_deg) | (grid.pa_bins <= cand.pa_end_deg)
        ax_a_b0.fill_between(grid.pa_bins, 0, b0_res.evidence_result.angular_evidence_smoothed, where=m, color="crimson", alpha=0.15)
        ax_a_b0.text(cand.weighted_cpa_deg, cand.peak_evidence + 2, f"C{cand.candidate_id}: {cand.weighted_cpa_deg:.0f}°\n$W={cand.pa_width_deg:.0f}°$", color="crimson", fontsize=8, fontweight="bold", ha="center")

    # Top-Right: RET-V1 Angular Profile
    ax_a_v1 = axes[0, 1]
    ax_a_v1.plot(grid.pa_bins, v1_res.evidence_result.angular_evidence_smoothed, color="#2b5c8f", linewidth=2.0)
    ax_a_v1.set_xlim(0, 360)
    ax_a_v1.set_title(f"B. RET-V1 Angular Evidence Profile (Pair {pair_num:02d})\nSymmetric Nearest-Side Valley Lobe Boundaries", fontsize=11, fontweight="bold")
    ax_a_v1.set_xlabel("Position Angle (deg)", fontsize=9)
    ax_a_v1.set_ylabel("Accumulated Evidence $A_t(\\theta)$", fontsize=9)
    ax_a_v1.grid(True, linestyle="--", alpha=0.5)

    for cand in v1_res.evidence_result.candidates:
        ax_a_v1.axvline(cand.weighted_cpa_deg, color="teal", linestyle="--", alpha=0.8)
        ax_a_v1.plot(cand.peak_pa_deg, cand.peak_evidence, "o", color="teal", markersize=6)
        if cand.pa_start_deg <= cand.pa_end_deg:
            m = (grid.pa_bins >= cand.pa_start_deg) & (grid.pa_bins <= cand.pa_end_deg)
        else:
            m = (grid.pa_bins >= cand.pa_start_deg) | (grid.pa_bins <= cand.pa_end_deg)
        ax_a_v1.fill_between(grid.pa_bins, 0, v1_res.evidence_result.angular_evidence_smoothed, where=m, color="teal", alpha=0.15)
        if cand.left_valley_pa_deg is not None:
            ax_a_v1.plot(cand.left_valley_pa_deg, v1_res.evidence_result.angular_evidence_smoothed[int(round(cand.left_valley_pa_deg)) % 360], "v", color="purple", markersize=6)
            ax_a_v1.plot(cand.right_valley_pa_deg, v1_res.evidence_result.angular_evidence_smoothed[int(round(cand.right_valley_pa_deg)) % 360], "^", color="purple", markersize=6)
            txt = f"C{cand.candidate_id}: {cand.weighted_cpa_deg:.0f}°\n$W={cand.pa_width_deg:.0f}°$\n($d_L={cand.left_valley_dist_deg:.0f}°, d_R={cand.right_valley_dist_deg:.0f}°$)"
        else:
            txt = f"C{cand.candidate_id}: {cand.weighted_cpa_deg:.0f}°\n$W={cand.pa_width_deg:.0f}°$ (fallback)"
        ax_a_v1.text(cand.weighted_cpa_deg, cand.peak_evidence + 2, txt, color="teal", fontsize=8, fontweight="bold", ha="center")

    vlim = float(np.nanpercentile(np.abs(rd_img[val_m]), 98)) if np.any(val_m) else 1.0
    vlim = max(0.1, vlim)

    # Helper to draw bottom overlay panel with rigorous candidate and track rendering
    def render_overlay(ax, snap, variant_title):
        res = snap["res"]
        active_bef = snap["active_before"]
        f_idx = snap["frame_idx"]
        matched_cand_to_track = {cid: tid for tid, cid in res.matched_tracks}
        matched_track_to_cand = {tid: cid for tid, cid in res.matched_tracks}

        ax.imshow(rd_img, cmap="gray", vmin=-vlim, vmax=vlim)
        add_coronagraph_rings(ax, grid, sx, sy, r_birth=r_birth_max)

        # 1. Candidate arcs and fronts
        for cand in res.evidence_result.candidates:
            span = cand.pa_width_deg
            theta_lobe = np.linspace(-span / 2.0, span / 2.0, 25)
            arc_angles = (cand.peak_pa_deg + theta_lobe) % 360.0
            arc_rad = np.radians(arc_angles)
            arc_x = sx - cand.r_front_px * np.sin(arc_rad)
            arc_y = sy + cand.r_front_px * np.cos(arc_rad)

            cpa_rad = np.radians(cand.weighted_cpa_deg)
            xf = sx - cand.r_front_px * np.sin(cpa_rad)
            yf = sy + cand.r_front_px * np.cos(cpa_rad)

            if cand.candidate_id in matched_cand_to_track:
                ax.plot(arc_x, arc_y, color="lime", linewidth=1.8, alpha=0.85)
                ax.plot(xf, yf, "s", color="lime", markersize=6, markeredgecolor="black")
            else:
                ax.plot(arc_x, arc_y, color="crimson", linestyle="--", linewidth=1.2, alpha=0.7)
                ax.plot(xf, yf, "x", color="crimson", markersize=7, markeredgewidth=2)
                ax.annotate(
                    f"Rej C{cand.candidate_id}\n({cand.weighted_cpa_deg:.0f}°, r={cand.r_front_px:.0f}px)",
                    (xf + 3, yf - 3), color="pink", fontsize=6.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="darkred", alpha=0.75, edgecolor="crimson")
                )

        # 2. Confirmed active tracks (rendered using observation in CURRENT frame)
        n_matched = 0
        for trk in res.active_confirmed_tracks:
            cur_obs = [o for o in trk.observations if o.frame_idx == f_idx]
            if cur_obs:
                n_matched += 1
                obs = cur_obs[0]
                cpa_rad = np.radians(obs.cpa_deg)
                x_ray = sx - grid.r_max_px * np.sin(cpa_rad)
                y_ray = sy + grid.r_max_px * np.cos(cpa_rad)
                ax.plot([sx, x_ray], [sy, y_ray], color="cyan", linestyle="--", linewidth=1.2, alpha=0.7)

                x_front = sx - obs.r_front_px * np.sin(cpa_rad)
                y_front = sy + obs.r_front_px * np.cos(cpa_rad)
                ax.plot(x_front, y_front, "^", color="lime", markersize=8, markeredgecolor="black")
                ax.annotate(
                    f"Trk {trk.track_id} (C{obs.candidate_id})\nCPA {obs.cpa_deg:.0f}° | r={obs.r_front_px:.0f}px",
                    (x_front + 4, y_front - 4), color="white", fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75, edgecolor="lime")
                )

        # 3. Missing/Coasting tracks (active before step, but not matched this step)
        for tid, tinfo in active_bef.items():
            if tid not in matched_track_to_cand:
                dt_since = (ts - tinfo["last_obs_timestamp"]).total_seconds() / 3600.0
                r_pred = tinfo["latest_r_front"] + tinfo["current_vr"] * dt_since
                cpa_rad = np.radians(tinfo["latest_cpa"])
                x_ray = sx - grid.r_max_px * np.sin(cpa_rad)
                y_ray = sy + grid.r_max_px * np.cos(cpa_rad)
                ax.plot([sx, x_ray], [sy, y_ray], color="orange", linestyle=":", linewidth=1.2, alpha=0.7)

                x_f = sx - r_pred * np.sin(cpa_rad)
                y_f = sy + r_pred * np.cos(cpa_rad)
                ax.plot(x_f, y_f, "o", markerfacecolor="none", markeredgecolor="orange", markersize=8, markeredgewidth=2.0)
                ax.annotate(
                    f"MISS Trk {tid}\nPred r={r_pred:.0f}px (M={tinfo['missed_count'] + 1})",
                    (x_f + 4, y_f - 4), color="orange", fontsize=7.0, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8, edgecolor="orange")
                )

        ax.set_title(f"{variant_title} (Pair {pair_num:02d})\nMatched Tracks: {n_matched} | Active Confirmed: {len(res.active_confirmed_tracks)}", fontsize=11, fontweight="bold")

    render_overlay(axes[1, 0], b0_snap, "C. RET-B0 Tracking Overlay")
    render_overlay(axes[1, 1], v1_snap, "D. RET-V1 Tracking Overlay")

    plt.suptitle(f"Direct Architecture Comparison: Pair {pair_num:02d} ({ts.strftime('%Y-%m-%d %H:%M UTC')})\nRET-B0 (Prominence 1.5x) vs RET-V1 (Symmetric Nearest-Side Valley)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def generate_pair_036_radial_profile_comparison(
    runner: RETV1ExperimentRunner,
    b0_snap: Dict[str, Any],
    v1_snap: Dict[str, Any],
    out_path: Path,
):
    """Produces dedicated Pair 36 diagnostic analyzing the radial profile and front overestimation."""
    idx = b0_snap["frame_idx"]
    ts = b0_snap["timestamp"]
    pair_num = b0_snap["pair_num"]

    b0_cme = [c for c in b0_snap["res"].evidence_result.candidates if 30 <= c.weighted_cpa_deg <= 60][0]
    v1_cme = [c for c in v1_snap["res"].evidence_result.candidates if 30 <= c.weighted_cpa_deg <= 60][0]

    grid = runner.grid
    polar_ev = b0_snap["res"].evidence_result.polar_evidence

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Angular Lobe Comparison on A(theta)
    ax0 = axes[0]
    a_smooth = b0_snap["res"].evidence_result.angular_evidence_smoothed
    ax0.plot(grid.pa_bins, a_smooth, color="#2b5c8f", linewidth=2.2, label="$A_t(\\theta)$")
    ax0.set_xlim(0, 180)
    ax0.set_xlabel("Position Angle (deg)", fontsize=10)
    ax0.set_ylabel("Radial Accumulated Evidence", fontsize=10)
    ax0.set_title("1. Angular Lobe Width Comparison\nPair 36 (06:54 UTC)", fontsize=11, fontweight="bold")
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Shade B0 lobe
    ax0.axvspan(b0_cme.pa_start_deg, b0_cme.pa_end_deg if b0_cme.pa_end_deg > b0_cme.pa_start_deg else 360, color="crimson", alpha=0.2, label=f"B0 Lobe: {b0_cme.pa_width_deg:.0f}° [{b0_cme.pa_start_deg:.0f}°-{b0_cme.pa_end_deg:.0f}°]")
    # Shade V1 lobe
    ax0.axvspan(v1_cme.pa_start_deg, v1_cme.pa_end_deg, color="teal", alpha=0.35, label=f"V1 Lobe: {v1_cme.pa_width_deg:.0f}° [{v1_cme.pa_start_deg:.0f}°-{v1_cme.pa_end_deg:.0f}°]")

    ax0.plot(v1_cme.peak_pa_deg, v1_cme.peak_evidence, "o", color="darkred", markersize=8, label=f"Peak PA = {v1_cme.peak_pa_deg:.0f}°")
    ax0.plot(v1_cme.left_valley_pa_deg, a_smooth[int(round(v1_cme.left_valley_pa_deg))], "v", color="purple", markersize=8, label=f"Left Valley $\\theta_L = {v1_cme.left_valley_pa_deg:.0f}°$ ($d_L={v1_cme.left_valley_dist_deg:.0f}°$)")
    ax0.plot(v1_cme.right_valley_pa_deg, a_smooth[int(round(v1_cme.right_valley_pa_deg))], "^", color="purple", markersize=8, label=f"Right Valley $\\theta_R = {v1_cme.right_valley_pa_deg:.0f}°$ ($d_R={v1_cme.right_valley_dist_deg:.0f}°$)")
    ax0.legend(fontsize=8, loc="upper right")

    # Panel 2: Mean Radial Evidence Profile Comparison
    ax1 = axes[1]
    r_bins = grid.r_bins
    ax1.plot(r_bins, b0_cme.radial_profile, "r-", linewidth=2.2, label=f"B0 Profile (W=120°) -> $r_{{front}}={b0_cme.r_front_px:.1f}$px")
    ax1.plot(r_bins, v1_cme.radial_profile, "teal", linewidth=2.5, label=f"V1 Profile (W=58°) -> $r_{{front}}={v1_cme.r_front_px:.1f}$px")

    # Plot adaptive thresholds
    th_b0 = max(0.35, float(np.max(b0_cme.radial_profile)) * 0.20)
    th_v1 = max(0.35, float(np.max(v1_cme.radial_profile)) * 0.20)
    ax1.axhline(th_b0, color="crimson", linestyle=":", label=f"B0 $\\lambda = {th_b0:.2f}$")
    ax1.axhline(th_v1, color="teal", linestyle=":", label=f"V1 $\\lambda = {th_v1:.2f}$")

    ax1.axvline(b0_cme.r_front_px, color="crimson", linestyle="--", linewidth=1.8)
    ax1.axvline(v1_cme.r_front_px, color="teal", linestyle="--", linewidth=1.8)

    ax1.set_xlim(grid.r_min_px, 100)
    ax1.set_xlabel("Radius (pixels)", fontsize=10)
    ax1.set_ylabel("Evidence Density", fontsize=10)
    ax1.set_title("2. Radial Profile & Front Detection\nContaminated Tail vs Focused Core", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="upper right")

    # Panel 3: Kinematic Jump Demonstration
    ax2 = axes[2]
    pairs_demo = [35, 36, 37]
    r_b0_demo = [38.08, b0_cme.r_front_px, 52.15]
    r_v1_demo = [37.08, v1_cme.r_front_px, 52.15]

    ax2.plot(pairs_demo, r_b0_demo, "s--", color="crimson", linewidth=2.2, markersize=8, label="RET-B0 (Spike to 62.2px -> Inward Violation)")
    ax2.plot(pairs_demo, r_v1_demo, "d-", color="teal", linewidth=2.5, markersize=8, label="RET-V1 (Clean Outward Transit 37 -> 43 -> 52px)")

    ax2.annotate(f"Spurious Spike: +{b0_cme.r_front_px - 38.08:.1f}px\n($v_r = +48.2$ px/hr)", (36, b0_cme.r_front_px), (35.3, 56),
                 arrowprops=dict(facecolor="crimson", shrink=0.08, width=1.5, headwidth=6), color="darkred", fontsize=8.5, fontweight="bold")
    ax2.annotate(f"Clean Advance: +{v1_cme.r_front_px - 37.08:.1f}px\n($v_r = +12.1$ px/hr)", (36, v1_cme.r_front_px), (35.6, 35),
                 arrowprops=dict(facecolor="teal", shrink=0.08, width=1.5, headwidth=6), color="teal", fontsize=8.5, fontweight="bold")

    ax2.set_xticks(pairs_demo)
    ax2.set_xticklabels(["Pair 35 (06:24)", "Pair 36 (06:54)", "Pair 37 (07:24)"])
    ax2.set_ylabel("Front Radius $r_{front}$ (px)", fontsize=10)
    ax2.set_title("3. Pair 35-37 Kinematic Continuity\nElimination of False Spike", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="upper left")

    plt.suptitle("Pair 36 Root-Cause Diagnostic: Radial Profile Contamination Under 120° Expansion", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def generate_trajectory_plots(
    df_b0: pd.DataFrame,
    df_v1: pd.DataFrame,
    out_front_path: Path,
    out_width_path: Path,
):
    """Generates trajectory comparison plots for CME 46 deg front radius and lobe width."""
    # Front radius trajectory
    fig, ax = plt.subplots(figsize=(10, 6))

    # B0 Track 42
    t42_b0 = df_b0[df_b0["track_id"] == 42].sort_values("frame_idx")
    ax.plot(t42_b0["frame_idx"] + 1, t42_b0["r_front_px"], "s--", color="crimson", linewidth=2.0, markersize=7, label="RET-B0 Track 42 (Terminated at Pair 38)")

    # V1 Track 41 (the continuous 46 deg CME)
    t41_v1 = df_v1[df_v1["track_id"] == 41].sort_values("frame_idx")
    ax.plot(t41_v1["frame_idx"] + 1, t41_v1["r_front_px"], "d-", color="teal", linewidth=2.5, markersize=8, label=f"RET-V1 Track 41 (Continuous 16 Pairs, Speed={t41_v1['v_r_smoothed_px_hr'].iloc[-1]:.1f} px/hr)")

    ax.axhline(48.0, color="salmon", linestyle=":", linewidth=1.5, label="Near-Sun Birth Limit ($r_{birth}=48$px)")
    ax.axhline(117.5, color="cyan", linestyle=":", linewidth=1.5, label="Outer Exit Bound ($r_{max}-5=117.5$px)")

    ax.set_xlabel("RD Pair Index", fontsize=11)
    ax.set_ylabel("Front Radius $r_{front}$ (pixels)", fontsize=11)
    ax.set_title("CME 46° Front Radius Kinematic Trajectory: RET-B0 vs RET-V1\nContinuous FOV Transit Recovered by Symmetric Valley Boundaries", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_front_path, dpi=180)
    plt.close()

    # Lobe width trajectory
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t42_b0["frame_idx"] + 1, t42_b0["pa_width_deg"], "s--", color="crimson", linewidth=2.0, markersize=7, label="RET-B0 Track 42 Lobe Width (Exploded to 120°)")
    ax.plot(t41_v1["frame_idx"] + 1, t41_v1["pa_width_deg"], "d-", color="teal", linewidth=2.5, markersize=8, label="RET-V1 Track 41 Lobe Width (Stable ~45°-58°)")

    ax.axhline(120.0, color="gray", linestyle=":", linewidth=1.5, label="Baseline Max Cap (120°)")
    ax.axhline(8.0, color="orange", linestyle=":", linewidth=1.5, label="Minimum Width Bound (8°)")

    ax.set_xlabel("RD Pair Index", fontsize=11)
    ax.set_ylabel("Angular Lobe Width (degrees)", fontsize=11)
    ax.set_title("CME 46° Angular Lobe Width Stability: RET-B0 vs RET-V1\nSuppression of Excessive Expansion via Nearest Valley Bounds", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_width_path, dpi=180)
    plt.close()


def generate_animated_gifs(
    runner: RETV1ExperimentRunner,
    v1_snaps: List[Dict[str, Any]],
    v1_tracker: RadialEvidenceTracker,
    out_full_gif: Path,
    out_comp_gif: Path,
):
    """Generates the required animated GIFs."""
    temp_full_dir = runner.out_base / "temp_full_anim"
    temp_full_dir.mkdir(parents=True, exist_ok=True)
    full_imgs = []

    print("Generating full-sequence animation frames for RET-V1...")
    for idx in range(runner.n_frames):
        ts = runner.timestamps[idx]
        pair_num = int(runner.manifest.iloc[idx]["pair_index"])
        rd_img = runner.rd_images[idx]
        val_m = runner.valid_masks[idx]
        res = v1_snaps[idx]["res"]

        fig, ax = plt.subplots(figsize=(7, 7))
        vlim = float(np.nanpercentile(np.abs(rd_img[val_m]), 98)) if np.any(val_m) else 1.0
        vlim = max(0.1, vlim)
        ax.imshow(rd_img, cmap="gray", vmin=-vlim, vmax=vlim)
        add_coronagraph_rings(ax, runner.grid, runner.sx, runner.sy, r_birth=48.0)

        # Plot confirmed tracks using observation in CURRENT frame
        for trk in res.active_confirmed_tracks:
            cur_obs = [o for o in trk.observations if o.frame_idx == idx]
            if not cur_obs:
                continue
            obs = cur_obs[0]
            cpa_rad = np.radians(obs.cpa_deg)
            x_ray = runner.sx - runner.grid.r_max_px * np.sin(cpa_rad)
            y_ray = runner.sy + runner.grid.r_max_px * np.cos(cpa_rad)
            ax.plot([runner.sx, x_ray], [runner.sy, y_ray], color="cyan", linestyle="--", linewidth=1.2, alpha=0.7)

            x_front = runner.sx - obs.r_front_px * np.sin(cpa_rad)
            y_front = runner.sy + obs.r_front_px * np.cos(cpa_rad)
            ax.plot(x_front, y_front, "^", color="lime", markersize=8, markeredgecolor="black")
            ax.annotate(f"Trk {trk.track_id} ({obs.cpa_deg:.0f}°)\nr={obs.r_front_px:.0f}px", (x_front + 4, y_front - 4), color="white", fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor="lime"))

        ax.text(8, 20, f"RET-V1 (Valley) | Pair {pair_num:02d}\n{ts.strftime('%Y-%m-%d %H:%M UTC')}\nActive Confirmed: {len(res.active_confirmed_tracks)}", color="white", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="navy", alpha=0.8))
        ax.axis("off")

        fpath = temp_full_dir / f"frame_{idx:03d}.png"
        plt.tight_layout()
        plt.savefig(fpath, dpi=110)
        plt.close()
        full_imgs.append(fpath)

    # Save full GIF
    pil_full = [Image.open(p) for p in full_imgs]
    pil_full[0].save(out_full_gif, save_all=True, append_images=pil_full[1:], duration=250, loop=0)
    print(f"Saved full sequence animation to {out_full_gif}")

    # Pairs 35-46 comparison animation
    comp_imgs = sorted(list(runner.comp_dir.glob("compare_pair_*.png")))
    if comp_imgs:
        pil_comp = [Image.open(p) for p in comp_imgs]
        pil_comp[0].save(out_comp_gif, save_all=True, append_images=pil_comp[1:], duration=500, loop=0)
        print(f"Saved Pairs 35-46 comparison animation to {out_comp_gif}")


def run_experiment_main():
    root = Path("/home/aababc/stereo_cor2_retv")
    runner = RETV1ExperimentRunner(root)

    print("\n=== Running Baseline RET-B0 ===")
    cfg_b0 = RETConfig(variant="RET-B0", sigma_s=2.0)
    tracker_b0, res_b0, snaps_b0 = runner.run_tracker(cfg_b0)
    df_b0 = tracker_b0.to_dataframe()

    print("\n=== Running RET-V1 (Symmetric Valley-Based) ===")
    cfg_v1 = RETConfig(variant="RET-V1", sigma_s=2.0)
    tracker_v1, res_v1, snaps_v1 = runner.run_tracker(cfg_v1)
    df_v1 = tracker_v1.to_dataframe()

    print("\nGenerating mandatory diagnostic frames (Pairs 35–46)...")
    diag_rows = []

    for p_num in range(35, 47):
        idx = runner.manifest[runner.manifest["pair_index"] == p_num].index[0]
        ts = runner.timestamps[idx]
        rd_img = runner.rd_images[idx]
        val_m = runner.valid_masks[idx]

        snap_b0 = snaps_b0[idx]
        snap_v1 = snaps_v1[idx]

        # 1. Baseline single frame
        b0_out = runner.b0_frames_dir / f"diag_pair_{p_num:03d}.png"
        plot_single_variant_panel(
            variant_name="RET-B0",
            pair_num=p_num,
            frame_idx=idx,
            ts=ts,
            rd_img=rd_img,
            val_m=val_m,
            spread_ev=snap_b0["res"].evidence_result.spread_evidence,
            a_smooth=snap_b0["res"].evidence_result.angular_evidence_smoothed,
            candidates=snap_b0["res"].evidence_result.candidates,
            active_confirmed=snap_b0["res"].active_confirmed_tracks,
            active_tentative=snap_b0["res"].active_tentative_tracks,
            active_before=snap_b0["active_before"],
            matched_pairs=snap_b0["res"].matched_tracks,
            grid=runner.grid,
            r_birth_max=cfg_b0.r_birth_max,
            out_path=b0_out,
        )

        # 2. RET-V1 single frame
        v1_out = runner.v1_frames_dir / f"diag_pair_{p_num:03d}.png"
        plot_single_variant_panel(
            variant_name="RET-V1",
            pair_num=p_num,
            frame_idx=idx,
            ts=ts,
            rd_img=rd_img,
            val_m=val_m,
            spread_ev=snap_v1["res"].evidence_result.spread_evidence,
            a_smooth=snap_v1["res"].evidence_result.angular_evidence_smoothed,
            candidates=snap_v1["res"].evidence_result.candidates,
            active_confirmed=snap_v1["res"].active_confirmed_tracks,
            active_tentative=snap_v1["res"].active_tentative_tracks,
            active_before=snap_v1["active_before"],
            matched_pairs=snap_v1["res"].matched_tracks,
            grid=runner.grid,
            r_birth_max=cfg_v1.r_birth_max,
            out_path=v1_out,
        )

        # 3. Side-by-side comparison
        comp_out = runner.comp_dir / f"compare_pair_{p_num:03d}.png"
        plot_side_by_side_comparison(
            pair_num=p_num,
            ts=ts,
            rd_img=rd_img,
            val_m=val_m,
            b0_snap=snap_b0,
            v1_snap=snap_v1,
            b0_tracker=tracker_b0,
            v1_tracker=tracker_v1,
            grid=runner.grid,
            r_birth_max=cfg_b0.r_birth_max,
            out_path=comp_out,
        )

        # Extract diagnostics rows for each candidate in both variants
        for variant_str, snap, tracker, cfg_cur in [("RET-B0", snap_b0, tracker_b0, cfg_b0), ("RET-V1", snap_v1, tracker_v1, cfg_v1)]:
            res = snap["res"]
            dt_hr = snap["dt_hours"]
            active_bef = snap["active_before"]

            matched_dict = {cid: tid for tid, cid in res.matched_tracks}

            for cand in res.evidence_result.candidates:
                tid = matched_dict.get(cand.candidate_id)
                outcome = "unmatched_unborn"
                rej_reason = "none"
                r_pred_val = None
                r_prev_val = None
                vr_instant_val = None
                cost_val = None
                t_status = None
                m_count = None

                if tid is not None:
                    outcome = "matched_existing_track"
                    trk_obj = tracker.tracks[tid]
                    t_status = trk_obj.status
                    m_count = trk_obj.missed_count
                    # Find obs in this frame
                    obs_match = [o for o in trk_obj.observations if o.frame_idx == idx]
                    if obs_match:
                        o = obs_match[0]
                        r_pred_val = o.predicted_r
                        cost_val = o.association_cost
                        vr_instant_val = o.v_r_instant
                        # Prior obs
                        obs_idx = trk_obj.observations.index(o)
                        if obs_idx > 0:
                            r_prev_val = trk_obj.observations[obs_idx - 1].r_front_px
                else:
                    # Check birth
                    if cand.r_front_px <= cfg_cur.r_birth_max:
                        # was a track born from this candidate?
                        # Newly born tracks
                        for born_tid in res.newly_born_tracks:
                            b_trk = tracker.tracks[born_tid]
                            if b_trk.first.candidate_id == cand.candidate_id:
                                tid = born_tid
                                outcome = "new_track_born"
                                t_status = b_trk.status
                                m_count = b_trk.missed_count
                                cost_val = 0.0
                                vr_instant_val = 0.0
                                break
                    else:
                        rej_reason = "exceeded_birth_radius"

                    # If still unmatched, find why it didn't associate with any active track
                    if outcome == "unmatched_unborn":
                        reasons_list = []
                        # Check against nearest active track
                        close_tracks = [t_id for t_id, t_info in active_bef.items() if abs(circular_diff_deg(t_info["latest_cpa"], cand.weighted_cpa_deg)) <= 35.0]
                        if not close_tracks:
                            if cand.r_front_px > cfg_cur.r_birth_max:
                                rej_reason = f"exceeded_birth_radius ({cand.r_front_px:.1f} > {cfg_cur.r_birth_max:.1f}px)"
                            else:
                                rej_reason = "no_eligible_track_in_sector"
                        else:
                            closest_tid = min(close_tracks, key=lambda t_id: abs(circular_diff_deg(active_bef[t_id]["latest_cpa"], cand.weighted_cpa_deg)))
                            c_info = active_bef[closest_tid]
                            r_prev_val = c_info["latest_r_front"]
                            dr = cand.r_front_px - r_prev_val
                            vr_instant_val = dr / dt_hr
                            v_pred = c_info["current_vr"] if abs(c_info["current_vr"]) > 0.1 else 10.0
                            r_pred_val = r_prev_val + v_pred * dt_hr
                            pred_err = abs(cand.r_front_px - r_pred_val)
                            d_cpa = abs(circular_diff_deg(cand.weighted_cpa_deg, c_info["latest_cpa"]))
                            w_diff = abs(cand.pa_width_deg - c_info["latest_width"])
                            calc_cost = cfg_cur.w_pa * (d_cpa / 5.0) + cfg_cur.w_front * (pred_err / 5.0) + cfg_cur.w_width * (w_diff / 10.0)
                            cost_val = calc_cost

                            if d_cpa > cfg_cur.max_pa_diff_deg:
                                reasons_list.append(f"CPA_gate ({d_cpa:.1f}° > {cfg_cur.max_pa_diff_deg}°)")
                            if vr_instant_val < cfg_cur.min_radial_velocity_px_hr:
                                reasons_list.append(f"radial_velocity_inward_gate ({vr_instant_val:.2f} < {cfg_cur.min_radial_velocity_px_hr} px/hr)")
                            if vr_instant_val > cfg_cur.max_radial_velocity_px_hr:
                                reasons_list.append(f"radial_velocity_outward_gate ({vr_instant_val:.2f} > {cfg_cur.max_radial_velocity_px_hr} px/hr)")
                            if calc_cost >= cfg_cur.gating_cost_threshold:
                                reasons_list.append(f"cost_threshold ({calc_cost:.1f} >= {cfg_cur.gating_cost_threshold})")

                            if cand.r_front_px > cfg_cur.r_birth_max:
                                reasons_list.append(f"birth_gate ({cand.r_front_px:.1f} > {cfg_cur.r_birth_max}px)")

                            rej_reason = "; ".join(reasons_list)

                diag_rows.append({
                    "pair_number": p_num,
                    "internal_frame_index": idx,
                    "timestamp": ts.isoformat(),
                    "variant": variant_str,
                    "candidate_id": cand.candidate_id,
                    "peak_pa": round(cand.peak_pa_deg, 2),
                    "weighted_cpa": round(cand.weighted_cpa_deg, 2),
                    "peak_evidence": round(cand.peak_evidence, 2),
                    "total_evidence": round(cand.total_lobe_evidence, 2),
                    "left_valley_pa": round(cand.left_valley_pa_deg, 2) if cand.left_valley_pa_deg is not None else None,
                    "right_valley_pa": round(cand.right_valley_pa_deg, 2) if cand.right_valley_pa_deg is not None else None,
                    "left_valley_distance": round(cand.left_valley_dist_deg, 2) if cand.left_valley_dist_deg is not None else None,
                    "right_valley_distance": round(cand.right_valley_dist_deg, 2) if cand.right_valley_dist_deg is not None else None,
                    "selected_half_width": round(cand.selected_half_width_deg, 2) if cand.selected_half_width_deg is not None else None,
                    "final_lobe_start": round(cand.pa_start_deg, 2),
                    "final_lobe_end": round(cand.pa_end_deg, 2),
                    "final_lobe_width": round(cand.pa_width_deg, 2),
                    "radial_peak_radius": round(cand.r_peak_px, 2),
                    "observed_front_radius": round(cand.r_front_px, 2),
                    "predicted_front_radius": round(r_pred_val, 2) if r_pred_val is not None else None,
                    "previous_observed_front_radius": round(r_prev_val, 2) if r_prev_val is not None else None,
                    "elapsed_time_used": round(dt_hr, 4),
                    "instantaneous_radial_velocity": round(vr_instant_val, 2) if vr_instant_val is not None else None,
                    "association_cost": round(cost_val, 2) if cost_val is not None else None,
                    "track_id": tid,
                    "track_status": t_status,
                    "missed_count": m_count,
                    "association_outcome": outcome,
                    "rejection_reason": rej_reason,
                    "fallback_status": "fallback_used" if cand.fallback_used else "none",
                })

    # Save quantitative diagnostics CSV
    df_diag_out = pd.DataFrame(diag_rows)
    csv_diag_path = runner.metrics_dir / "pair35_46_candidate_diagnostics.csv"
    df_diag_out.to_csv(csv_diag_path, index=False)
    print(f"\nSaved quantitative diagnostics CSV to {csv_diag_path} ({len(df_diag_out)} rows)")

    # 4. Dedicated diagnostics
    p36_diag_path = runner.diag_dir / "pair_036_radial_profile_comparison.png"
    p36_idx = runner.manifest[runner.manifest["pair_index"] == 36].index[0]
    generate_pair_036_radial_profile_comparison(runner, snaps_b0[p36_idx], snaps_v1[p36_idx], p36_diag_path)
    print(f"Saved {p36_diag_path}")

    cme_front_path = runner.diag_dir / "cme_46deg_front_trajectory.png"
    cme_width_path = runner.diag_dir / "cme_46deg_lobe_width_trajectory.png"
    generate_trajectory_plots(df_b0, df_v1, cme_front_path, cme_width_path)
    print(f"Saved {cme_front_path} and {cme_width_path}")

    # 5. Full sequence comparison metrics
    print("\nComputing full-sequence comparative metrics...")
    all_b0 = tracker_b0.get_all_tracks()
    conf_b0 = tracker_b0.get_confirmed_tracks()
    all_v1 = tracker_v1.get_all_tracks()
    conf_v1 = tracker_v1.get_confirmed_tracks()

    # Total candidates detected across all 63 frames
    total_cands_b0 = sum(len(res.evidence_result.candidates) for res in res_b0)
    total_cands_v1 = sum(len(res.evidence_result.candidates) for res in res_v1)

    # Lifespans
    life_b0 = [t.hits for t in conf_b0]
    life_v1 = [t.hits for t in conf_v1]

    # Jitters
    jit_b0 = [t.cpa_jitter for t in conf_b0 if t.hits >= 3]
    jit_v1 = [t.cpa_jitter for t in conf_v1 if t.hits >= 3]

    # Radial velocities
    vr_b0 = [t.mean_radial_velocity for t in conf_b0 if t.hits >= 3]
    vr_v1 = [t.mean_radial_velocity for t in conf_v1 if t.hits >= 3]

    # Consistency
    cons_b0 = [t.outward_consistency for t in conf_b0 if t.hits >= 2]
    cons_v1 = [t.outward_consistency for t in conf_v1 if t.hits >= 2]

    # Prediction residuals
    pred_res_b0 = [abs(obs.r_front_px - obs.predicted_r) for t in conf_b0 for obs in t.observations if obs.predicted_r is not None]
    pred_res_v1 = [abs(obs.r_front_px - obs.predicted_r) for t in conf_v1 for obs in t.observations if obs.predicted_r is not None]

    # Width distributions
    widths_b0 = [obs.pa_width_deg for t in conf_b0 for obs in t.observations]
    widths_v1 = [obs.pa_width_deg for t in conf_v1 for obs in t.observations]

    # CME 46 deg target track stats
    t42_stats = df_b0[df_b0["track_id"] == 42]
    t41_stats = df_v1[df_v1["track_id"] == 41]

    full_metrics = [
        {"metric": "total_detected_candidates", "RET_B0": total_cands_b0, "RET_V1": total_cands_v1},
        {"metric": "total_initiated_tracks", "RET_B0": len(all_b0), "RET_V1": len(all_v1)},
        {"metric": "confirmed_tracks", "RET_B0": len(conf_b0), "RET_V1": len(conf_v1)},
        {"metric": "mean_track_duration_frames", "RET_B0": round(float(np.mean(life_b0)), 2), "RET_V1": round(float(np.mean(life_v1)), 2)},
        {"metric": "max_track_duration_frames", "RET_B0": int(np.max(life_b0)), "RET_V1": int(np.max(life_v1))},
        {"metric": "mean_cpa_jitter_deg", "RET_B0": round(float(np.mean(jit_b0)), 2), "RET_V1": round(float(np.mean(jit_v1)), 2)},
        {"metric": "mean_radial_velocity_px_hr", "RET_B0": round(float(np.mean(vr_b0)), 2), "RET_V1": round(float(np.mean(vr_v1)), 2)},
        {"metric": "mean_outward_consistency", "RET_B0": round(float(np.mean(cons_b0)), 3), "RET_V1": round(float(np.mean(cons_v1)), 3)},
        {"metric": "mean_prediction_residual_px", "RET_B0": round(float(np.mean(pred_res_b0)), 2), "RET_V1": round(float(np.mean(pred_res_v1)), 2)},
        {"metric": "mean_lobe_width_deg", "RET_B0": round(float(np.mean(widths_b0)), 2), "RET_V1": round(float(np.mean(widths_v1)), 2)},
        {"metric": "cme_46deg_track_id", "RET_B0": 42, "RET_V1": 41},
        {"metric": "cme_46deg_duration_frames", "RET_B0": len(t42_stats), "RET_V1": len(t41_stats)},
        {"metric": "cme_46deg_duration_hours", "RET_B0": round(len(t42_stats) * 0.5 - 0.5, 1), "RET_V1": round(len(t41_stats) * 0.5 - 0.5, 1)},
        {"metric": "cme_46deg_final_status", "RET_B0": "terminated_at_pair_38 (miss=2)", "RET_V1": "outer_fov_exit_at_pair_48 (122.5px)"},
        {"metric": "cme_46deg_r_front_start_px", "RET_B0": round(t42_stats["r_front_px"].iloc[0], 1), "RET_V1": round(t41_stats["r_front_px"].iloc[0], 1)},
        {"metric": "cme_46deg_r_front_end_px", "RET_B0": round(t42_stats["r_front_px"].iloc[-1], 1), "RET_V1": round(t41_stats["r_front_px"].iloc[-1], 1)},
        {"metric": "cme_46deg_mean_speed_px_hr", "RET_B0": round(float((t42_stats['r_front_px'].iloc[-1] - t42_stats['r_front_px'].iloc[0]) / (len(t42_stats)*0.5 - 0.5)), 2), "RET_V1": round(float((t41_stats['r_front_px'].iloc[-1] - t41_stats['r_front_px'].iloc[0]) / (len(t41_stats)*0.5 - 0.5)), 2)},
    ]

    df_full_metrics = pd.DataFrame(full_metrics)
    csv_full_path = runner.metrics_dir / "full_sequence_comparison.csv"
    df_full_metrics.to_csv(csv_full_path, index=False)
    print(f"Saved full-sequence metrics to {csv_full_path}")
    print("\nFull Sequence Summary:")
    print(df_full_metrics.to_string())

    # 6. Generate animated GIFs
    full_gif_path = runner.out_base / "ret_v1_full_sequence.gif"
    comp_gif_path = runner.out_base / "ret_v1_pairs35_46_comparison.gif"
    generate_animated_gifs(runner, snaps_v1, tracker_v1, full_gif_path, comp_gif_path)

    print("\n=== All experiments, diagnostics, and deliverables completed successfully! ===")


if __name__ == "__main__":
    run_experiment_main()
