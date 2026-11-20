r"""Comprehensive Visualization & Animation Suite for Radial Evidence Tracking (RET).

Generates:
1. CPA vs Time (Angular stability across variants and baselines)
2. Radius vs Time (Height-time kinematic ridges with physical velocity fits)
3. Quantitative Architecture & Sensitivity Summary (Track count, Jitter, Consistency, Runtime)
4. Gaussian Scale Sensitivity Analysis
5. Unwrapped Polar Evidence Map & Angular Profile for CME peak frame
6. Frame-by-frame 4-panel diagnostics for key frames (Pairs 35, 38, 41, 46, 52, 61)
7. Full 63-frame tracking evolution animation (ret_tracking_evolution.gif)
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from PIL import Image

from cme_pipeline.cadence_selection import parse_timestamp
from cme_pipeline.utils.geometry import circular_diff_deg
from experiments.ret_tracking.ret_evidence_engine import (
    PolarGridGeometry,
    compute_pixel_evidence,
    apply_spatial_gaussian_spreading,
    sample_polar_evidence,
    accumulate_radial_evidence,
    detect_angular_candidates,
)


def plot_kinematic_trajectories(results_dir: Path, out_dir: Path):
    """Plots CPA vs Time and Radius vs Time across RET variants."""
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["RET1_OutwardConfirm", "RET2_MotionPred", "RET3_MultiLobe"]
    display_names = {
        "RET1_OutwardConfirm": "RET-1: Outward Confirmation",
        "RET2_MotionPred": "RET-2: Motion Prediction",
        "RET3_MultiLobe": "RET-3: Multi-Lobe & Overlap",
    }

    # 1. CPA vs Time
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for idx, var in enumerate(variants):
        ax = axes[idx]
        csv_file = results_dir / f"tracks_{var}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        df["dt"] = pd.to_datetime(df["timestamp"])
        confirmed_ids = df[df["status"] == "confirmed"]["track_id"].unique()

        # Plot long confirmed tracks
        for tid in confirmed_ids:
            tdf = df[df["track_id"] == tid].sort_values("dt")
            if len(tdf) < 3:
                continue
            ax.plot(tdf["dt"], tdf["cpa_deg"], marker="o", markersize=4, linewidth=1.5, alpha=0.85, label=f"Trk {tid} (len={len(tdf)})")

        ax.set_title(display_names[var], fontsize=12, fontweight="bold")
        ax.set_ylabel("CPA (deg)" if idx == 0 else "", fontsize=10)
        ax.set_xlabel("Time (2010-12-12 UTC)", fontsize=10)
        ax.set_ylim(0, 360)
        ax.grid(True, linestyle="--", alpha=0.4)
        if len(confirmed_ids) <= 8 and len(confirmed_ids) > 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.suptitle("CPA Stability vs Time across RET Architectures (Mean Jitter ~5.2 deg)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    cpa_path = out_dir / "cpa_trajectory_comparison.png"
    plt.savefig(cpa_path, dpi=200)
    plt.close()
    print(f"Saved {cpa_path}")

    # 2. Radius vs Time (Height-Time Kinematics)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for idx, var in enumerate(variants):
        ax = axes[idx]
        csv_file = results_dir / f"tracks_{var}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        df["dt"] = pd.to_datetime(df["timestamp"])
        confirmed_ids = df[df["status"] == "confirmed"]["track_id"].unique()

        for tid in confirmed_ids:
            tdf = df[df["track_id"] == tid].sort_values("dt")
            if len(tdf) < 3:
                continue
            ax.plot(tdf["dt"], tdf["r_front_px"], marker="^", markersize=4, linestyle="-", linewidth=1.5, alpha=0.85, label=f"Trk {tid} Front")

        ax.set_title(display_names[var], fontsize=12, fontweight="bold")
        ax.set_ylabel("Front Radius (pixels)" if idx == 0 else "", fontsize=10)
        ax.set_xlabel("Time (2010-12-12 UTC)", fontsize=10)
        ax.set_ylim(20, 125)
        ax.axhline(22.0, color="gray", linestyle=":", label="Occulter Boundary (r=22px)")
        ax.axhline(122.5, color="red", linestyle=":", label="FOV Boundary (r=122.5px)")
        ax.grid(True, linestyle="--", alpha=0.4)
        if len(confirmed_ids) <= 8 and len(confirmed_ids) > 0:
            ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Height–Time (Radius vs Time) Kinematic Ridges for RET Architectures", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    rad_path = out_dir / "radius_trajectory_comparison.png"
    plt.savefig(rad_path, dpi=200)
    plt.close()
    print(f"Saved {rad_path}")


def plot_metrics_summary(results_dir: Path, out_dir: Path):
    """Plots architecture benchmark and comparison against A5b and Gaussian tracking."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "track_summary_table.csv"
    if not summary_csv.exists():
        return

    df = pd.read_csv(summary_csv)

    # Comparison data: A5b baseline, G0, G1, G3, RET-1, RET-2, RET-3
    comparison_names = ["A5b\nBaseline", "G0\nRaw", "G1\nSpatial", "G3\nMotion", "RET-1\nConfirm", "RET-2\nMotion", "RET-3\nMultiLobe"]
    confirmed_counts = [182, 358, 170, 149, 41, 41, 40]
    cpa_jitters = [54.7, 62.4, 40.2, 41.5, 5.49, 5.26, 5.24]
    consistencies = [92.0, 96.6, 95.9, 90.5, 94.8, 95.1, 98.9]
    runtimes = [5.5, 8.07, 2.21, 2.21, 0.34, 0.34, 0.34]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    x = np.arange(len(comparison_names))
    colors = ["#7f7f7f", "#d9534f", "#e67e22", "#f1c40f", "#3498db", "#2980b9", "#2ecc71"]

    # Panel 1: Confirmed Track Count (Fragmentation)
    ax1.bar(x, confirmed_counts, 0.55, color=colors, edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_names, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Confirmed Track Count", fontsize=11)
    ax1.set_title("Track Fragmentation (Lower = Better Coherence)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(confirmed_counts):
        ax1.text(i, v + 8, str(v), ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Panel 2: CPA Jitter (Angular Stability)
    ax2.bar(x, cpa_jitters, 0.55, color=colors, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_names, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean CPA Jitter (deg)", fontsize=11)
    ax2.set_title("Directional Stability (Lower = Less Angular Hopping)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(cpa_jitters):
        ax2.text(i, v + 1.5, f"{v:.1f}°", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Panel 3: Physical Outward Consistency
    ax3.bar(x, consistencies, 0.55, color=colors, edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(comparison_names, fontsize=10, fontweight="bold")
    ax3.set_ylabel("Outward Displacement Steps (%)", fontsize=11)
    ax3.set_ylim(85, 102)
    ax3.set_title("Physical Outward Motion Consistency", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(consistencies):
        ax3.text(i, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Panel 4: Runtime Performance
    ax4.bar(x, runtimes, 0.55, color=colors, edgecolor="black")
    ax4.set_xticks(x)
    ax4.set_xticklabels(comparison_names, fontsize=10, fontweight="bold")
    ax4.set_ylabel("Total Runtime for 63 Frames (sec)", fontsize=11)
    ax4.set_title("Execution Runtime (63 Observation Frames)", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(runtimes):
        ax4.text(i, v + 0.2, f"{v:.2f}s", ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.suptitle("Architecture Benchmark Comparison: Baseline A5b vs Gaussian Tracking vs RET", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    summary_path = out_dir / "tracking_metrics_summary.png"
    plt.savefig(summary_path, dpi=200)
    plt.close()
    print(f"Saved {summary_path}")

    # Gaussian Scale Sensitivity Plot for RET-2
    sigma_runs = ["RET2_sigma_0px", "RET2_sigma_1px", "RET2_sigma_2px", "RET2_sigma_4px", "RET2_sigma_6px"]
    df_sig = df[df["run_name"].isin(sigma_runs)].sort_values("sigma_s")
    if len(df_sig) >= 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(df_sig["sigma_s"], df_sig["confirmed_tracks"], marker="o", color="#2b5c8f", linewidth=2, label="Confirmed Tracks")
        ax1.plot(df_sig["sigma_s"], df_sig["total_tracks_born"], marker="s", color="#d9534f", linewidth=2, linestyle="--", label="Total Tracks Born")
        ax1.set_xlabel(r"Gaussian Spreading Scale $\sigma_s$ (pixels)", fontsize=11)
        ax1.set_ylabel("Track Count", fontsize=11)
        ax1.set_title(r"RET-2 Sensitivity: Track Count vs $\sigma_s$", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.plot(df_sig["sigma_s"], df_sig["mean_cpa_jitter_deg"], marker="^", color="#3b8b54", linewidth=2, label="Mean CPA Jitter (deg)")
        ax2.plot(df_sig["sigma_s"], df_sig["mean_radial_velocity_px_hr"], marker="d", color="#e67e22", linewidth=2, linestyle="--", label="Mean Velocity (px/hr)")
        ax2.set_xlabel(r"Gaussian Spreading Scale $\sigma_s$ (pixels)", fontsize=11)
        ax2.set_ylabel("Value", fontsize=11)
        ax2.set_title(r"RET-2 Sensitivity: Stability & Speed vs $\sigma_s$", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        sig_path = out_dir / "gaussian_scale_sensitivity.png"
        plt.savefig(sig_path, dpi=200)
        plt.close()
        print(f"Saved {sig_path}")


def plot_unwrapped_polar_diagnostic(
    frame_idx: int,
    timestamp: datetime,
    polar_ev: np.ndarray,
    a_smooth: np.ndarray,
    candidates: List[Any],
    grid: PolarGridGeometry,
    output_path: Path,
):
    """Plots unwrapped polar evidence map E(r, theta) and angular profile A(theta)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1.5, 1.0]})

    # 1. Unwrapped Polar Map
    im = ax1.imshow(
        polar_ev,
        cmap="inferno",
        aspect="auto",
        origin="lower",
        extent=[0, 360, grid.r_min_px, grid.r_max_px],
    )
    ax1.set_ylabel("Radius (pixels)", fontsize=11)
    ax1.set_title(f"Unwrapped Polar Evidence Map $\\widetilde{{E}}_t(r, \\theta)$ — {timestamp.strftime('%Y-%m-%d %H:%M UTC')}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.02, label="Spread Evidence")

    # Overlay detected candidate fronts
    for cand in candidates:
        ax1.axvline(cand.weighted_cpa_deg, color="cyan", linestyle="--", alpha=0.8)
        ax1.plot(cand.weighted_cpa_deg, cand.r_front_px, "^", color="lime", markersize=9, markeredgecolor="black")
        ax1.plot(cand.weighted_cpa_deg, cand.r_peak_px, "o", color="yellow", markersize=6, markeredgecolor="black")
        ax1.text(cand.weighted_cpa_deg + 2, cand.r_front_px + 2, f"CPA {cand.weighted_cpa_deg:.0f}°\nFront {cand.r_front_px:.0f}px", color="white", fontsize=8, fontweight="bold")

    # 2. Angular Accumulation Profile A(theta)
    ax2.plot(grid.pa_bins, a_smooth, color="#2b5c8f", linewidth=2, label="Radial Accumulation $A_t(\\theta)$")
    ax2.set_xlabel("Position Angle (deg) [0=N, 90=E, 180=S, 270=W]", fontsize=11)
    ax2.set_ylabel("Accumulated Evidence", fontsize=11)
    ax2.set_xlim(0, 360)
    ax2.grid(True, linestyle="--", alpha=0.5)

    for cand in candidates:
        ax2.axvline(cand.weighted_cpa_deg, color="cyan", linestyle="--", alpha=0.8)
        ax2.plot(cand.weighted_cpa_deg, cand.peak_evidence, "o", color="red", markersize=7)
        ax2.text(cand.weighted_cpa_deg, cand.peak_evidence * 1.05, f"CPA {cand.weighted_cpa_deg:.0f}°", ha="center", fontsize=9, fontweight="bold", color="darkred")

    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {output_path}")


def plot_frame_diagnostic_panel(
    frame_idx: int,
    pair_idx: int,
    timestamp: datetime,
    rd_image: np.ndarray,
    spread_ev: np.ndarray,
    a_smooth: np.ndarray,
    candidates: List[Any],
    active_tracks: List[Any],
    grid: PolarGridGeometry,
    valid_mask: np.ndarray,
    output_path: Path,
):
    """Produces 4-panel diagnostic for an observation frame: RD, Spread Evidence, A(theta), Tracking Overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    sx, sy = grid.sx, grid.sy

    def add_rings(ax):
        c_in = plt.Circle((sx, sy), grid.r_min_px, color="cyan", fill=False, linestyle="--", linewidth=1.2)
        c_out = plt.Circle((sx, sy), grid.r_max_px, color="cyan", fill=False, linestyle=":", linewidth=1.2)
        ax.add_patch(c_in)
        ax.add_patch(c_out)
        ax.plot(sx, sy, "+", color="yellow", markersize=6)
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_aspect("equal")

    # 1. Raw North-up Running Difference
    vlim = float(np.nanpercentile(np.abs(rd_image[valid_mask]), 98)) if np.any(valid_mask) else 1.0
    vlim = max(0.1, vlim)
    im0 = axes[0].imshow(rd_image, cmap="gray", vmin=-vlim, vmax=vlim)
    add_rings(axes[0])
    axes[0].set_title(f"1. Raw Running Difference\nPair {pair_idx} ({timestamp.strftime('%H:%M UTC')})", fontsize=10, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Gaussian Spread Evidence
    im1 = axes[1].imshow(spread_ev, cmap="magma", vmin=0, vmax=4)
    add_rings(axes[1])
    axes[1].set_title(f"2. Spatial Gaussian Evidence $\\widetilde{{E}}_t$\n($\\sigma_s=2.0$ px, Mask-Normalized)", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Angular Evidence Profile A(theta)
    axes[2].plot(grid.pa_bins, a_smooth, color="#2b5c8f", linewidth=2)
    axes[2].set_xlim(0, 360)
    axes[2].set_xlabel("Position Angle (deg)", fontsize=10)
    axes[2].set_ylabel("Radial Accumulation $A_t(\\theta)$", fontsize=10)
    axes[2].set_title(f"3. Angular Evidence Profile $A_t(\\theta)$\n{len(candidates)} Candidate Lobe(s) Detected", fontsize=10, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    for cand in candidates:
        axes[2].axvline(cand.weighted_cpa_deg, color="crimson", linestyle="--", alpha=0.8)
        axes[2].plot(cand.weighted_cpa_deg, cand.peak_evidence, "o", color="crimson", markersize=6)
        axes[2].text(cand.weighted_cpa_deg, cand.peak_evidence + 2, f"{cand.weighted_cpa_deg:.0f}°", ha="center", fontsize=8, fontweight="bold", color="crimson")

    # 4. RET Tracking Overlay on RD
    axes[3].imshow(rd_image, cmap="gray", vmin=-vlim, vmax=vlim)
    add_rings(axes[3])
    axes[3].set_title(f"4. RET Tracking Overlay\nConfirmed Tracks & Front Arcs", fontsize=10, fontweight="bold")

    for trk in active_tracks:
        obs = trk.latest
        cpa_rad = np.radians(obs.cpa_deg)
        # CPA ray end at outer FOV
        x_ray = sx - grid.r_max_px * np.sin(cpa_rad)
        y_ray = sy + grid.r_max_px * np.cos(cpa_rad)
        axes[3].plot([sx, x_ray], [sy, y_ray], color="cyan", linestyle="--", linewidth=1.2, alpha=0.7)

        # Observed Front position
        x_front = sx - obs.r_front_px * np.sin(cpa_rad)
        y_front = sy + obs.r_front_px * np.cos(cpa_rad)
        axes[3].plot(x_front, y_front, "^", color="lime", markersize=9, markeredgecolor="black")

        # Peak evidence position
        x_pk = sx - obs.r_peak_px * np.sin(cpa_rad)
        y_pk = sy + obs.r_peak_px * np.cos(cpa_rad)
        axes[3].plot(x_pk, y_pk, "o", color="yellow", markersize=5, markeredgecolor="black")

        # Label
        axes[3].annotate(
            f"Trk {trk.track_id}\nCPA {obs.cpa_deg:.0f}°\nr={obs.r_front_px:.0f}px",
            (x_front + 4, y_front - 4),
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor="lime"),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved diagnostic panel {output_path}")


def generate_full_animation(
    runner: Any,
    tracker_config: Any,
    out_gif_path: Path,
):
    """Generates full-sequence 63-frame GIF animation of RET tracking."""
    frames_dir = out_gif_path.parent / "anim_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    from experiments.ret_tracking.ret_tracker import RadialEvidenceTracker
    tracker = RadialEvidenceTracker(tracker_config, runner.grid)
    frame_img_paths = []

    print("Generating frames for tracking animation...")
    for idx in range(runner.n_frames):
        t_stamp = runner.timestamps[idx]
        res = tracker.step(idx, t_stamp, runner.z_maps[idx], runner.valid_masks[idx])
        conf_tracks = res.active_confirmed_tracks

        fig, ax = plt.subplots(figsize=(7, 7))
        rd_img = runner.rd_images[idx]
        val_m = runner.valid_masks[idx]
        vlim = float(np.nanpercentile(np.abs(rd_img[val_m]), 98)) if np.any(val_m) else 1.0
        vlim = max(0.1, vlim)
        ax.imshow(rd_img, cmap="gray", vmin=-vlim, vmax=vlim)

        # Occulter & FOV rings
        c_in = plt.Circle((runner.sx, runner.sy), runner.grid.r_min_px, color="cyan", fill=False, linestyle="--", linewidth=1.2)
        c_out = plt.Circle((runner.sx, runner.sy), runner.grid.r_max_px, color="cyan", fill=False, linestyle=":", linewidth=1.2)
        ax.add_patch(c_in)
        ax.add_patch(c_out)
        ax.plot(runner.sx, runner.sy, "+", color="yellow", markersize=6)
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        # Overlay active confirmed tracks
        for trk in conf_tracks:
            obs = trk.latest
            cpa_rad = np.radians(obs.cpa_deg)
            x_ray = runner.sx - runner.grid.r_max_px * np.sin(cpa_rad)
            y_ray = runner.sy + runner.grid.r_max_px * np.cos(cpa_rad)
            ax.plot([runner.sx, x_ray], [runner.sy, y_ray], color="cyan", linestyle="--", linewidth=1.2, alpha=0.7)

            x_front = runner.sx - obs.r_front_px * np.sin(cpa_rad)
            y_front = runner.sy + obs.r_front_px * np.cos(cpa_rad)
            ax.plot(x_front, y_front, "^", color="lime", markersize=8, markeredgecolor="black")

            ax.annotate(
                f"Trk {trk.track_id} ({obs.cpa_deg:.0f}°)",
                (x_front + 4, y_front - 4),
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor="lime")
            )

        pair_num = runner.manifest.iloc[idx]["pair_index"]
        ax.text(8, 20, f"STEREO-A COR2 RD | Pair {pair_num:02d}\n{t_stamp.strftime('%Y-%m-%d %H:%M UTC')}\nActive Tracks: {len(conf_tracks)}", color="white", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="navy", alpha=0.8))

        f_path = frames_dir / f"frame_{idx:03d}.png"
        plt.tight_layout()
        plt.savefig(f_path, dpi=120)
        plt.close()
        frame_img_paths.append(f_path)

    # Stitch into animated GIF
    print(f"Stitching {len(frame_img_paths)} frames into animated GIF...")
    pil_images = [Image.open(p) for p in frame_img_paths]
    pil_images[0].save(
        out_gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=250,  # 250ms per frame = 4 fps
        loop=0,
    )
    print(f"Saved tracking animation to {out_gif_path}")


def main():
    from experiments.ret_tracking.run_experiments import RETPipelineRunner
    from experiments.ret_tracking.ret_tracker import RETConfig, RadialEvidenceTracker

    root = Path(__file__).resolve().parent.parent.parent
    results_dir = root / "experiments/ret_tracking/results"
    vis_dir = root / "experiments/ret_tracking/visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary and trajectory plots
    plot_kinematic_trajectories(results_dir, vis_dir)
    plot_metrics_summary(results_dir, vis_dir)

    # 2. Key Frame Diagnostics
    runner = RETPipelineRunner(data_dir=root / "runs/dual_archive_20101212/preprocessed")
    cfg = RETConfig(variant="RET-2", sigma_s=2.0)
    tracker = RadialEvidenceTracker(cfg, runner.grid)

    target_pairs = [35, 38, 41, 46, 52, 61]
    df_manifest = runner.manifest

    print("\nGenerating key frame diagnostics...")
    for idx in range(runner.n_frames):
        p_idx = int(df_manifest.iloc[idx]["pair_index"])
        t_stamp = runner.timestamps[idx]
        res = tracker.step(idx, t_stamp, runner.z_maps[idx], runner.valid_masks[idx])

        if p_idx in target_pairs:
            diag_file = vis_dir / f"diag_tracking_pair_{p_idx}.png"
            plot_frame_diagnostic_panel(
                frame_idx=idx,
                pair_idx=p_idx,
                timestamp=t_stamp,
                rd_image=runner.rd_images[idx],
                spread_ev=res.evidence_result.spread_evidence,
                a_smooth=res.evidence_result.angular_evidence_smoothed,
                candidates=res.evidence_result.candidates,
                active_tracks=res.active_confirmed_tracks,
                grid=runner.grid,
                valid_mask=runner.valid_masks[idx],
                output_path=diag_file,
            )

            # Unwrapped polar map for CME peak frame (Pair 41)
            if p_idx == 41:
                polar_file = vis_dir / "unwrapped_polar_evidence_map.png"
                plot_unwrapped_polar_diagnostic(
                    frame_idx=idx,
                    timestamp=t_stamp,
                    polar_ev=res.evidence_result.polar_evidence,
                    a_smooth=res.evidence_result.angular_evidence_smoothed,
                    candidates=res.evidence_result.candidates,
                    grid=runner.grid,
                    output_path=polar_file,
                )

    # 3. Full Sequence Tracking Animation
    anim_gif_path = vis_dir / "ret_tracking_evolution.gif"
    generate_full_animation(runner, cfg, anim_gif_path)


if __name__ == "__main__":
    main()
