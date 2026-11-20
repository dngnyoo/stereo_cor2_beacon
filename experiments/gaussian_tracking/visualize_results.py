"""Visualization and Diagnostic Suite for Gaussian Evidence Tracking.

Generates:
1. CPA vs Time (Angular evolution and stability)
2. Radius vs Time (Height-time kinematic ridges showing CME outward expansion)
3. Quantitative Architecture & Sensitivity Comparison (Track count, Lifespan, Jitter, Consistency)
4. Key Frame Diagnostic Panels (Raw RD, Z-score, Gaussian Spread Evidence, Extracted Regions, Active Track Overlays)
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

from cme_pipeline.cadence_selection import parse_timestamp
from cme_pipeline.utils.geometry import compute_wcs_rasters, circular_diff_deg


def plot_kinematic_trajectories(
    results_dir: Path,
    out_dir: Path,
    a5b_csv_path: Optional[Path] = None,
):
    """Plots CPA vs Time and Radius vs Time across G0, G1, G2, G3 and baseline A5b."""
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["G0_Raw", "G1_Spatial_sigma2", "G2_Temporal_sigma2", "G3_MotionAware_sigma2"]
    display_names = {
        "G0_Raw": "G0: Raw Pixel Evidence",
        "G1_Spatial_sigma2": "G1: Spatial Gaussian (sigma=2px)",
        "G2_Temporal_sigma2": "G2: Causal Temporal Memory",
        "G3_MotionAware_sigma2": "G3: Motion-Aware Association",
    }

    # 1. CPA vs Time
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

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
            if len(tdf) < 2:
                continue
            ax.plot(tdf["dt"], tdf["centroid_pa_deg"], marker="o", markersize=4, linewidth=1.5, alpha=0.8, label=f"Trk {tid} (len={len(tdf)})")

        ax.set_title(display_names[var], fontsize=12, fontweight="bold")
        ax.set_ylabel("Central Position Angle (deg)", fontsize=10)
        ax.set_ylim(0, 360)
        ax.grid(True, linestyle="--", alpha=0.4)
        if len(confirmed_ids) > 0 and len(confirmed_ids) <= 8:
            ax.legend(fontsize=8, loc="upper right")

    for ax in axes[2:]:
        ax.set_xlabel("Observation Time (2010-12-12 UTC)", fontsize=10)

    plt.suptitle("CPA vs Time Trajectory Comparison across Architectures", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    cpa_plot = out_dir / "cpa_trajectory_comparison.png"
    plt.savefig(cpa_plot, dpi=200)
    plt.close()
    print(f"Saved {cpa_plot}")

    # 2. Radius vs Time (Height-Time Kinematics)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

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
            if len(tdf) < 2:
                continue
            # Plot leading front and centroid
            ax.plot(tdf["dt"], tdf["r_front_px"], marker="^", markersize=4, linestyle="-", linewidth=1.5, alpha=0.8, label=f"Trk {tid} Front")
            ax.plot(tdf["dt"], tdf["centroid_r_px"], marker="o", markersize=3, linestyle="--", linewidth=1.0, alpha=0.6)

        ax.set_title(display_names[var], fontsize=12, fontweight="bold")
        ax.set_ylabel("Radial Distance (pixels)", fontsize=10)
        ax.set_ylim(15, 125)
        ax.axhline(22.0, color="gray", linestyle=":", label="Occulter Boundary (r=22px)")
        ax.axhline(122.5, color="red", linestyle=":", label="FOV Boundary (r=122.5px)")
        ax.grid(True, linestyle="--", alpha=0.4)
        if len(confirmed_ids) > 0 and len(confirmed_ids) <= 6:
            ax.legend(fontsize=8, loc="upper left")

    for ax in axes[2:]:
        ax.set_xlabel("Observation Time (2010-12-12 UTC)", fontsize=10)

    plt.suptitle("Height–Time (Radius vs Time) Kinematic Ridges across Architectures", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    rad_plot = out_dir / "radius_trajectory_comparison.png"
    plt.savefig(rad_plot, dpi=200)
    plt.close()
    print(f"Saved {rad_plot}")


def plot_metrics_summary(results_dir: Path, out_dir: Path):
    """Plots architecture benchmark and sensitivity analysis summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "track_summary_table.csv"
    if not summary_csv.exists():
        return

    df = pd.read_csv(summary_csv)

    # 1. Architecture comparison: G0, G1, G2, G3
    arch_runs = ["G0_Raw", "G1_Spatial_sigma2", "G2_Temporal_sigma2", "G3_MotionAware_sigma2"]
    df_arch = df[df["run_name"].isin(arch_runs)].copy()
    df_arch["display"] = ["G0\nRaw", "G1\nSpatial", "G2\nTemporal", "G3\nMotion-Aware"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Track counts
    x = np.arange(len(df_arch))
    w = 0.35
    ax1.bar(x - w/2, df_arch["confirmed_tracks"], w, label="Confirmed Tracks", color="#2b5c8f", edgecolor="black")
    ax1.bar(x + w/2, df_arch["total_tracks_born"], w, label="Total Tracks Born", color="#a0c4df", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_arch["display"], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Track Count", fontsize=10)
    ax1.set_title("Track Formation and Confirmation", fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Lifespan
    ax2.bar(x - w/2, df_arch["mean_lifespan_frames"], w, label="Mean Lifespan", color="#3b8b54", edgecolor="black")
    ax2.bar(x + w/2, df_arch["max_lifespan_frames"], w, label="Max Lifespan", color="#8ed3a2", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_arch["display"], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Observation Frames", fontsize=10)
    ax2.set_title("Track Persistence / Lifespan", fontsize=11, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: CPA Jitter
    ax3.bar(x, df_arch["mean_cpa_jitter_deg"], 0.5, color="#d9534f", edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_arch["display"], fontsize=10, fontweight="bold")
    ax3.set_ylabel("CPA Jitter (deg)", fontsize=10)
    ax3.set_title("Directional Stability (Mean CPA Jitter)", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 4: Outward Consistency
    ax4.bar(x, df_arch["mean_outward_consistency"] * 100.0, 0.5, color="#f0ad4e", edgecolor="black")
    ax4.set_xticks(x)
    ax4.set_xticklabels(df_arch["display"], fontsize=10, fontweight="bold")
    ax4.set_ylabel("Outward Steps (%)", fontsize=10)
    ax4.set_ylim(0, 105)
    ax4.set_title("Physical Outward Motion Consistency", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Architecture Benchmark Summary (2010-12-12 STEREO-A COR2)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    metrics_plot = out_dir / "tracking_metrics_summary.png"
    plt.savefig(metrics_plot, dpi=200)
    plt.close()
    print(f"Saved {metrics_plot}")

    # 2. Sigma Sensitivity Plot
    sigma_runs = ["G1_sigma_1px", "G1_Spatial_sigma2", "G1_sigma_4px", "G1_sigma_6px"]
    df_sig = df[df["run_name"].isin(sigma_runs)].sort_values("sigma_s")
    if len(df_sig) >= 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(df_sig["sigma_s"], df_sig["confirmed_tracks"], marker="o", color="#2b5c8f", linewidth=2, label="Confirmed Tracks")
        ax1.plot(df_sig["sigma_s"], df_sig["total_tracks_born"], marker="s", color="#d9534f", linewidth=2, linestyle="--", label="Total Tracks Born")
        ax1.set_xlabel("Gaussian Spreading Scale $\\sigma_s$ (pixels)", fontsize=11)
        ax1.set_ylabel("Track Count", fontsize=11)
        ax1.set_title("Sensitivity: Track Count vs $\\sigma_s$", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.plot(df_sig["sigma_s"], df_sig["mean_cpa_jitter_deg"], marker="^", color="#3b8b54", linewidth=2, label="Mean CPA Jitter (deg)")
        ax2.plot(df_sig["sigma_s"], df_sig["mean_lifespan_frames"], marker="d", color="#f0ad4e", linewidth=2, linestyle="--", label="Mean Lifespan (frames)")
        ax2.set_xlabel("Gaussian Spreading Scale $\\sigma_s$ (pixels)", fontsize=11)
        ax2.set_ylabel("Value", fontsize=11)
        ax2.set_title("Sensitivity: Stability & Persistence vs $\\sigma_s$", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        sig_plot = out_dir / "gaussian_scale_sensitivity.png"
        plt.savefig(sig_plot, dpi=200)
        plt.close()
        print(f"Saved {sig_plot}")


def plot_frame_diagnostic_panel(
    frame_idx: int,
    timestamp: datetime,
    rd_image: np.ndarray,
    z_map: np.ndarray,
    spread_ev: np.ndarray,
    valid_mask: np.ndarray,
    sx: float,
    sy: float,
    r_min_px: float,
    r_max_px: float,
    df_tracks_g1: pd.DataFrame,
    df_tracks_g3: pd.DataFrame,
    output_path: Path,
):
    """Produces multi-panel diagnostic figure for an individual observation frame."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Helper for coronagraph rings
    def add_rings(ax):
        c_in = plt.Circle((sx, sy), r_min_px, color="cyan", fill=False, linestyle="--", linewidth=1.2)
        c_out = plt.Circle((sx, sy), r_max_px, color="cyan", fill=False, linestyle=":", linewidth=1.2)
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
    axes[0].set_title(f"Raw RD (DN/s)\n{timestamp.strftime('%Y-%m-%d %H:%M')}", fontsize=10, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Frame-level smoothed Z-score
    im1 = axes[1].imshow(z_map, cmap="inferno", vmin=0, vmax=6)
    add_rings(axes[1])
    axes[1].set_title(f"Standardized Z-Score\nFrame {frame_idx}", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Spatial Gaussian Spread Evidence E_tilde
    im2 = axes[2].imshow(spread_ev, cmap="magma", vmin=0, vmax=4)
    add_rings(axes[2])
    axes[2].set_title(f"Spread Evidence $\\widetilde{{E}}$ ($\\sigma_s$=2px)\nMask-Normalized Convolution", fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Active Tracks Overlay on RD
    axes[3].imshow(rd_image, cmap="gray", vmin=-vlim, vmax=vlim)
    add_rings(axes[3])
    axes[3].set_title(f"Discovered CME Tracks Overlay\nG1 (Green) vs G3 (Magenta)", fontsize=10, fontweight="bold")

    # Plot G1 tracks at this frame
    g1_obs = df_tracks_g1[df_tracks_g1["frame_idx"] == frame_idx]
    for _, obs in g1_obs.iterrows():
        cx, cy = obs["cx"], obs["cy"]
        tid = int(obs["track_id"])
        axes[3].plot(cx, cy, "o", color="#00ff66", markersize=8, markeredgecolor="black")
        axes[3].annotate(f"G1-T{tid}\n{obs['centroid_pa_deg']:.0f}°", (cx + 3, cy - 3), color="#00ff66", fontsize=8, fontweight="bold")

    # Plot G3 tracks at this frame
    g3_obs = df_tracks_g3[df_tracks_g3["frame_idx"] == frame_idx]
    for _, obs in g3_obs.iterrows():
        cx, cy = obs["cx"], obs["cy"]
        tid = int(obs["track_id"])
        axes[3].plot(cx, cy, "s", color="#ff00ff", markersize=6, fillstyle="none", markeredgewidth=1.5)
        axes[3].annotate(f"G3-T{tid}", (cx - 15, cy + 12), color="#ff00ff", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved diagnostic panel {output_path}")


if __name__ == "__main__":
    results_dir = Path("experiments/gaussian_tracking/results")
    vis_dir = Path("experiments/gaussian_tracking/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    plot_kinematic_trajectories(results_dir, vis_dir)
    plot_metrics_summary(results_dir, vis_dir)
