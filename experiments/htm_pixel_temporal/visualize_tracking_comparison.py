"""Diagnostic Visualizations for Stereo_COR2 Phase 3: A5b vs H3 vs H4 Tracking Comparison.

Generates:
1. `cpa_trajectory_comparison.png`: Frame-by-frame CPA trajectory of major CME tracks (A5b vs H3 vs H4).
2. `tracking_intervention_analysis.png`: Temporal intervention distribution (outlier rejections, contested tie-breaks).
3. `track_continuity_and_duration.png`: Lifespan and active observations comparison across variants.
4. `frame_diagnostic_interventions.png`: Visual overlays showing frames where temporal evidence intervened on RD images.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.utils.geometry import circular_diff_deg


def plot_cpa_trajectories(results_dir: Path, vis_dir: Path) -> Path:
    """Plots CPA trajectories over time for A5b Baseline, H3, and H4."""
    variants = [
        ("a5b_baseline", "A5b Baseline", "#1f77b4", "-"),
        ("h3_h1", "H3 (H1 Static)", "#2ca02c", "--"),
        ("h3_h2", "H3 (H2 Motion)", "#17becf", "-"),
        ("h4_h1_direct", "H4 (H1 Direct)", "#ff7f0e", "--"),
        ("h4_h2_direct", "H4 (H2 Direct)", "#d62728", "-"),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for var_name, label, color, ls in variants:
        csv_file = results_dir / f"track_states_{var_name}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        obs = df[df["state"] == "OBSERVED"].copy()
        if obs.empty:
            continue
        obs["pair_index"] = obs["pair_index"].astype(int)

        # Track 1 (SE limb, ~110°-130°)
        t1 = obs[obs["track_id"] == 1].sort_values("pair_index")
        if not t1.empty:
            cpa_col = "tracked_cpa" if "tracked_cpa" in t1.columns else "observed_cpa"
            ax1.plot(t1["pair_index"], t1[cpa_col], label=label, color=color, linestyle=ls, marker="o", markersize=4, alpha=0.85)

        # Track 2 (NE limb, ~30°-50°)
        t2 = obs[obs["track_id"] == 2].sort_values("pair_index")
        if not t2.empty:
            cpa_col = "tracked_cpa" if "tracked_cpa" in t2.columns else "observed_cpa"
            ax2.plot(t2["pair_index"], t2[cpa_col], label=label, color=color, linestyle=ls, marker="s", markersize=4, alpha=0.85)

    # Reference lines from catalogs
    # SEEDS 04:54 UT Event: CEN_PA 91° (overall CME centroid)
    # MVCC Event: Track 1 at ~117°, Track 2 at ~40°
    ax1.axhline(117.0, color="gray", linestyle=":", label="SEEDS/MVCC SE Reference (117°)")
    ax2.axhline(40.0, color="gray", linestyle=":", label="SEEDS/MVCC NE Reference (40°)")

    ax1.set_ylabel("Track 1 CPA [deg] (Southeast Lobe)", fontsize=11)
    ax1.set_title("Stereo_COR2 (2010-12-12) — CME Track CPA Continuity: A5b vs H3 vs H4", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylim(80, 160)

    ax2.set_xlabel("COR2 Observation Pair Index (2010-12-12)", fontsize=11)
    ax2.set_ylabel("Track 2 CPA [deg] (Northeast Lobe)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_ylim(10, 80)

    plt.tight_layout()
    out_path = vis_dir / "cpa_trajectory_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_metrics_summary(results_dir: Path, vis_dir: Path) -> Optional[Path]:
    """Plots comparative metric bar charts: CPA jitter, continuity, interventions."""
    csv_file = results_dir / "tracking_comparison_metrics.csv"
    if not csv_file.exists():
        return None
    df = pd.read_csv(csv_file)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(df))
    labels = df["variant"].str.replace("_", "\n")

    # 1. Mean CPA Jitter
    colors = ["#1f77b4" if "a5b" in v else ("#2ca02c" if "h3" in v else "#d62728") for v in df["variant"]]
    bars1 = ax1.bar(x, df["mean_cpa_jitter_deg"], color=colors, alpha=0.8, edgecolor="black")
    ax1.set_title("Mean Frame-to-Frame CPA Jitter [deg] (Lower is smoother)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("CPA Jitter [deg]")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}°", ha="center", va="bottom", fontsize=8)

    # 2. Mean Track Lifespan
    bars2 = ax2.bar(x, df["mean_lifespan_frames"], color=colors, alpha=0.8, edgecolor="black")
    ax2.set_title("Mean Track Lifespan [frames] (Higher indicates continuity)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Lifespan [frames]")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # 3. Interventions Count
    bars3 = ax3.bar(x, df["num_interventions"], color=colors, alpha=0.8, edgecolor="black")
    ax3.set_title("Temporal Evidence Interventions Count", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel("Interventions Count")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars3:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{int(b.get_height())}", ha="center", va="bottom", fontsize=8)

    # 4. Outward Motion Consistency
    bars4 = ax4.bar(x, df["outward_consistency_pct"], color=colors, alpha=0.8, edgecolor="black")
    ax4.set_title("Kinematic Outward Propagation Consistency [%]", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=8)
    ax4.set_ylabel("Consistency [%]")
    ax4.set_ylim(80, 105)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars4:
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Quantitative Tracking Benchmark: A5b vs H3 (Prior) vs H4 (Association)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = vis_dir / "tracking_metrics_summary.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main():
    results_dir = repo_root / "experiments/htm_pixel_temporal/results"
    vis_dir = repo_root / "experiments/htm_pixel_temporal/visualizations/tracking"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("Generating CPA trajectory comparison...")
    p1 = plot_cpa_trajectories(results_dir, vis_dir)
    print(f"Saved: {p1}")

    print("Generating metrics summary chart...")
    p2 = plot_metrics_summary(results_dir, vis_dir)
    if p2:
        print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
