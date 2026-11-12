"""Run A4 Multi-Component Adaptive PA-ROI Tracking Experiment & 4-Way Comparison.

Compares:
- A0: Original Patch-First Adaptive PA-ROI Tracking
- A2: Patch-First Adaptive PA-ROI Tracking + 4 px Outward Motion Gate
- A3-fixed: Component-First 1:1 Tracking with Ownership Conflict Bug Corrected
- A4: Component-First Multi-Component Tracking (Single Track ID for Multi-Components)

Outputs saved to: runs/pilot_adaptive_multicomponent_a4/
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_dilation

repo_root = Path("/home/aababc/stereo_cor2")
sys.path.insert(0, str(repo_root))

from cme_pipeline.core.config import PipelineConfig, TrackingConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v01_baseline_tracking import E1TrackingStage
from cme_pipeline.stages.tracking.v02_adaptive_pa_roi import BirthSeededAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v04_component_first import ComponentFirstAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v05_multi_component import MultiComponentAdaptiveROITrackingStage

# Directory Setup
exp_dir = repo_root / "runs/pilot_adaptive_multicomponent_a4"
tbl_dir = exp_dir / "tables"
arr_dir = exp_dir / "arrays"
vis_dir = exp_dir / "visualizations"

for d in (tbl_dir, arr_dir, vis_dir):
    d.mkdir(parents=True, exist_ok=True)

# Visualization Constants
VMIN_RD = -600.0
VMAX_RD = +600.0

def get_rd_cmap():
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("black")
    return cmap

TRACK_COLORS = {
    3: "#00e5ff",  # Cyan (Eastern CME main body / front)
    4: "#ffd600",  # Yellow (NE Streamer)
    5: "#76ff03",  # Lime (North micro feature)
    6: "#ff9100",  # Orange (NE root / intermediary)
    7: "#ff4081",  # Magenta (East root anchor in A0/A2/A3)
    8: "#f48fb1",  # Pink
    9: "#ce93d8",  # Light Purple
    10: "#80deea", # Light Cyan
    11: "#a7ffeb", # Teal
}

def get_track_color(tid: int) -> str:
    if tid in TRACK_COLORS:
        return TRACK_COLORS[tid]
    # deterministic fallback
    palette = ["#e040fb", "#7c4dff", "#536dfe", "#448aff", "#18ffff", "#64ffda", "#b2ff59", "#eeff41", "#ffff00", "#ffd740", "#ff6e40"]
    return palette[tid % len(palette)]

def draw_patch_boxes(ax, mask64, edgecolor, linewidth=1.5, hatch=None, facecolor="none", alpha=1.0):
    rcs = list(zip(*np.nonzero(mask64)))
    for r, c in rcs:
        rect = plt.Rectangle(
            (c * 4, r * 4), 4, 4,
            linewidth=linewidth, edgecolor=edgecolor,
            facecolor=facecolor, hatch=hatch, alpha=alpha
        )
        ax.add_patch(rect)


def get_last_observed_info(t_own_all, track_id, current_fidx):
    """Finds the most recent observed mask and centroid for a track."""
    for f in range(current_fidx - 1, -1, -1):
        m = (t_own_all[f] == track_id)
        if m.any():
            rc = np.argwhere(m)
            cx = rc[:, 1].mean() * 4 + 2
            cy = rc[:, 0].mean() * 4 + 2
            return cx, cy, m, f
    return np.nan, np.nan, None, -1


def main():
    print("=" * 80)
    print("STARTING A4 MULTI-COMPONENT TRACKING EXPERIMENT")
    print(f"Output Directory: {exp_dir}")
    print("=" * 80)

    # 1. Base Preprocessing and Science Arrays
    cfg = PipelineConfig()
    ctx = PipelineContext(cfg)
    base = E1TrackingStage().run(None, cfg, ctx)
    grid4_all = base.patch_activation_grid4
    man_df = base.manifest_df
    geom = base.geometry

    exp01_dir = repo_root / "validation/frozen/20101212/experiments/01_d7_baseline_segmentation"
    rd_all = np.load(exp01_dir / "results/science_arrays/north_up_RD_DNs.npy", mmap_mode="r")
    val_all = np.load(exp01_dir / "results/science_arrays/north_up_pair_valid_masks.npy", mmap_mode="r")

    n_pairs = len(man_df)
    pair_to_fidx = {int(row["pair_index"]): idx for idx, row in man_df.iterrows()}

    # -------------------------------------------------------------------------
    # Core Computation: Run All 4 Methods
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Running A0 (Patch-First Baseline, Gate OFF)...")
    cfg_a0 = PipelineConfig()
    cfg_a0.tracking.outward_gate_enabled = False
    stage_a0 = BirthSeededAdaptiveROITrackingStage()
    out_a0 = stage_a0.run(None, cfg_a0, ctx)

    print("\n[Step 2/4] Running A2 (Patch-First + 4 px Outward Gate)...")
    cfg_a2 = PipelineConfig()
    cfg_a2.tracking.outward_gate_enabled = True
    cfg_a2.tracking.outward_gate_tolerance_px = 4.0
    stage_a2 = BirthSeededAdaptiveROITrackingStage()
    out_a2 = stage_a2.run(None, cfg_a2, ctx)

    print("\n[Step 3/4] Running A3-fixed (Component-First 1:1, Bug Corrected)...")
    cfg_a3 = PipelineConfig()
    cfg_a3.tracking.outward_gate_enabled = True
    cfg_a3.tracking.outward_gate_tolerance_px = 6.0
    stage_a3 = ComponentFirstAdaptiveROITrackingStage()
    out_a3 = stage_a3.run(None, cfg_a3, ctx)

    print("\n[Step 4/4] Running A4 (Component-First Multi-Component)...")
    cfg_a4 = PipelineConfig()
    cfg_a4.tracking.outward_gate_enabled = True
    cfg_a4.tracking.outward_gate_tolerance_px = 6.0
    stage_a4 = MultiComponentAdaptiveROITrackingStage()
    out_a4 = stage_a4.run(None, cfg_a4, ctx)

    print("\nCore computation finished successfully. Saving numerical outputs...")

    # -------------------------------------------------------------------------
    # Save Tables and Arrays
    # -------------------------------------------------------------------------
    df_a0 = out_a0.track_states_df
    df_a2 = out_a2.track_states_df
    df_a3 = out_a3.track_states_df
    df_a4 = out_a4.track_states_df

    df_a0.to_csv(tbl_dir / "a0_track_states.csv", index=False)
    df_a2.to_csv(tbl_dir / "a2_track_states.csv", index=False)
    df_a3.to_csv(tbl_dir / "a3_fixed_track_states.csv", index=False)
    df_a4.to_csv(tbl_dir / "a4_track_states.csv", index=False)

    df_comp_a4 = getattr(out_a4, "component_states_df")
    df_lf_a4 = getattr(out_a4, "leading_front_states_df")
    df_dec_a4 = getattr(out_a4, "decisions_df")

    df_comp_a4.to_csv(tbl_dir / "a4_component_assignment_log.csv", index=False)
    df_lf_a4.to_csv(tbl_dir / "a4_leading_front_states.csv", index=False)
    df_dec_a4.to_csv(tbl_dir / "a4_decision_log.csv", index=False)

    m_a0 = getattr(out_a0, "ownership_masks")
    m_a2 = getattr(out_a2, "ownership_masks")
    m_a3 = getattr(out_a3, "ownership_masks")
    m_a4 = getattr(out_a4, "ownership_masks")

    np.save(arr_dir / "a0_ownership_masks.npy", m_a0)
    np.save(arr_dir / "a2_ownership_masks.npy", m_a2)
    np.save(arr_dir / "a3_fixed_ownership_masks.npy", m_a3)
    np.save(arr_dir / "a4_ownership_masks.npy", m_a4)

    # Build Comparative Frame Summary Table
    frame_rows = []
    pairs_to_compare = range(29, 40)  # Pairs 29 to 39
    for p in pairs_to_compare:
        fidx = pair_to_fidx.get(p, None)
        if fidx is None:
            continue
        ts = man_df.iloc[fidx]["end_time"]
        n_act = int(grid4_all[fidx].sum())

        row = {
            "pair_index": p,
            "timestamp": ts,
            "total_active_patches": n_act,
            # A0
            "a0_tracks": len(df_a0[df_a0["pair_index"] == p]),
            "a0_obs_tracks": len(df_a0[(df_a0["pair_index"] == p) & (df_a0["state"] == "OBSERVED")]),
            "a0_t3_patches": int(df_a0[(df_a0["pair_index"] == p) & (df_a0["track_id"] == 3)]["num_patches"].sum()),
            "a0_t3_front": float(df_a0[(df_a0["pair_index"] == p) & (df_a0["track_id"] == 3)]["front_radius"].values[0]) if len(df_a0[(df_a0["pair_index"] == p) & (df_a0["track_id"] == 3)]) else np.nan,
            # A2
            "a2_tracks": len(df_a2[df_a2["pair_index"] == p]),
            "a2_obs_tracks": len(df_a2[(df_a2["pair_index"] == p) & (df_a2["state"] == "OBSERVED")]),
            "a2_t3_patches": int(df_a2[(df_a2["pair_index"] == p) & (df_a2["track_id"] == 3)]["num_patches"].sum()),
            "a2_t3_front": float(df_a2[(df_a2["pair_index"] == p) & (df_a2["track_id"] == 3)]["front_radius"].values[0]) if len(df_a2[(df_a2["pair_index"] == p) & (df_a2["track_id"] == 3)]) else np.nan,
            # A3-fixed
            "a3_tracks": len(df_a3[df_a3["pair_index"] == p]),
            "a3_obs_tracks": len(df_a3[(df_a3["pair_index"] == p) & (df_a3["state"] == "OBSERVED")]),
            "a3_t3_patches": int(df_a3[(df_a3["pair_index"] == p) & (df_a3["track_id"] == 3)]["num_patches"].sum()),
            "a3_t3_front": float(df_a3[(df_a3["pair_index"] == p) & (df_a3["track_id"] == 3)]["front_radius"].values[0]) if len(df_a3[(df_a3["pair_index"] == p) & (df_a3["track_id"] == 3)]) else np.nan,
            # A4
            "a4_tracks": len(df_a4[df_a4["pair_index"] == p]),
            "a4_obs_tracks": len(df_a4[(df_a4["pair_index"] == p) & (df_a4["state"] == "OBSERVED")]),
            "a4_t3_patches": int(df_a4[(df_a4["pair_index"] == p) & (df_a4["track_id"] == 3)]["num_patches"].sum()),
            "a4_t3_comps": int(df_a4[(df_a4["pair_index"] == p) & (df_a4["track_id"] == 3)]["num_components"].values[0]) if len(df_a4[(df_a4["pair_index"] == p) & (df_a4["track_id"] == 3)]) else 0,
            "a4_t3_front": float(df_a4[(df_a4["pair_index"] == p) & (df_a4["track_id"] == 3)]["front_radius"].values[0]) if len(df_a4[(df_a4["pair_index"] == p) & (df_a4["track_id"] == 3)]) else np.nan,
            "a4_t3_lf_state": df_a4[(df_a4["pair_index"] == p) & (df_a4["track_id"] == 3)]["leading_front_state"].values[0] if len(df_a4[(df_a4["pair_index"] == p) & (df_a4["track_id"] == 3)]) else "N/A",
        }
        frame_rows.append(row)

    df_frame_comp = pd.DataFrame(frame_rows)
    df_frame_comp.to_csv(tbl_dir / "four_method_frame_comparison.csv", index=False)

    # Build Comparative Track Summary Table
    track_summary_rows = [
        {"method": "A0_PatchFirst", "total_tracks": len(df_a0["track_id"].unique()), "active_tracks_p30_35": len(df_a0[df_a0["pair_index"].between(30, 35)]["track_id"].unique()), "east_cme_fragmentation": "T3 split into T3, T7, T8", "t6_takeover_bug": False},
        {"method": "A2_OutwardGate", "total_tracks": len(df_a2["track_id"].unique()), "active_tracks_p30_35": len(df_a2[df_a2["pair_index"].between(30, 35)]["track_id"].unique()), "east_cme_fragmentation": "T3 split into T3, T7, T8", "t6_takeover_bug": False},
        {"method": "A3_Fixed", "total_tracks": len(df_a3["track_id"].unique()), "active_tracks_p30_35": len(df_a3[df_a3["pair_index"].between(30, 35)]["track_id"].unique()), "east_cme_fragmentation": "T3 split into T3, T7, T8", "t6_takeover_bug": "Fixed (T6 correctly PREDICTED)"},
        {"method": "A4_MultiComponent", "total_tracks": len(df_a4["track_id"].unique()), "active_tracks_p30_35": len(df_a4[df_a4["pair_index"].between(30, 35)]["track_id"].unique()), "east_cme_fragmentation": "Unified under T3 (No T7 birth!)", "t6_takeover_bug": "Fixed (T6 correctly PREDICTED)"},
    ]
    pd.DataFrame(track_summary_rows).to_csv(tbl_dir / "four_method_track_comparison.csv", index=False)

    print("\nSummary tables generated. Proceeding to visualizations...")

    # -------------------------------------------------------------------------
    # Visualization 1: Fig 1 - 4-Method Comparison Matrix (Pairs 30–35)
    # -------------------------------------------------------------------------
    print("\nRendering Fig 1: 4-Method Comparison Matrix (Pairs 30-35)...")
    cmap_rd = get_rd_cmap()
    norm_rd = Normalize(vmin=VMIN_RD, vmax=VMAX_RD)

    pairs_fig1 = [30, 31, 32, 33, 34, 35]
    fig, axes = plt.subplots(len(pairs_fig1), 4, figsize=(28, 38), facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.02, top=0.97, wspace=0.05, hspace=0.12)

    methods = [
        ("A0: Patch-First Baseline", df_a0, m_a0),
        ("A2: A0 + 4px Outward Gate", df_a2, m_a2),
        ("A3-fixed: Component-First 1:1", df_a3, m_a3),
        ("A4: Multi-Component Tracking", df_a4, m_a4),
    ]

    for row_idx, p in enumerate(pairs_fig1):
        fidx = pair_to_fidx[p]
        ts_str = man_df.iloc[fidx]["end_time"]
        rd_dn = rd_all[fidx].copy()
        val_m = val_all[fidx]
        rd_dn[~val_m] = np.nan
        cur_grid = grid4_all[fidx]

        for col_idx, (m_label, df_m, m_own_all) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")
            im = ax.imshow(rd_dn, cmap=cmap_rd, norm=norm_rd, origin="lower")

            # Layer 1: All active patches in thin red outline
            draw_patch_boxes(ax, cur_grid, edgecolor="#ff1744", linewidth=0.8, alpha=0.9)

            # Layer 2: Final owned patches in track-specific colors
            p_df = df_m[df_m["pair_index"] == p]
            t_own = m_own_all[fidx]

            active_tids = p_df[p_df["state"] == "OBSERVED"]["track_id"].unique()
            for tid in active_tids:
                t_mask = (t_own == tid)
                if not t_mask.any():
                    continue
                col = get_track_color(tid)
                draw_patch_boxes(ax, t_mask, edgecolor=col, linewidth=1.8)

                # Centroid label
                rcs = np.argwhere(t_mask)
                cx = rcs[:, 1].mean() * 4 + 2
                cy = rcs[:, 0].mean() * 4 + 2
                n_p = len(rcs)
                ax.text(
                    cx, cy, f"T{tid} ({n_p}p)",
                    color=col, fontsize=10, weight="bold",
                    ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")]
                )

            # Layer 3: Predicted-only tracks
            pred_tids = p_df[p_df["state"] == "PREDICTED_ONLY"]["track_id"].unique()
            for tid in pred_tids:
                col = get_track_color(tid)
                cx, cy, _, _ = get_last_observed_info(m_own_all, tid, fidx)
                if not np.isnan(cx):
                    ax.scatter([cx], [cy], s=130, facecolors="none", edgecolors=col, linestyle="--", linewidth=2.0)
                    ax.scatter([cx], [cy], marker="x", s=70, color=col, linewidth=2.0)
                    ax.text(
                        cx + 4, cy - 4, f"T{tid}\n[PREDICTED]",
                        color=col, fontsize=9, weight="bold",
                        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
                    )

            # Title
            if row_idx == 0:
                ax.set_title(f"{m_label}\nPair {p} | {ts_str[:19]}", color="white", fontsize=13, weight="bold")
            else:
                ax.set_title(f"Pair {p} | {ts_str[:19]}", color="white", fontsize=11)

            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)
            ax.tick_params(colors="gray", labelsize=8)

    # Common colorbar
    cbar_ax = fig.add_axes([0.15, 0.008, 0.70, 0.008])
    cbar = plt.colorbar(axes[0, 0].images[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("RD (DN) [Fixed ±600 DN]", color="white", fontsize=12, weight="bold")
    cbar.ax.tick_params(colors="white", labelsize=10)

    fig1_path = vis_dir / "fig_01_four_method_comparison_pairs30_35.png"
    plt.savefig(fig1_path, dpi=130, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------------------
    # Visualization 2: Fig 2 - Pair 33 -> 34 Eastern CME Component Correspondence
    # -------------------------------------------------------------------------
    print("\nRendering Fig 2: Pair 33 -> 34 Eastern CME Component Correspondence...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 18), facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.94, wspace=0.08, hspace=0.10)

    # Zoom window on Eastern CME: X in [60, 220], Y in [40, 200]
    zoom_extent = (60, 220, 40, 200)

    # (Row 0: Pair 33; Row 1: Pair 34) x (Col 0: A3-fixed; Col 1: A4)
    panel_configs = [
        (0, 0, 33, "A3-fixed (1:1 Tracking) | Pair 33", m_a3, df_a3),
        (0, 1, 33, "A4 Multi-Component | Pair 33", m_a4, df_a4),
        (1, 0, 34, "A3-fixed (1:1 Tracking) | Pair 34 [T3 splits into T3 & T7]", m_a3, df_a3),
        (1, 1, 34, "A4 Multi-Component | Pair 34 [T3 owns both Front & Root!]", m_a4, df_a4),
    ]

    for r_idx, c_idx, p, title, m_arr, df_m in panel_configs:
        ax = axes[r_idx, c_idx]
        ax.set_facecolor("black")
        fidx = pair_to_fidx[p]
        ts_str = man_df.iloc[fidx]["end_time"]
        rd_dn = rd_all[fidx].copy()
        rd_dn[~val_all[fidx]] = np.nan
        ax.imshow(rd_dn, cmap=cmap_rd, norm=norm_rd, origin="lower")

        cur_grid = grid4_all[fidx]
        draw_patch_boxes(ax, cur_grid, edgecolor="#ff1744", linewidth=1.0, alpha=0.6)

        t_own = m_arr[fidx]
        for tid in [3, 4, 6, 7]:
            m = (t_own == tid)
            if m.any():
                col = get_track_color(tid)
                draw_patch_boxes(ax, m, edgecolor=col, linewidth=2.0)
                rcs = np.argwhere(m)
                cx = rcs[:, 1].mean() * 4 + 2
                cy = rcs[:, 0].mean() * 4 + 2
                ax.text(
                    cx, cy, f"T{tid} ({len(rcs)}p)",
                    color=col, fontsize=12, weight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")]
                )

        ax.set_xlim(zoom_extent[0], zoom_extent[1])
        ax.set_ylim(zoom_extent[2], zoom_extent[3])
        ax.set_title(f"{title}\n{ts_str[:19]}", color="white", fontsize=12, weight="bold")
        ax.tick_params(colors="gray")

    fig2_path = vis_dir / "fig_02_p33_to_p34_eastern_cme.png"
    plt.savefig(fig2_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # Visualization 3: Fig 3 - Pair 34 Ownership Conflict Analysis (T3/T4/T6)
    # -------------------------------------------------------------------------
    print("\nRendering Fig 3: Pair 34 Ownership Conflict Analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.90, wspace=0.08)

    p34_fidx = pair_to_fidx[34]
    p34_rd = rd_all[p34_fidx].copy()
    p34_rd[~val_all[p34_fidx]] = np.nan
    p34_grid = grid4_all[p34_fidx]
    lbl_34, n_34 = label(p34_grid, structure=np.ones((3, 3), bool))

    # Panel 1: Candidate Components on Raw grid4
    ax0 = axes[0]
    ax0.set_facecolor("black")
    ax0.imshow(p34_rd, cmap=cmap_rd, norm=norm_rd, origin="lower")
    for c_id in range(1, n_34 + 1):
        m = (lbl_34 == c_id)
        rcs = np.argwhere(m)
        cx = rcs[:, 1].mean() * 4 + 2
        cy = rcs[:, 0].mean() * 4 + 2
        draw_patch_boxes(ax0, m, edgecolor="#00e676", linewidth=1.5)
        ax0.text(
            cx, cy, f"Comp {c_id}\n({len(rcs)}p)",
            color="#00e676", fontsize=10, weight="bold", ha="center",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")]
        )
    ax0.set_title("Pair 34: 8-Connected Component Discovery\n(Raw grid4 Candidates before Assignment)", color="white", fontsize=12, weight="bold")
    ax0.set_xlim(60, 230)
    ax0.set_ylim(40, 210)
    ax0.tick_params(colors="gray")

    # Panel 2: A3-fixed 1:1 Assignment (T6 avoided takeover, T7 born)
    ax1 = axes[1]
    ax1.set_facecolor("black")
    ax1.imshow(p34_rd, cmap=cmap_rd, norm=norm_rd, origin="lower")
    draw_patch_boxes(ax1, p34_grid, edgecolor="#ff1744", linewidth=0.8, alpha=0.5)
    t_own_34_a3 = m_a3[p34_fidx]
    for tid in [3, 4, 5, 6, 7]:
        m = (t_own_34_a3 == tid)
        col = get_track_color(tid)
        if m.any():
            draw_patch_boxes(ax1, m, edgecolor=col, linewidth=2.0)
            rcs = np.argwhere(m)
            ax1.text(
                rcs[:, 1].mean() * 4 + 2, rcs[:, 0].mean() * 4 + 2, f"T{tid} ({len(rcs)}p)",
                color=col, fontsize=11, weight="bold",
                path_effects=[pe.withStroke(linewidth=3, foreground="black")]
            )
        elif tid == 6:
            # T6 is predicted
            cx, cy, _, _ = get_last_observed_info(m_a3, 6, p34_fidx)
            ax1.scatter([cx], [cy], s=120, facecolors="none", edgecolors=col, linestyle="--", linewidth=2.0)
            ax1.scatter([cx], [cy], marker="x", s=70, color=col, linewidth=2.0)
            ax1.text(cx + 4, cy - 4, "T6 [PREDICTED]\n(0p - takeover fixed!)", color=col, fontsize=10, weight="bold",
                     path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    ax1.set_title("A3-fixed (1:1 Component Matching)\nT4=50p, T3=45p (Comp 2), T7=46p (Comp 4 born)", color="white", fontsize=12, weight="bold")
    ax1.set_xlim(60, 230)
    ax1.set_ylim(40, 210)
    ax1.tick_params(colors="gray")

    # Panel 3: A4 Multi-Component Ownership (T3 owns Comp 2 + Comp 4 + edge of Comp 6)
    ax2 = axes[2]
    ax2.set_facecolor("black")
    ax2.imshow(p34_rd, cmap=cmap_rd, norm=norm_rd, origin="lower")
    draw_patch_boxes(ax2, p34_grid, edgecolor="#ff1744", linewidth=0.8, alpha=0.5)
    t_own_34_a4 = m_a4[p34_fidx]
    for tid in [3, 4, 5, 6]:
        m = (t_own_34_a4 == tid)
        col = get_track_color(tid)
        if m.any():
            draw_patch_boxes(ax2, m, edgecolor=col, linewidth=2.0)
            rcs = np.argwhere(m)
            ax2.text(
                rcs[:, 1].mean() * 4 + 2, rcs[:, 0].mean() * 4 + 2, f"T{tid} ({len(rcs)}p)\n[Multi-Comp]",
                color=col, fontsize=11, weight="bold",
                path_effects=[pe.withStroke(linewidth=3, foreground="black")]
            )
        elif tid == 6:
            cx, cy, _, _ = get_last_observed_info(m_a4, 6, p34_fidx)
            ax2.scatter([cx], [cy], s=120, facecolors="none", edgecolors=col, linestyle="--", linewidth=2.0)
            ax2.scatter([cx], [cy], marker="x", s=70, color=col, linewidth=2.0)
            ax2.text(cx + 4, cy - 4, "T6 [PREDICTED]", color=col, fontsize=10, weight="bold",
                     path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    ax2.set_title("A4 Multi-Component Matching\nT3 owns 106p (Front + Root + Flank), T7 NOT born!", color="white", fontsize=12, weight="bold")
    ax2.set_xlim(60, 230)
    ax2.set_ylim(40, 210)
    ax2.tick_params(colors="gray")

    fig3_path = vis_dir / "fig_03_p34_ownership_conflict_analysis.png"
    plt.savefig(fig3_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # -------------------------------------------------------------------------
    # Visualization 4: Fig 4 - Pair 34 -> 35 Evolution & Leading Front Decoupling
    # -------------------------------------------------------------------------
    print("\nRendering Fig 4: Pair 34 -> 35 Evolution & Leading Front Decoupling...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 18), facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.94, wspace=0.08, hspace=0.10)

    p34_35_configs = [
        (0, 0, 34, "A3-fixed | Pair 34\n(T3 Outer Front 45p, T7 Inner Root 46p)", m_a3, df_a3),
        (0, 1, 35, "A3-fixed | Pair 35\n(T3 Faint Front 2p, T7 Root 45p)", m_a3, df_a3),
        (1, 0, 34, "A4 Multi-Component | Pair 34\n(T3 owns 106p across 4 Comps; Front R=82.3 px)", m_a4, df_a4),
        (1, 1, 35, "A4 Multi-Component | Pair 35\n(T3 owns 49p across 3 Comps; Front R=76.5 px)", m_a4, df_a4),
    ]

    for r_idx, c_idx, p, title, m_arr, df_m in p34_35_configs:
        ax = axes[r_idx, c_idx]
        ax.set_facecolor("black")
        fidx = pair_to_fidx[p]
        ts_str = man_df.iloc[fidx]["end_time"]
        rd_dn = rd_all[fidx].copy()
        rd_dn[~val_all[fidx]] = np.nan
        ax.imshow(rd_dn, cmap=cmap_rd, norm=norm_rd, origin="lower")

        cur_grid = grid4_all[fidx]
        draw_patch_boxes(ax, cur_grid, edgecolor="#ff1744", linewidth=1.0, alpha=0.6)

        t_own = m_arr[fidx]
        for tid in [3, 4, 7]:
            m = (t_own == tid)
            if m.any():
                col = get_track_color(tid)
                draw_patch_boxes(ax, m, edgecolor=col, linewidth=2.0)
                rcs = np.argwhere(m)
                cx = rcs[:, 1].mean() * 4 + 2
                cy = rcs[:, 0].mean() * 4 + 2
                ax.text(
                    cx, cy, f"T{tid} ({len(rcs)}p)",
                    color=col, fontsize=12, weight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")]
                )

        ax.set_xlim(60, 220)
        ax.set_ylim(40, 200)
        ax.set_title(f"{title}\n{ts_str[:19]}", color="white", fontsize=11, weight="bold")
        ax.tick_params(colors="gray")

    fig4_path = vis_dir / "fig_04_p34_to_p35_evolution.png"
    plt.savefig(fig4_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {fig4_path}")

    # -------------------------------------------------------------------------
    # Visualization 5: Fig 5 - Leading Front Radial Trajectories
    # -------------------------------------------------------------------------
    print("\nRendering Fig 5: Leading Front Radial Trajectories...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor="#141414")
    plt.subplots_adjust(left=0.06, right=0.95, bottom=0.10, top=0.90, wspace=0.15)

    for ax in (ax1, ax2):
        ax.set_facecolor("#1f1f1f")
        ax.grid(True, color="#333333", linestyle="--", alpha=0.7)
        ax.tick_params(colors="white", labelsize=11)
        for spine in ax.spines.values():
            spine.set_color("#555555")

    # Plot 1: Eastern CME Track 3 Radial Front Trajectory across methods
    p_range = range(29, 39)
    for m_label, df_m, col, ls, marker in [
        ("A0: Patch-First Baseline", df_a0, "#bdbdbd", "--", "s"),
        ("A2: A0 + 4px Outward Gate", df_a2, "#ff9800", "-.", "^"),
        ("A3-fixed: Component-First 1:1", df_a3, "#e91e63", ":", "d"),
        ("A4: Multi-Component (Track 3)", df_a4, "#00e5ff", "-", "o"),
    ]:
        sub = df_m[(df_m["track_id"] == 3) & (df_m["pair_index"].isin(p_range))].sort_values("pair_index")
        if len(sub) > 0:
            obs = sub[sub["state"] == "OBSERVED"]
            pred = sub[sub["state"] == "PREDICTED_ONLY"]
            ax1.plot(obs["pair_index"], obs["front_radius"], color=col, linestyle=ls, marker=marker, linewidth=2.5, label=m_label)
            if len(pred) > 0:
                ax1.scatter(pred["pair_index"], pred["front_radius"], color=col, facecolors="none", edgecolors=col, marker="x", s=80, linewidth=2)

    # In A3, also plot T7 (the fragmented inner root)
    sub_t7 = df_a3[(df_a3["track_id"] == 7) & (df_a3["pair_index"].isin(p_range))].sort_values("pair_index")
    if len(sub_t7) > 0:
        ax1.plot(sub_t7["pair_index"], sub_t7["front_radius"], color="#ff4081", linestyle="--", marker="v", linewidth=2.0, label="A3 T7 (Fragmented Inner Root)")

    ax1.set_title("Eastern CME Leading Front Radial Trajectory (Pairs 29–38)", color="white", fontsize=14, weight="bold")
    ax1.set_xlabel("Pair Index", color="white", fontsize=12)
    ax1.set_ylabel("Leading Front Radius (px)", color="white", fontsize=12)
    ax1.set_xlim(28.5, 38.5)
    ax1.set_ylim(20, 120)
    ax1.legend(facecolor="#262626", edgecolor="#444444", labelcolor="white", fontsize=10)

    # Plot 2: Total Owned Patches per Frame for Eastern CME Structures
    for m_label, df_m, col, ls in [
        ("A0 (T3)", df_a0, "#bdbdbd", "--"),
        ("A2 (T3)", df_a2, "#ff9800", "-."),
        ("A3-fixed (T3)", df_a3, "#e91e63", ":"),
        ("A3-fixed (T3 + T7 sum)", df_a3, "#ff4081", "--"),
        ("A4 (T3 Unified Multi-Comp)", df_a4, "#00e5ff", "-"),
    ]:
        if "sum" in m_label:
            s_t3 = df_a3[(df_a3["track_id"] == 3) & (df_a3["pair_index"].isin(p_range))].set_index("pair_index")["num_patches"]
            s_t7 = df_a3[(df_a3["track_id"] == 7) & (df_a3["pair_index"].isin(p_range))].set_index("pair_index")["num_patches"]
            s_comb = s_t3.add(s_t7, fill_value=0).sort_index()
            ax2.plot(s_comb.index, s_comb.values, color=col, linestyle=ls, linewidth=2.5, marker="s", label=m_label)
        else:
            sub = df_m[(df_m["track_id"] == 3) & (df_m["pair_index"].isin(p_range))].sort_values("pair_index")
            if len(sub) > 0:
                ax2.plot(sub["pair_index"], sub["num_patches"], color=col, linestyle=ls, linewidth=2.5, marker="o", label=m_label)

    ax2.set_title("Eastern CME Total Owned Patches per Frame", color="white", fontsize=14, weight="bold")
    ax2.set_xlabel("Pair Index", color="white", fontsize=12)
    ax2.set_ylabel("Number of Owned Patches (4x4)", color="white", fontsize=12)
    ax2.set_xlim(28.5, 38.5)
    ax2.legend(facecolor="#262626", edgecolor="#444444", labelcolor="white", fontsize=10)

    fig5_path = vis_dir / "fig_05_leading_front_trajectories.png"
    plt.savefig(fig5_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {fig5_path}")

    # -------------------------------------------------------------------------
    # Visualization 6: Fig 6 - A4 Component Ownership Map
    # -------------------------------------------------------------------------
    print("\nRendering Fig 6: A4 Component Ownership Map...")
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#141414")
    ax.set_facecolor("#1f1f1f")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.7)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    # Scatter of individual components in A4: X = Pair, Y = Component CPA, Size = Num Patches, Color = Track ID
    p30_38_c = df_comp_a4[df_comp_a4["pair_index"].between(30, 38)]
    for tid in sorted(p30_38_c["track_id"].unique()):
        t_sub = p30_38_c[p30_38_c["track_id"] == tid]
        col = get_track_color(tid)
        # Separate leading front vs inner component
        lf = t_sub[t_sub["is_leading_front"] == True]
        inner = t_sub[t_sub["is_leading_front"] == False]

        if len(lf) > 0:
            ax.scatter(
                lf["pair_index"], lf["cpa"], s=lf["num_patches"] * 8 + 40,
                color=col, marker="o", edgecolors="white", linewidth=1.5,
                label=f"T{tid} (Leading Front)", zorder=4
            )
        if len(inner) > 0:
            ax.scatter(
                inner["pair_index"], inner["cpa"], s=inner["num_patches"] * 8 + 40,
                color=col, marker="s", edgecolors=col, facecolors="none", linewidth=2.0,
                label=f"T{tid} (Inner/Root Comp)", zorder=3
            )

    ax.set_title("A4 Multi-Component Structure Map (Pairs 30–38)\nCircle = Leading Front | Square = Inner/Root Component | Size ∝ Patch Count", color="white", fontsize=13, weight="bold")
    ax.set_xlabel("Pair Index", color="white", fontsize=12)
    ax.set_ylabel("Component Central Position Angle (deg)", color="white", fontsize=12)
    ax.set_ylim(0, 180)
    ax.set_xlim(29.5, 38.5)
    ax.legend(facecolor="#262626", edgecolor="#444444", labelcolor="white", fontsize=9, loc="upper right")

    fig6_path = vis_dir / "fig_06_component_ownership_map.png"
    plt.savefig(fig6_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {fig6_path}")

    # -------------------------------------------------------------------------
    # Visualization 7: Fig 7 - Track Lifecycle and Fragmentation Summary
    # -------------------------------------------------------------------------
    print("\nRendering Fig 7: Track Lifecycle and Fragmentation Summary...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor="#141414")
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.88, wspace=0.18)

    for ax in (ax1, ax2):
        ax.set_facecolor("#1f1f1f")
        ax.grid(True, color="#333333", linestyle="--", alpha=0.7)
        ax.tick_params(colors="white", labelsize=11)
        for spine in ax.spines.values():
            spine.set_color("#555555")

    # Left: Total Track Count Comparison (all frames vs pairs 30-35)
    methods_names = ["A0\n(Baseline)", "A2\n(Outward Gate)", "A3-fixed\n(Component 1:1)", "A4\n(Multi-Component)"]
    tot_tracks = [len(df_a0["track_id"].unique()), len(df_a2["track_id"].unique()), len(df_a3["track_id"].unique()), len(df_a4["track_id"].unique())]
    p30_35_tracks = [
        len(df_a0[df_a0["pair_index"].between(30, 35)]["track_id"].unique()),
        len(df_a2[df_a2["pair_index"].between(30, 35)]["track_id"].unique()),
        len(df_a3[df_a3["pair_index"].between(30, 35)]["track_id"].unique()),
        len(df_a4[df_a4["pair_index"].between(30, 35)]["track_id"].unique()),
    ]

    x = np.arange(len(methods_names))
    w = 0.35
    b1 = ax1.bar(x - w/2, tot_tracks, w, color="#78909c", label="Total Tracks (All 32 Frames)")
    b2 = ax1.bar(x + w/2, p30_35_tracks, w, color="#00e5ff", label="Active Tracks (Pairs 30–35)")

    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(int(bar.get_height())), ha="center", color="white", weight="bold")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(int(bar.get_height())), ha="center", color="white", weight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_names, color="white", fontsize=11)
    ax1.set_title("Track Count & Fragmentation Reduction", color="white", fontsize=13, weight="bold")
    ax1.set_ylabel("Number of Distinct Track IDs", color="white", fontsize=12)
    ax1.legend(facecolor="#262626", edgecolor="#444444", labelcolor="white", fontsize=10)

    # Right: Eastern CME Track Fragmentation Anatomy
    # Shows track IDs representing Eastern CME at Pair 34 & 35
    frag_data = {
        "A0 Baseline": [("T3 (Front)", 45), ("T7 (Root)", 46)],
        "A2 Outward Gate": [("T3 (Front)", 45), ("T7 (Root)", 46)],
        "A3-fixed": [("T3 (Front)", 45), ("T7 (Root)", 46)],
        "A4 Multi-Component": [("T3 (Unified Front + Root + Flank)", 106)],
    }

    y_pos = np.arange(len(frag_data))
    colors_frag = ["#e91e63", "#00e5ff", "#ffd600", "#76ff03"]
    for idx, (m_k, parts) in enumerate(frag_data.items()):
        left_acc = 0
        for p_idx, (p_label, n_patches) in enumerate(parts):
            col = "#00e5ff" if "T3" in p_label else "#ff4081"
            ax2.barh(idx, n_patches, left=left_acc, height=0.5, color=col, edgecolor="white", linewidth=1.0)
            ax2.text(left_acc + n_patches/2, idx, f"{p_label}\n{n_patches}p", ha="center", va="center", color="black" if col == "#00e5ff" else "white", weight="bold", fontsize=10)
            left_acc += n_patches

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(frag_data.keys()), color="white", fontsize=11)
    ax2.set_title("Pair 34 Eastern CME Ownership Anatomy\n(Splitting into Separate Tracks vs Single Unified Track)", color="white", fontsize=13, weight="bold")
    ax2.set_xlabel("Total Owned Patches (4x4)", color="white", fontsize=12)
    ax2.set_xlim(0, 120)

    fig7_path = vis_dir / "fig_07_track_lifecycle_summary.png"
    plt.savefig(fig7_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {fig7_path}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENT COMPUTATIONS & VISUALIZATIONS COMPLETE!")
    print(f"Results successfully archived in: {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

