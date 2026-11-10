"""Full-Day (2010-12-12) Corrected Visualization & 4-Method Comparison.

Corrects the background RD indexing:
- north_up_RD_DNs.npy is indexed as p_raw = p - 1 (for Pair p).
- Generates 32 individual high-resolution PNG frames in frames/.
- Compiles full-day animated GIF in animations/.
- Generates 3 key-interval summary multi-panel sheets in summary_sheets/.
- Conforms strictly to VISUALIZATION_STANDARDS.md.

Outputs saved to: runs/pilot_adaptive_multicomponent_a4/full_day_corrected/
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import imageio.v3 as iio

repo_root = Path("/home/aababc/stereo_cor2")
sys.path.insert(0, str(repo_root))

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.e1_tracking import E1TrackingStage

# Output Directories
base_out_dir = repo_root / "runs/pilot_adaptive_multicomponent_a4"
full_day_dir = base_out_dir / "full_day_corrected"
frames_dir = full_day_dir / "frames"
anim_dir = full_day_dir / "animations"
summ_dir = full_day_dir / "summary_sheets"
tbl_out_dir = full_day_dir / "tables"

for d in (frames_dir, anim_dir, summ_dir, tbl_out_dir):
    d.mkdir(parents=True, exist_ok=True)

tbl_dir = base_out_dir / "tables"
arr_dir = base_out_dir / "arrays"

# Constants
VMIN_RD = -600.0
VMAX_RD = +600.0
SX, SY = 126.618, 127.503
R_OCCULTER_PX = 22.0
R_FOV_PX = 122.0

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
    palette = ["#e040fb", "#7c4dff", "#536dfe", "#448aff", "#18ffff", "#64ffda", "#b2ff59", "#eeff41", "#ffff00", "#ffd740", "#ff6e40"]
    return palette[tid % len(palette)]

def draw_patch_boxes(ax, mask64, edgecolor, linewidth=1.5):
    rcs = list(zip(*np.nonzero(mask64)))
    for r, c in rcs:
        rect = plt.Rectangle(
            (c * 4, r * 4), 4, 4,
            linewidth=linewidth, edgecolor=edgecolor,
            facecolor="none", alpha=1.0
        )
        ax.add_patch(rect)

def get_last_observed_info(t_own_all, track_id, current_fidx):
    for f in range(current_fidx - 1, -1, -1):
        m = (t_own_all[f] == track_id)
        if m.any():
            rc = np.argwhere(m)
            cx = rc[:, 1].mean() * 4 + 2
            cy = rc[:, 0].mean() * 4 + 2
            return cx, cy, m, f
    return np.nan, np.nan, None, -1

def add_geometry_decorations(ax):
    # Inner occulter circle
    occ = Circle((SX, SY), R_OCCULTER_PX, color="white", fill=False, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.add_patch(occ)
    # Outer valid FOV circle
    fov = Circle((SX, SY), R_FOV_PX, color="white", fill=False, linestyle=":", linewidth=1.0, alpha=0.4)
    ax.add_patch(fov)


def main():
    print("=" * 80)
    print("STARTING FULL-DAY (2010-12-12) CORRECTED VISUALIZATION")
    print(f"Output Directory: {full_day_dir}")
    print("=" * 80)

    # 1. Base Preprocessing and Science Arrays
    cfg = PipelineConfig()
    ctx = PipelineContext(cfg)
    base = E1TrackingStage().run(None, cfg, ctx)
    grid4_all = base.patch_activation_grid4
    man_df = base.manifest_df

    exp01_dir = repo_root / "validation_20101212/experiments/01_d7_baseline_segmentation"
    rd_all = np.load(exp01_dir / "results/science_arrays/north_up_RD_DNs.npy", mmap_mode="r")
    val_all = np.load(exp01_dir / "results/science_arrays/north_up_pair_valid_masks.npy", mmap_mode="r")

    # Load Tracking States & Ownership Masks
    df_a0 = pd.read_csv(tbl_dir / "a0_track_states.csv")
    df_a2 = pd.read_csv(tbl_dir / "a2_track_states.csv")
    df_a3 = pd.read_csv(tbl_dir / "a3_fixed_track_states.csv")
    df_a4 = pd.read_csv(tbl_dir / "a4_track_states.csv")

    m_a0 = np.load(arr_dir / "a0_ownership_masks.npy")
    m_a2 = np.load(arr_dir / "a2_ownership_masks.npy")
    m_a3 = np.load(arr_dir / "a3_fixed_ownership_masks.npy")
    m_a4 = np.load(arr_dir / "a4_ownership_masks.npy")

    cmap_rd = get_rd_cmap()
    norm_rd = Normalize(vmin=VMIN_RD, vmax=VMAX_RD)

    methods = [
        ("A0: Patch-First Baseline", df_a0, m_a0),
        ("A2: A0 + 4px Outward Gate", df_a2, m_a2),
        ("A3-fixed: Component-First 1:1", df_a3, m_a3),
        ("A4: Multi-Component Tracking", df_a4, m_a4),
    ]

    n_pairs = len(man_df)
    frame_png_paths = []

    print(f"\n[Step 1/3] Generating 32 Full-Day High-Resolution Frame PNGs...")

    for fidx in range(n_pairs):
        row_man = man_df.iloc[fidx]
        p = int(row_man["pair_index"])
        p_raw = p - 1  # Crucial fix: exact frame index in north_up_RD_DNs.npy
        ts_str = row_man["end_time"]
        rd_id = row_man["rd_pair_id"]
        dt_min = float(row_man["delta_t_min"])
        is_large_gap = bool(row_man["is_large_gap"])

        # Prepare display RD image
        disp_rd = np.full((256, 256), np.nan, dtype=np.float32)
        disp_rd[val_all[p_raw]] = rd_all[p_raw][val_all[p_raw]]
        cur_grid = grid4_all[fidx]

        fig, axes = plt.subplots(1, 4, figsize=(24, 6.8), facecolor="#0e0e0e", dpi=120)
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.88, wspace=0.06)

        # Super title & telemetry banner
        if is_large_gap:
            banner_txt = f"Pair {p:02d} ({rd_id}) — {ts_str[:19]} UTC | Δt = {dt_min:.1f}m  [⚠ CADENCE GAP — Tracks Terminated]"
            title_color = "#ffb300"
        else:
            banner_txt = f"Pair {p:02d} ({rd_id}) — {ts_str[:19]} UTC | Δt = {dt_min:.1f}m"
            title_color = "#ffffff"

        fig.suptitle(banner_txt, color=title_color, fontsize=13, weight="bold", y=0.96)

        for col_idx, (m_label, df_m, m_own_all) in enumerate(methods):
            ax = axes[col_idx]
            ax.set_facecolor("black")
            im = ax.imshow(disp_rd, cmap=cmap_rd, norm=norm_rd, origin="lower", extent=[0, 256, 0, 256])
            add_geometry_decorations(ax)

            # Layer 1: Global active patches in thin red outline
            draw_patch_boxes(ax, cur_grid, edgecolor="#ff1744", linewidth=0.8)

            # Layer 2: Final owned patches in track-specific colors
            p_df = df_m[df_m["pair_index"] == p]
            t_own = m_own_all[fidx]

            obs_df = p_df[p_df["state"] == "OBSERVED"]
            for _, r_state in obs_df.iterrows():
                tid = int(r_state["track_id"])
                t_mask = (t_own == tid)
                if not t_mask.any():
                    continue
                col = get_track_color(tid)
                draw_patch_boxes(ax, t_mask, edgecolor=col, linewidth=1.8)

                rcs = np.argwhere(t_mask)
                cx = rcs[:, 1].mean() * 4 + 2
                cy = rcs[:, 0].mean() * 4 + 2
                ax.text(
                    cx, cy, f"T{tid} ({len(rcs)}p)",
                    color=col, fontsize=9.5, weight="bold",
                    ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")]
                )

            # Layer 3: Predicted-only tracks
            pred_df = p_df[p_df["state"] == "PREDICTED_ONLY"]
            for _, r_state in pred_df.iterrows():
                tid = int(r_state["track_id"])
                col = get_track_color(tid)
                cx, cy, _, _ = get_last_observed_info(m_own_all, tid, fidx)
                if not np.isnan(cx):
                    ax.scatter([cx], [cy], s=110, facecolors="none", edgecolors=col, linestyle="--", linewidth=1.8)
                    ax.scatter([cx], [cy], marker="x", s=60, color=col, linewidth=1.8)
                    ax.text(
                        cx + 4, cy - 4, f"T{tid}\n[PRED]",
                        color=col, fontsize=8.5, weight="bold",
                        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
                    )

            n_obs = len(obs_df)
            n_pred = len(pred_df)
            tot_p = int(obs_df["num_patches"].sum()) if len(obs_df) else 0

            ax.set_title(
                f"{m_label}\nActive: {n_obs + n_pred} (Obs: {n_obs}, Pred: {n_pred}) | {tot_p}p owned",
                color="#00e5ff" if "A4" in m_label else "white", fontsize=10.5, weight="bold"
            )
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)
            ax.set_xticks([])
            ax.set_yticks([])

        # Colorbar
        cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
        cbar = plt.colorbar(axes[0].images[0], cax=cbar_ax, orientation="horizontal")
        cbar.set_label("RD (DN) [Fixed ±600 DN]", color="white", fontsize=10, weight="bold")
        cbar.ax.tick_params(colors="white", labelsize=9)

        frame_path = frames_dir / f"pair_{p:03d}.png"
        plt.savefig(frame_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        frame_png_paths.append(frame_path)

        if (fidx + 1) % 8 == 0 or fidx == n_pairs - 1:
            print(f"  Processed {fidx + 1}/{n_pairs} frames ({frame_path.name})")

    # -------------------------------------------------------------------------
    # Step 2: Compile Full-Day Animated GIF
    # -------------------------------------------------------------------------
    print("\n[Step 2/3] Compiling Full-Day Comparison GIF (32 frames @ 500 ms / frame)...")
    gif_path = anim_dir / "full_day_four_method_comparison.gif"
    images = [iio.imread(p) for p in frame_png_paths]
    iio.imwrite(gif_path, images, duration=500, loop=0)
    print(f"  Saved Animation: {gif_path.name} ({gif_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # -------------------------------------------------------------------------
    # Step 3: Generate 3 Key-Interval Multi-Panel Summary Sheets
    # -------------------------------------------------------------------------
    print("\n[Step 3/3] Generating 3 Key-Interval Summary Multi-Panel Sheets...")

    intervals = [
        ("05:24–08:24 UTC (Pairs 30, 32, 34, 36) — Morning Primary CME", [30, 32, 34, 36], "summary_01_morning_pairs_30_36.png"),
        ("11:54–14:54 UTC (Pairs 43, 45, 47, 49) — Afternoon CME Flank & Streamer", [43, 45, 47, 49], "summary_02_afternoon_pairs_43_49.png"),
        ("19:24–22:24 UTC (Pairs 51, 53, 55, 57) — Evening CME Propagation & Recovery", [51, 53, 55, 57], "summary_03_evening_pairs_51_57.png"),
    ]

    for int_name, target_pairs, out_name in intervals:
        n_rows = len(target_pairs)
        fig, axes = plt.subplots(n_rows, 4, figsize=(24, 5.5 * n_rows), facecolor="#0e0e0e", dpi=120)
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.95, wspace=0.05, hspace=0.15)
        fig.suptitle(f"Key Diagnostic Review: {int_name}", color="white", fontsize=15, weight="bold", y=0.98)

        for row_idx, p in enumerate(target_pairs):
            p_raw = p - 1
            fidx = list(man_df["pair_index"]).index(p)
            row_man = man_df.iloc[fidx]
            ts_str = row_man["end_time"]
            rd_id = row_man["rd_pair_id"]

            disp_rd = np.full((256, 256), np.nan, dtype=np.float32)
            disp_rd[val_all[p_raw]] = rd_all[p_raw][val_all[p_raw]]
            cur_grid = grid4_all[fidx]

            for col_idx, (m_label, df_m, m_own_all) in enumerate(methods):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("black")
                ax.imshow(disp_rd, cmap=cmap_rd, norm=norm_rd, origin="lower", extent=[0, 256, 0, 256])
                add_geometry_decorations(ax)

                draw_patch_boxes(ax, cur_grid, edgecolor="#ff1744", linewidth=0.8)

                p_df = df_m[df_m["pair_index"] == p]
                t_own = m_own_all[fidx]

                obs_df = p_df[p_df["state"] == "OBSERVED"]
                for _, r_state in obs_df.iterrows():
                    tid = int(r_state["track_id"])
                    t_mask = (t_own == tid)
                    if not t_mask.any():
                        continue
                    col = get_track_color(tid)
                    draw_patch_boxes(ax, t_mask, edgecolor=col, linewidth=1.8)

                    rcs = np.argwhere(t_mask)
                    cx = rcs[:, 1].mean() * 4 + 2
                    cy = rcs[:, 0].mean() * 4 + 2
                    ax.text(
                        cx, cy, f"T{tid} ({len(rcs)}p)",
                        color=col, fontsize=9.5, weight="bold",
                        ha="center", va="center",
                        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
                    )

                pred_df = p_df[p_df["state"] == "PREDICTED_ONLY"]
                for _, r_state in pred_df.iterrows():
                    tid = int(r_state["track_id"])
                    col = get_track_color(tid)
                    cx, cy, _, _ = get_last_observed_info(m_own_all, tid, fidx)
                    if not np.isnan(cx):
                        ax.scatter([cx], [cy], s=100, facecolors="none", edgecolors=col, linestyle="--", linewidth=1.8)
                        ax.scatter([cx], [cy], marker="x", s=50, color=col, linewidth=1.8)
                        ax.text(
                            cx + 4, cy - 4, f"T{tid}\n[PRED]",
                            color=col, fontsize=8.0, weight="bold",
                            path_effects=[pe.withStroke(linewidth=3, foreground="black")]
                        )

                if row_idx == 0:
                    ax.set_title(f"{m_label}\nPair {p:02d} ({rd_id}) — {ts_str[11:19]} UTC", color="white", fontsize=11, weight="bold")
                else:
                    ax.set_title(f"Pair {p:02d} ({rd_id}) — {ts_str[11:19]} UTC", color="white", fontsize=10)

                ax.set_xlim(0, 256)
                ax.set_ylim(0, 256)
                ax.set_xticks([])
                ax.set_yticks([])

        cbar_ax = fig.add_axes([0.25, 0.015, 0.50, 0.015])
        cbar = plt.colorbar(axes[0, 0].images[0], cax=cbar_ax, orientation="horizontal")
        cbar.set_label("RD (DN) [Fixed ±600 DN]", color="white", fontsize=11, weight="bold")
        cbar.ax.tick_params(colors="white", labelsize=9)

        sheet_path = summ_dir / out_name
        plt.savefig(sheet_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"  Saved Summary Sheet: {sheet_path.name}")

    print("\n" + "=" * 80)
    print("ALL FULL-DAY VISUAL PRODUCTS SUCCESSFULLY CREATED!")
    print(f"Location: {full_day_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

