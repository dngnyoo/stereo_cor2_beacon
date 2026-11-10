"""Experiment 07 Phase 3: Track-Colored Visualizations, Overlap Highlights, and Animations.

Generates:
1. 32 High-Resolution PNG Frames with Track-Colored Overlays and White Overlap Highlights:
   frames/track_colored/pair_{p:03d}_{timestamp}.png
2. Full-Day Animated GIF (32 frames, 0.5s/frame):
   animations/track_colored_full_day.gif
3. Key Expansion Animated GIF (Pairs 30-39, 0.6s/frame):
   animations/track_colored_pairs30_39_detail.gif
4. Detailed Diagnostic Figures:
   figures/t3_t4_pairs31_37_multistage_comparison.png (Highlighting Pair 36 peak overlap)
   figures/t4_t5_boundary_overlap_detail.png (Highlighting 52.4% overlap on northern boundary)
   figures/t3_evolution_multistage.png (Major Eastern CME solo evolution)
   figures/t4_evolution_multistage.png (NE Streamer solo evolution)
   figures/t5_evolution_multistage.png (NNE Transient solo evolution)
   figures/track_overlap_distribution_and_matrix.png (Overlap matrix and temporal curves)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


TRACK_COLORS = {
    1: "#FF8800",   # Orange (West)
    2: "#00E5FF",   # Cyan (NNE early)
    3: "#00FF66",   # Bright Green (Major CME)
    4: "#FFD700",   # Gold (NE Streamer)
    5: "#FF007F",   # Magenta / Rose (NNE pulse)
    8: "#3399FF",   # Sky Blue (NNW transient)
    10: "#9933FF",  # Purple (ESE mid-day CME)
    12: "#FF5555",  # Coral Red (NE afternoon streamer)
    13: "#00CC99",  # Teal (East fragment)
    16: "#A6FF00",  # Lime (Late East CME)
    17: "#66E0FF",  # Light Cyan (Late NNW)
    18: "#EE82EE",  # Violet (Late SW outflow)
}

OVERLAP_COLOR = "#FFFFFF"  # Bright white for shared interaction zones


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def main():
    t_start = time.perf_counter()
    print("=" * 80)
    print("EXPERIMENT 07 PHASE 3: TRACK-COLORED VISUALIZATIONS & OVERLAP ANALYSIS")
    print("=" * 80)

    exp_dir = repo_root / "runs" / "07_a5b_end_to_end_segmentation"
    b2_dir = exp_dir / "experimental_a5b_identity"
    track_dir = exp_dir / "track_level_outputs"

    arrays_dir = track_dir / "arrays"
    tables_dir = track_dir / "tables"
    figs_dir = track_dir / "figures"
    frames_dir = track_dir / "frames" / "track_colored"
    anim_dir = track_dir / "animations"

    for d in [figs_dir, frames_dir, anim_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load Datasets & Masks
    # -------------------------------------------------------------------------
    print("\n>>> [1/5] Loading Track-Level 3D Tensors & Raw RD Arrays <<<")
    track_act_masks = np.load(arrays_dir / "track_active_patch_masks.npy")     # (20, 32, 64, 64)
    track_rec_masks = np.load(arrays_dir / "track_recovered_patch_masks.npy")  # (20, 32, 64, 64)
    track_pix_masks = np.load(arrays_dir / "track_final_pixel_masks.npy")      # (20, 32, 256, 256)

    manifest_df = pd.read_csv(tables_dir / "track_frame_mask_summary.csv")
    track_map_df = pd.read_csv(tables_dir / "track_index_map.csv")
    overlap_df = pd.read_csv(tables_dir / "track_overlap_statistics.csv")

    val_root = repo_root / "validation_20101212"
    exp01_dir = val_root / "experiments/01_d7_baseline_segmentation"
    rd_all = np.load(exp01_dir / "results/science_arrays/north_up_RD_DNs.npy", mmap_mode="r")
    val_all = np.load(exp01_dir / "results/science_arrays/north_up_pair_valid_masks.npy", mmap_mode="r")

    raw_manifest = pd.read_csv(b2_dir / "tables" / "01_frame_manifest.csv")
    n_frames = len(raw_manifest)

    cmap_rd = plt.get_cmap("coolwarm").copy()
    cmap_rd.set_bad("black")
    vmin_rd, vmax_rd = -600.0, 600.0

    admitted_track_ids = sorted(track_map_df[track_map_df["admitted_flag"]]["track_id"].tolist())

    # -------------------------------------------------------------------------
    # 2. Render 32 Individual Track-Colored PNG Frames with White Overlap
    # -------------------------------------------------------------------------
    print("\n>>> [2/5] Rendering 32 Individual Track-Colored PNG Frames <<<")
    png_paths = []

    for f_idx in range(n_frames):
        m_row = raw_manifest.iloc[f_idx]
        p_num = int(m_row["pair_index"])
        p_raw = p_num - 1
        ts_str = str(m_row["end_time"])
        dt_clean = ts_str.replace("T", " ").split(".")[0]
        ts_file = ts_str.replace(":", "").replace("-", "").replace("T", "_").split(".")[0]

        rd_raw = np.asarray(rd_all[p_raw], dtype=np.float32)
        val_mask = np.asarray(val_all[p_raw], dtype=bool)
        rd_display = np.where(val_mask, rd_raw, np.nan)

        # Build Frame Overlay RGBA Layer
        # Base count map for this frame
        frame_pix_stack = track_pix_masks[:, f_idx, :, :]  # (20, 256, 256)
        pix_counts = frame_pix_stack.sum(axis=0)           # (256, 256)
        overlap_mask = (pix_counts > 1)
        single_mask = (pix_counts == 1)

        f_total_pix = int((pix_counts >= 1).sum())
        f_overlap_pix = int(overlap_mask.sum())

        fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), dpi=140, facecolor="#0e0e0e")
        plt.subplots_adjust(left=0.04, right=0.92, top=0.88, bottom=0.06, wspace=0.12)

        # Panel 1: Full-FOV Track-Colored Final Pixel Segmentation
        ax1 = axes[0]
        ax1.set_facecolor("black")
        im1 = ax1.imshow(rd_display, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")

        # Overlay Active grid4 patches as subtle thin boxes
        act_stack = track_act_masks[:, f_idx, :, :]  # (20, 64, 64)
        tot_act_pat = int((act_stack.sum(axis=0) > 0).sum())
        act_y, act_x = np.nonzero(act_stack.sum(axis=0) > 0)
        for gy, gx in zip(act_y, act_x):
            rect = mpatches.Rectangle((gx * 4 - 0.5, gy * 4 - 0.5), 4, 4, fill=False, edgecolor="#FF3333", linewidth=0.5, alpha=0.55)
            ax1.add_patch(rect)

        # Overlay each admitted track's unique pixels
        legend_handles = []
        active_tids_in_frame = []

        for tid in admitted_track_ids:
            idx = tid - 1
            t_mask = frame_pix_stack[idx]
            t_pix_cnt = int(t_mask.sum())
            if t_pix_cnt == 0:
                continue

            active_tids_in_frame.append(tid)
            color_hex = TRACK_COLORS.get(tid, "#CCCCCC")
            t_only = t_mask & (~overlap_mask)

            # Draw track unique area
            if t_only.sum() > 0:
                m_disp = np.ma.masked_where(~t_only, t_only)
                ax1.imshow(m_disp, cmap=ListedColormap([color_hex]), alpha=0.55, origin="lower")

            # Draw contour boundary for the entire track mask
            ax1.contour(t_mask, levels=[0.5], colors=[color_hex], linewidths=1.2, origin="lower")

            # Add Centroid Label
            y_pts, x_pts = np.nonzero(t_mask)
            cx, cy = float(np.mean(x_pts)), float(np.mean(y_pts))
            ax1.text(
                cx, cy, f"T{tid}",
                color="white",
                fontsize=10,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color_hex, edgecolor="black", alpha=0.85, linewidth=0.8),
            )

            legend_handles.append(mpatches.Patch(color=color_hex, label=f"Track {tid} ({t_pix_cnt:,} px)"))

        # Overlay Overlapping (Shared) pixels in BRIGHT WHITE
        if f_overlap_pix > 0:
            m_ov_disp = np.ma.masked_where(~overlap_mask, overlap_mask)
            ax1.imshow(m_ov_disp, cmap=ListedColormap([OVERLAP_COLOR]), alpha=0.95, origin="lower")
            ax1.contour(overlap_mask, levels=[0.5], colors=["#FFFFFF"], linewidths=1.4, linestyles="dashed", origin="lower")
            legend_handles.append(mpatches.Patch(color=OVERLAP_COLOR, label=f"Shared Overlap ({f_overlap_pix:,} px)"))

        # Subtle occulter and FOV guideline
        circle_occ = plt.Circle((128, 128), 16.326, color="black", fill=True, zorder=4)
        ax1.add_patch(circle_occ)
        circle_occ_b = plt.Circle((128, 128), 16.326, color="#444444", fill=False, linewidth=1.0, linestyle=":", zorder=5)
        ax1.add_patch(circle_occ_b)

        ax1.set_title(
            f"[Panel 1] Track-Colored Final CME Segmentation (Pair {p_num:03d})\n"
            f"Total Area: {f_total_pix:,} px | Shared Overlap: {f_overlap_pix:,} px | Active Tracks: {len(active_tids_in_frame)}",
            color="white", fontsize=12, weight="bold"
        )
        ax1.set_xlabel("X (pixels)", color="white", fontsize=10)
        ax1.set_ylabel("Y (pixels)", color="white", fontsize=10)
        ax1.tick_params(colors="white")

        # Legend on Panel 1
        if legend_handles:
            ax1.legend(handles=legend_handles, loc="upper right", facecolor="#1a1a1a", edgecolor="#444444", fontsize=8, labelcolor="white")

        # Panel 2: Tracking Active Patch Ownership & Contested Flank Zoom
        ax2 = axes[1]
        ax2.set_facecolor("black")
        ax2.imshow(rd_display, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")

        # In Panel 2, show the Active Patch Ownership (mutually exclusive) vs Overlap Envelopes
        for tid in admitted_track_ids:
            idx = tid - 1
            act_m = act_stack[idx]
            color_hex = TRACK_COLORS.get(tid, "#CCCCCC")
            if act_m.sum() > 0:
                act_m_px = np.kron(act_m, np.ones((4, 4), bool)) & val_mask
                ax2.imshow(np.ma.masked_where(~act_m_px, act_m_px), cmap=ListedColormap([color_hex]), alpha=0.65, origin="lower")
                ax2.contour(act_m_px, levels=[0.5], colors=[color_hex], linewidths=1.0, origin="lower")

        # Overlay overlap region in Panel 2 to highlight where corridor expanded
        if f_overlap_pix > 0:
            ax2.contour(overlap_mask, levels=[0.5], colors=["#FFFFFF"], linewidths=1.5, linestyles="dashed", origin="lower")

        circle_occ2 = plt.Circle((128, 128), 16.326, color="black", fill=True, zorder=4)
        ax2.add_patch(circle_occ2)
        circle_occ2_b = plt.Circle((128, 128), 16.326, color="#444444", fill=False, linewidth=1.0, linestyle=":", zorder=5)
        ax2.add_patch(circle_occ2_b)

        ax2.set_title(
            f"[Panel 2] Tracking Active Patch Ownership (Mutually Exclusive)\n"
            f"Active Patches: {tot_act_pat} pat (Overlap=0) | White Dashed: Downstream Interaction Zone",
            color="#FFD700", fontsize=12, weight="bold"
        )
        ax2.set_xlabel("X (pixels)", color="white", fontsize=10)
        ax2.tick_params(colors="white")

        # Global Header
        fig.suptitle(
            f"STEREO-A COR2 Beacon d7 CME Track-Level Segmentation — Pair {p_num:03d} | {dt_clean} UTC\n"
            f"Distinct Track Colors with Explicit White Overlap Highlighting | Standard coolwarm [-600, +600] DN",
            color="white", fontsize=14, weight="bold", y=0.98,
        )

        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.70])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label("RD (DN)", color="white", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        frame_png_name = f"pair_{p_num:03d}_{ts_file}.png"
        frame_png_path = frames_dir / frame_png_name
        plt.savefig(frame_png_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        png_paths.append(frame_png_path)
        if (f_idx + 1) % 8 == 0 or f_idx == n_frames - 1:
            print(f"  Rendered {f_idx + 1:02d}/{n_frames:02d} track-colored frames -> {frame_png_name}")

    # -------------------------------------------------------------------------
    # 3. Compile Animated GIFs
    # -------------------------------------------------------------------------
    print("\n>>> [3/5] Compiling Animated GIFs <<<")
    # GIF 1: Full-Day 32 frames (0.5s/frame)
    gif_full_day = anim_dir / "track_colored_full_day.gif"
    with imageio.get_writer(gif_full_day, mode="I", duration=0.5, loop=0) as writer:
        for p in png_paths:
            writer.append_data(imageio.imread(p))
    print(f"  Full-Day GIF saved: {gif_full_day} ({gif_full_day.stat().st_size / (1024*1024):.2f} MB)")

    # GIF 2: Key Expansion Detail (Pairs 30-39, frames 1 to 10, 0.6s/frame)
    gif_detail = anim_dir / "track_colored_pairs30_39_detail.gif"
    with imageio.get_writer(gif_detail, mode="I", duration=0.6, loop=0) as writer:
        for p in png_paths[1:11]:  # Pairs 30 to 39
            writer.append_data(imageio.imread(p))
    print(f"  Expansion Detail GIF saved: {gif_detail} ({gif_detail.stat().st_size / (1024*1024):.2f} MB)")

    # -------------------------------------------------------------------------
    # 4. Generate Diagnostic Figures
    # -------------------------------------------------------------------------
    print("\n>>> [4/5] Generating Diagnostic Figures <<<")

    # Figure 1: Pairs 31-37 T3/T4 Multi-Stage Comparison (Highlighting Pair 36)
    print("  Generating: t3_t4_pairs31_37_multistage_comparison.png...")
    pairs_focus = [31, 32, 33, 34, 35, 36, 37]
    fig1, axes1 = plt.subplots(3, len(pairs_focus), figsize=(24, 12), dpi=130, facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.06, right=0.97, top=0.89, bottom=0.04, wspace=0.08, hspace=0.35)

    row_titles = [
        "Stage 1:\nActive Patches\n(Overlap = 0)",
        "Stage 2:\nRecovered Patches\n(Overlap in White)",
        "Stage 3:\nFinal Pixels\n(Overlap in White)",
    ]

    t3_idx = 2
    t4_idx = 3

    for col_idx, p_idx in enumerate(pairs_focus):
        f_idx = p_idx - 29
        p_raw = p_idx - 1
        rd_raw = np.asarray(rd_all[p_raw], dtype=np.float32)
        val_mask = np.asarray(val_all[p_raw], dtype=bool)
        rd_disp = np.where(val_mask, rd_raw, np.nan)

        # Stage 1: Active Patches
        ax_s1 = axes1[0, col_idx]
        ax_s1.set_facecolor("black")
        ax_s1.imshow(rd_disp, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")
        act3 = track_act_masks[t3_idx, f_idx]
        act4 = track_act_masks[t4_idx, f_idx]
        if act3.sum() > 0:
            act3_px = np.kron(act3, np.ones((4, 4), bool)) & val_mask
            ax_s1.imshow(np.ma.masked_where(~act3_px, act3_px), cmap=ListedColormap([TRACK_COLORS[3]]), alpha=0.7, origin="lower")
        if act4.sum() > 0:
            act4_px = np.kron(act4, np.ones((4, 4), bool)) & val_mask
            ax_s1.imshow(np.ma.masked_where(~act4_px, act4_px), cmap=ListedColormap([TRACK_COLORS[4]]), alpha=0.7, origin="lower")
        ax_s1.set_title(f"Pair {p_idx:03d} | Act: T3={act3.sum()}, T4={act4.sum()}", color="white", fontsize=10, weight="bold")
        ax_s1.tick_params(colors="#888888", labelsize=8)

        # Stage 2: Recovered Patches
        ax_s2 = axes1[1, col_idx]
        ax_s2.set_facecolor("black")
        ax_s2.imshow(rd_disp, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")
        rec3 = track_rec_masks[t3_idx, f_idx]
        rec4 = track_rec_masks[t4_idx, f_idx]
        rec_ov = rec3 & rec4
        rec3_only = rec3 & ~rec_ov
        rec4_only = rec4 & ~rec_ov
        if rec3_only.sum() > 0:
            r3_px = np.kron(rec3_only, np.ones((4, 4), bool)) & val_mask
            ax_s2.imshow(np.ma.masked_where(~r3_px, r3_px), cmap=ListedColormap([TRACK_COLORS[3]]), alpha=0.55, origin="lower")
        if rec4_only.sum() > 0:
            r4_px = np.kron(rec4_only, np.ones((4, 4), bool)) & val_mask
            ax_s2.imshow(np.ma.masked_where(~r4_px, r4_px), cmap=ListedColormap([TRACK_COLORS[4]]), alpha=0.55, origin="lower")
        if rec_ov.sum() > 0:
            rov_px = np.kron(rec_ov, np.ones((4, 4), bool)) & val_mask
            ax_s2.imshow(np.ma.masked_where(~rov_px, rov_px), cmap=ListedColormap([OVERLAP_COLOR]), alpha=0.9, origin="lower")
        ax_s2.set_title(f"Rec: T3={rec3.sum()}, T4={rec4.sum()} | Ov={rec_ov.sum()} pat", color="#FFD700" if rec_ov.sum() > 0 else "white", fontsize=10)
        ax_s2.tick_params(colors="#888888", labelsize=8)

        # Stage 3: Final Pixel Segmentation
        ax_s3 = axes1[2, col_idx]
        ax_s3.set_facecolor("black")
        ax_s3.imshow(rd_disp, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")
        pix3 = track_pix_masks[t3_idx, f_idx]
        pix4 = track_pix_masks[t4_idx, f_idx]
        pix_ov = pix3 & pix4
        pix3_only = pix3 & ~pix_ov
        pix4_only = pix4 & ~pix_ov
        if pix3_only.sum() > 0:
            ax_s3.imshow(np.ma.masked_where(~pix3_only, pix3_only), cmap=ListedColormap([TRACK_COLORS[3]]), alpha=0.55, origin="lower")
            ax_s3.contour(pix3, levels=[0.5], colors=[TRACK_COLORS[3]], linewidths=0.9, origin="lower")
        if pix4_only.sum() > 0:
            ax_s3.imshow(np.ma.masked_where(~pix4_only, pix4_only), cmap=ListedColormap([TRACK_COLORS[4]]), alpha=0.55, origin="lower")
            ax_s3.contour(pix4, levels=[0.5], colors=[TRACK_COLORS[4]], linewidths=0.9, origin="lower")
        if pix_ov.sum() > 0:
            ax_s3.imshow(np.ma.masked_where(~pix_ov, pix_ov), cmap=ListedColormap([OVERLAP_COLOR]), alpha=0.95, origin="lower")
            ax_s3.contour(pix_ov, levels=[0.5], colors=[OVERLAP_COLOR], linewidths=1.2, linestyles="dashed", origin="lower")

        # Highlight Pair 36 peak
        if p_idx == 36:
            for spine in ax_s3.spines.values():
                spine.set_color("#FF3333")
                spine.set_linewidth(2.5)
            ax_s3.set_title(f"★ PEAK OVERLAP ★\nOv: {pix_ov.sum():,} px ({pix_ov.sum()/pix3.sum()*100:.1f}% of T3)", color="#FF5555", fontsize=10, weight="bold")
        else:
            ax_s3.set_title(f"Pix: T3={pix3.sum():,}, T4={pix4.sum():,}\nOv={pix_ov.sum():,} px", color="white", fontsize=9)
        ax_s3.tick_params(colors="#888888", labelsize=8)

    # Label Row Titles
    for r_idx in range(3):
        axes1[r_idx, 0].set_ylabel(row_titles[r_idx], color="white", fontsize=11, weight="bold")

    fig1.suptitle(
        "T3 (Provisional: Eastern CME) vs T4 (Provisional: NE Streamer) Multi-Stage Evolution across Pairs 31–37\n"
        "Green: Track 3 | Gold: Track 4 | White: Shared Overlap Region | Pair 36 marks peak corridor interaction (1,287 px)",
        color="white", fontsize=15, weight="bold", y=0.98,
    )
    fig1_path = figs_dir / "t3_t4_pairs31_37_multistage_comparison.png"
    plt.savefig(fig1_path, dpi=130, facecolor=fig1.get_facecolor(), edgecolor="none")
    plt.close(fig1)
    print(f"  Saved: {fig1_path}")

    # Figure 2: T4/T5 Boundary Overlap Detail Figure
    print("  Generating: t4_t5_boundary_overlap_detail.png...")
    pairs_t4t5 = [32, 33, 34, 35, 36, 46]
    fig2, axes2 = plt.subplots(2, len(pairs_t4t5), figsize=(22, 9), dpi=130, facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.05, right=0.97, top=0.83, bottom=0.06, wspace=0.10, hspace=0.35)

    t5_idx = 4
    # Zoom window for northern streamer / pulse: X in [70, 200], Y in [120, 250]
    zx0, zx1, zy0, zy1 = 70, 200, 120, 250

    for col_idx, p_idx in enumerate(pairs_t4t5):
        f_idx = p_idx - 29
        p_raw = p_idx - 1
        rd_raw = np.asarray(rd_all[p_raw], dtype=np.float32)
        val_mask = np.asarray(val_all[p_raw], dtype=bool)
        rd_disp = np.where(val_mask, rd_raw, np.nan)

        pix4 = track_pix_masks[t4_idx, f_idx]
        pix5 = track_pix_masks[t5_idx, f_idx]
        ov45 = pix4 & pix5

        # Row 1: Full FOV view
        ax_top = axes2[0, col_idx]
        ax_top.set_facecolor("black")
        ax_top.imshow(rd_disp, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")
        if pix4.sum() > 0:
            ax_top.imshow(np.ma.masked_where(~pix4, pix4), cmap=ListedColormap([TRACK_COLORS[4]]), alpha=0.45, origin="lower")
        if pix5.sum() > 0:
            ax_top.imshow(np.ma.masked_where(~pix5, pix5), cmap=ListedColormap([TRACK_COLORS[5]]), alpha=0.65, origin="lower")
        if ov45.sum() > 0:
            ax_top.imshow(np.ma.masked_where(~ov45, ov45), cmap=ListedColormap([OVERLAP_COLOR]), alpha=0.95, origin="lower")
        # Draw zoom rectangle
        rect_z = mpatches.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, edgecolor="#00FFFF", linewidth=1.2, linestyle="--")
        ax_top.add_patch(rect_z)
        ax_top.set_title(f"Pair {p_idx:03d} (Full FOV)\nT4={pix4.sum():,}, T5={pix5.sum():,}", color="white", fontsize=10)
        ax_top.tick_params(colors="#888888", labelsize=8)

        # Row 2: Zoomed view of northern boundary
        ax_bot = axes2[1, col_idx]
        ax_bot.set_facecolor("black")
        ax_bot.imshow(rd_disp, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")
        if pix4.sum() > 0:
            ax_bot.imshow(np.ma.masked_where(~pix4, pix4), cmap=ListedColormap([TRACK_COLORS[4]]), alpha=0.55, origin="lower")
            ax_bot.contour(pix4, levels=[0.5], colors=[TRACK_COLORS[4]], linewidths=1.2, origin="lower")
        if pix5.sum() > 0:
            ax_bot.imshow(np.ma.masked_where(~pix5, pix5), cmap=ListedColormap([TRACK_COLORS[5]]), alpha=0.75, origin="lower")
            ax_bot.contour(pix5, levels=[0.5], colors=[TRACK_COLORS[5]], linewidths=1.2, origin="lower")
        if ov45.sum() > 0:
            ax_bot.imshow(np.ma.masked_where(~ov45, ov45), cmap=ListedColormap([OVERLAP_COLOR]), alpha=0.95, origin="lower")
            ax_bot.contour(ov45, levels=[0.5], colors=[OVERLAP_COLOR], linewidths=1.4, linestyles="dashed", origin="lower")

        ax_bot.set_xlim(zx0, zx1)
        ax_bot.set_ylim(zy0, zy1)
        t5_pct = (ov45.sum() / max(pix5.sum(), 1)) * 100.0
        ax_bot.set_title(f"Northern Fringe Zoom\nOverlap: {ov45.sum()} px ({t5_pct:.1f}% of T5)", color="#FF007F" if ov45.sum() > 0 else "white", fontsize=10, weight="bold")
        ax_bot.tick_params(colors="#888888", labelsize=8)

    axes2[0, 0].set_ylabel("Full Field of View", color="white", fontsize=11, weight="bold")
    axes2[1, 0].set_ylabel("Streamer Fringe Zoom", color="#00FFFF", fontsize=11, weight="bold")

    fig2.suptitle(
        "T4 (Provisional: NE Streamer, Gold) vs T5 (Provisional: NNE Transient, Magenta) Boundary Overlap Detail\n"
        "White: Shared Overlap | T5 shares 1,178 px (52.43% of total area) situated along the northern fringe of Track 4",
        color="white", fontsize=14, weight="bold", y=0.97,
    )
    fig2_path = figs_dir / "t4_t5_boundary_overlap_detail.png"
    plt.savefig(fig2_path, dpi=130, facecolor=fig2.get_facecolor(), edgecolor="none")
    plt.close(fig2)
    print(f"  Saved: {fig2_path}")

    # Figure 3, 4, 5: Individual Evolution Panels for Major Tracks (T3, T4, T5)
    for target_tid, t_name, t_color in [
        (3, "Provisional: Major Eastern CME", TRACK_COLORS[3]),
        (4, "Provisional: NE Coronal Streamer", TRACK_COLORS[4]),
        (5, "Provisional: NNE Narrow Transient", TRACK_COLORS[5]),
    ]:
        print(f"  Generating: t{target_tid}_evolution_multistage.png...")
        tid_idx = target_tid - 1
        rep_pairs = [30, 31, 32, 33, 34, 35, 36, 37]
        fig_t, axes_t = plt.subplots(3, len(rep_pairs), figsize=(24, 10), dpi=130, facecolor="#0e0e0e")
        plt.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.05, wspace=0.08, hspace=0.18)

        for col_idx, p_idx in enumerate(rep_pairs):
            f_idx = p_idx - 29
            p_raw = p_idx - 1
            rd_raw = np.asarray(rd_all[p_raw], dtype=np.float32)
            val_mask = np.asarray(val_all[p_raw], dtype=bool)
            rd_disp = np.where(val_mask, rd_raw, np.nan)

            act_m = track_act_masks[tid_idx, f_idx]
            rec_m = track_rec_masks[tid_idx, f_idx]
            pix_m = track_pix_masks[tid_idx, f_idx]

            # Row 1: Active Patches
            ax_a = axes_t[0, col_idx]
            ax_a.set_facecolor("black")
            ax_a.imshow(rd_disp, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")
            if act_m.sum() > 0:
                act_px = np.kron(act_m, np.ones((4, 4), bool)) & val_mask
                ax_a.imshow(np.ma.masked_where(~act_px, act_px), cmap=ListedColormap([t_color]), alpha=0.75, origin="lower")
            ax_a.set_title(f"Pair {p_idx:03d} | {act_m.sum()} pat", color="white", fontsize=10)
            ax_a.tick_params(colors="#888888", labelsize=8)

            # Row 2: Recovered Patches
            ax_r = axes_t[1, col_idx]
            ax_r.set_facecolor("black")
            ax_r.imshow(rd_disp, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")
            if rec_m.sum() > 0:
                rec_px = np.kron(rec_m, np.ones((4, 4), bool)) & val_mask
                ax_r.imshow(np.ma.masked_where(~rec_px, rec_px), cmap=ListedColormap([t_color]), alpha=0.60, origin="lower")
            ax_r.set_title(f"Rec: {rec_m.sum()} pat", color="#FFD700" if rec_m.sum() > 0 else "white", fontsize=10)
            ax_r.tick_params(colors="#888888", labelsize=8)

            # Row 3: Final Pixel Segmentation
            ax_p = axes_t[2, col_idx]
            ax_p.set_facecolor("black")
            ax_p.imshow(rd_disp, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")
            if pix_m.sum() > 0:
                ax_p.imshow(np.ma.masked_where(~pix_m, pix_m), cmap=ListedColormap([t_color]), alpha=0.60, origin="lower")
                ax_p.contour(pix_m, levels=[0.5], colors=[t_color], linewidths=1.2, origin="lower")
            ax_p.set_title(f"Pix: {pix_m.sum():,} px", color=t_color if pix_m.sum() > 0 else "white", fontsize=10, weight="bold")
            ax_p.tick_params(colors="#888888", labelsize=8)

        axes_t[0, 0].set_ylabel("Stage 1: Active Patches", color="white", fontsize=11, weight="bold")
        axes_t[1, 0].set_ylabel("Stage 2: Recovered Patches", color="white", fontsize=11, weight="bold")
        axes_t[2, 0].set_ylabel("Stage 3: Final Segmentation", color=t_color, fontsize=11, weight="bold")

        fig_t.suptitle(
            f"Track {target_tid} ({t_name}) Solo Multi-Stage Progression across Pairs 30–37\n"
            f"Complete Independent Evolution (Active Patches -> Dynamic Corridor Recovery -> Sub-pixel Island Segmentation)",
            color=t_color, fontsize=15, weight="bold", y=0.98,
        )
        fig_t_path = figs_dir / f"t{target_tid}_evolution_multistage.png"
        plt.savefig(fig_t_path, dpi=130, facecolor=fig_t.get_facecolor(), edgecolor="none")
        plt.close(fig_t)
        print(f"  Saved: {fig_t_path}")

    # Figure 6: Overlap Matrix & Framewise Overlap Curves
    print("  Generating: track_overlap_distribution_and_matrix.png...")
    fig6, axes6 = plt.subplots(1, 2, figsize=(19, 7.5), dpi=130, facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.05, right=0.96, top=0.86, bottom=0.10, wspace=0.28)

    # Subplot A: Overlap Matrix Heatmap
    matrix_df = pd.read_csv(tables_dir / "track_overlap_summary_matrix.csv", index_col=0)
    ax_mat = axes6[0]
    ax_mat.set_facecolor("black")
    mat_vals = matrix_df.values.astype(float)
    im_mat = ax_mat.imshow(mat_vals, cmap="magma", origin="upper")
    cbar_mat = fig6.colorbar(im_mat, ax=ax_mat, fraction=0.046, pad=0.04)
    cbar_mat.set_label("Overlapping Pixels", color="white", fontsize=10)
    cbar_mat.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar_mat.ax.axes, "yticklabels"), color="white")

    ticks = range(len(matrix_df.columns))
    ax_mat.set_xticks(ticks)
    ax_mat.set_yticks(ticks)
    ax_mat.set_xticklabels(matrix_df.columns, rotation=90, color="white", fontsize=8)
    ax_mat.set_yticklabels(matrix_df.index, color="white", fontsize=8)
    ax_mat.set_title("Pairwise Pixel Overlap Matrix (32-Frame Cumulative)", color="white", fontsize=12, weight="bold")

    # Annotate significant non-zero overlaps
    for y in range(len(ticks)):
        for x in range(len(ticks)):
            val = mat_vals[y, x]
            if val > 100:
                ax_mat.text(x, y, f"{int(val)}", color="white" if val < 2000 else "black", fontsize=7, ha="center", va="center", weight="bold")

    # Subplot B: Temporal Overlap Curves for Key Pairs
    ax_curve = axes6[1]
    ax_curve.set_facecolor("#181818")
    ax_curve.grid(True, color="#333333", linestyle="--", alpha=0.7)

    # Extract temporal curves from track_overlap_statistics.csv
    frame_ov = overlap_df[overlap_df["row_type"] == "FRAME_LEVEL"].copy()
    frame_ov["pair_num"] = frame_ov["pair_index"].astype(int)

    for tp, c_line, lbl in [
        ("T03_T04", "#00FF66", "T3 & T4 (ESE vs NE, 4,281 px total)"),
        ("T04_T05", "#FF007F", "T4 & T5 (NE vs NNE, 1,178 px total)"),
        ("T12_T13", "#FF5555", "T12 & T13 (NE vs East, 961 px)"),
        ("T13_T17", "#00CC99", "T13 & T17 (East vs NNW, 367 px)"),
        ("T05_T08", "#3399FF", "T5 & T8 (North-wrap NNE vs NNW, 275 px)"),
        ("T12_T16", "#A6FF00", "T12 & T16 (NE vs East, 223 px)"),
    ]:
        sub_tp = frame_ov[frame_ov["track_pair"] == tp].sort_values("pair_num")
        if not sub_tp.empty:
            ax_curve.plot(sub_tp["pair_num"], sub_tp["overlap_pixels"], marker="o", markersize=4, linewidth=1.8, color=c_line, label=lbl)

    # Also plot T3 & T5 as explicit 0 line
    ax_curve.axhline(0, color="#888888", linestyle=":", linewidth=1.0, label="T3 & T5 (Zero Overlap across all 32 frames)")

    ax_curve.set_xlabel("Observation Pair Index (Pairs 29–60)", color="white", fontsize=10)
    ax_curve.set_ylabel("Frame Overlapping Pixels", color="white", fontsize=10)
    ax_curve.set_title("Temporal Overlap Evolution Across Observation Frames", color="white", fontsize=12, weight="bold")
    ax_curve.tick_params(colors="white")
    ax_curve.legend(loc="upper right", facecolor="#111111", edgecolor="#444444", fontsize=8.5, labelcolor="white")

    fig6.suptitle(
        "Quantitative Distribution of Track Overlaps and Spatial Boundary Interactions\n"
        "Total Union Area: 59,899 px | Overlapping Pixels: 7,285 px (12.16%) | Non-overlapping: 52,614 px (87.84%)",
        color="white", fontsize=14, weight="bold", y=0.98,
    )
    fig6_path = figs_dir / "track_overlap_distribution_and_matrix.png"
    plt.savefig(fig6_path, dpi=130, facecolor=fig6.get_facecolor(), edgecolor="none")
    plt.close(fig6)
    print(f"  Saved: {fig6_path}")

    print(f"\n>>> Phase 3 Visualizations Completed in {time.perf_counter() - t_start:.2f} seconds <<<")


if __name__ == "__main__":
    main()
