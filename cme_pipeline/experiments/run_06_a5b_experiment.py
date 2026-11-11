"""Experiment 06 Runner: A5b PA-Width-Normalized CPA Ownership.

Performs controlled comparison between A4 Baseline and A5b on 2010-12-12 STEREO-A COR2 Beacon d7 data.
Generates all artifacts: tables, arrays, diagnostic figures, 32 individual frame PNGs, and comparison GIF.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v05_multi_component import MultiComponentAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v07_pa_width_weighted_cpa import PAWidthWeightedCPAAdaptiveROITrackingStage
from cme_pipeline.utils.io import save_yaml


# Canonical track colors
TRACK_COLORS = {
    1: "#1f77b4",  # Blue
    2: "#ff7f0e",  # Orange
    3: "#2ca02c",  # Green (Eastern CME)
    4: "#d62728",  # Red (Northeastern Streamer)
    5: "#9467bd",  # Purple (Northern Feature)
    6: "#8c564b",  # Brown (Predicted / Rogue)
    7: "#e377c2",  # Pink
    8: "#7f7f7f",  # Gray
    9: "#bcbd22",  # Olive
    10: "#17becf", # Cyan
}

def get_track_color(tid: int) -> str:
    if tid in TRACK_COLORS:
        return TRACK_COLORS[tid]
    cmap = plt.get_cmap("tab20")
    return cmap((tid * 3) % 20)


def main():
    print("=" * 75)
    print("STARTING EXPERIMENT 06: A5b PA-WIDTH-NORMALIZED CPA DISTANCE TRACKING")
    print("=" * 75)

    out_base = repo_root / "runs" / "06_pa_width_weighted_cpa_a5b"
    config_dir = out_base / "config"
    tables_dir = out_base / "tables"
    arrays_dir = out_base / "arrays"
    vis_dir = out_base / "visualizations"
    frames_dir = out_base / "frames"
    anim_dir = out_base / "animations"

    for d in [config_dir, tables_dir, arrays_dir, vis_dir, frames_dir, anim_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()
    ctx = PipelineContext(cfg)

    # 1. Run A4 Baseline
    print("\n--- Running Baseline A4 (Multi-Component Tracking) ---")
    a4_stage = MultiComponentAdaptiveROITrackingStage()
    a4_out = a4_stage.run(None, cfg, ctx)
    df_a4 = a4_out.track_states_df
    masks_a4 = getattr(a4_out, "ownership_masks")
    print(f"A4 completed: {len(df_a4)} state records across {masks_a4.shape[0]} frames.")

    # 2. Run A5b
    print("\n--- Running Variant A5b (PA-Width-Normalized CPA Distance Tracking) ---")
    cfg_a5b = PipelineConfig()
    cfg_a5b.tracking.method = "a5b"
    a5b_stage = PAWidthWeightedCPAAdaptiveROITrackingStage()
    a5b_out = a5b_stage.run(None, cfg_a5b, ctx)
    df_a5b = a5b_out.track_states_df
    masks_a5b = getattr(a5b_out, "ownership_masks")
    df_alloc = getattr(a5b_out, "contested_allocation_df")
    df_dec = getattr(a5b_out, "decisions_df")
    df_comp = getattr(a5b_out, "component_states_df")
    df_front = getattr(a5b_out, "leading_front_states_df")
    print(f"A5b completed: {len(df_a5b)} state records, {len(df_alloc)} contested allocation events.")

    # 3. Save Config & Numerical Artifacts
    print("\n--- Saving Configuration & Numerical Artifacts ---")
    save_yaml(cfg_a5b.to_dict(), config_dir / "config.yaml")
    df_a4.to_csv(tables_dir / "a4_track_states.csv", index=False)
    df_a5b.to_csv(tables_dir / "a5b_track_states.csv", index=False)
    df_alloc.to_csv(tables_dir / "a5b_contested_allocation_log.csv", index=False)
    df_dec.to_csv(tables_dir / "a5b_decision_log.csv", index=False)
    df_comp.to_csv(tables_dir / "a5b_component_states.csv", index=False)
    df_front.to_csv(tables_dir / "a5b_leading_front_states.csv", index=False)
    np.save(arrays_dir / "a5b_ownership_masks.npy", masks_a5b)

    # Comparison dataframe for Pairs 30-35
    p30_35_a4 = df_a4[df_a4["pair_index"].between(30, 35)].copy()
    p30_35_a4["method"] = "A4_Baseline"
    p30_35_a5b = df_a5b[df_a5b["pair_index"].between(30, 35)].copy()
    p30_35_a5b["method"] = "A5b_PA_Width_Weighted_CPA"
    df_comp_p30_35 = pd.concat([p30_35_a4, p30_35_a5b], ignore_index=True)
    df_comp_p30_35.to_csv(tables_dir / "a4_vs_a5b_pairs30_35_states.csv", index=False)
    print("Saved numerical artifacts successfully.")

    # 4. Load Raw RD Background Data
    print("\n--- Loading Raw RD Background Array ---")
    exp01_dir = repo_root / "validation_20101212/experiments/01_d7_baseline_segmentation"
    rd_path = exp01_dir / "results/science_arrays/north_up_RD_DNs.npy"
    assert rd_path.exists(), f"RD array missing at {rd_path}!"
    rd_all = np.load(rd_path, mmap_mode="r")
    print(f"Loaded RD background array: shape {rd_all.shape}, range [{np.nanmin(rd_all):.1f}, {np.nanmax(rd_all):.1f}] DN")

    man_df = a5b_out.manifest_df
    grid4_all = a5b_out.patch_activation_grid4
    geom = a5b_out.geometry
    p_size = 4
    n_patches = 64
    cmap_rd = plt.get_cmap("coolwarm").copy()
    cmap_rd.set_bad("black")
    vmin_rd, vmax_rd = -600.0, 600.0

    # 5. Generate Figure 1: Pairs 30-35 Side-by-Side Matrix Comparison
    print("\n--- Generating Figure 1: Pairs 30-35 Side-by-Side Comparison ---")
    pairs_pilot = [30, 31, 32, 33, 34, 35]
    fig, axes = plt.subplots(len(pairs_pilot), 2, figsize=(16, 5.2 * len(pairs_pilot)), dpi=140)

    for r_idx, pair_idx in enumerate(pairs_pilot):
        fidx = man_df[man_df["pair_index"] == pair_idx].index[0]
        t_str = str(man_df.loc[fidx, "end_time"])
        p_raw = pair_idx - 1
        rd_frame = rd_all[p_raw]
        active_g = grid4_all[fidx]

        for c_idx, (method_title, masks_data, df_data) in enumerate([
            ("A4 Baseline (Midpoint Voronoi)", masks_a4, df_a4),
            ("A5b (PA-Width-Normalized CPA)", masks_a5b, df_a5b),
        ]):
            ax = axes[r_idx, c_idx]
            ax.imshow(rd_frame, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")

            # Global active patches (thin red outline)
            for pr in range(n_patches):
                for pc in range(n_patches):
                    if active_g[pr, pc]:
                        rect = mpatches.Rectangle(
                            (pc * p_size - 0.5, pr * p_size - 0.5), p_size, p_size,
                            linewidth=0.7, edgecolor="red", facecolor="none", alpha=0.85
                        )
                        ax.add_patch(rect)

            # Track ownership overlays
            curr_mask = masks_data[fidx]
            for tid in np.unique(curr_mask):
                if tid == 0: continue
                col = get_track_color(tid)
                t_patch_mask = (curr_mask == tid)
                for pr in range(n_patches):
                    for pc in range(n_patches):
                        if t_patch_mask[pr, pc]:
                            rect = mpatches.Rectangle(
                                (pc * p_size - 0.5, pr * p_size - 0.5), p_size, p_size,
                                linewidth=1.2, edgecolor=col, facecolor=col, alpha=0.35
                            )
                            ax.add_patch(rect)

            # Frame tracks text
            frame_tracks = df_data[df_data["pair_index"] == pair_idx]
            lines = []
            for _, trk in frame_tracks.iterrows():
                tid = int(trk["track_id"])
                st = trk["state"]
                np_val = int(trk["num_patches"])
                cpa_val = float(trk["central_pa"])
                rf_val = float(trk["front_radius"])
                col = get_track_color(tid)
                lines.append(f"T{tid} ({st[:3]}): N={np_val}, CPA={cpa_val:.1f}°, R={rf_val:.1f}")

                if st == "OBSERVED":
                    theta = np.deg2rad(90.0 - cpa_val)
                    cx, cy = 128.0, 128.0
                    arrow_len = rf_val * 2.0
                    ax.plot([cx, cx + arrow_len * np.cos(theta)], [cy, cy + arrow_len * np.sin(theta)],
                            color=col, linestyle="--", linewidth=1.0, alpha=0.7)
                elif st == "PREDICTED_ONLY":
                    # Mark predicted track with dashed circle
                    theta = np.deg2rad(90.0 - cpa_val)
                    cx, cy = 128.0, 128.0
                    ax.scatter([cx + rf_val * 2.0 * np.cos(theta)], [cy + rf_val * 2.0 * np.sin(theta)],
                               marker="o", facecolors="none", edgecolors=col, linestyle="--", s=80, linewidth=1.5)

            ax.text(
                0.02, 0.98, f"Pair {pair_idx} | {t_str[11:19]}\n" + "\n".join(lines),
                transform=ax.transAxes, verticalalignment="top", fontsize=8.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75, edgecolor="white")
            )
            ax.set_title(f"{method_title} — Pair {pair_idx} ({t_str[11:19]} UTC)", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)
            ax.set_aspect("equal")

    plt.tight_layout()
    fig1_path = vis_dir / "fig_01_a4_vs_a5b_pairs30_35_comparison.png"
    plt.savefig(fig1_path, dpi=140)
    plt.close()
    print(f"Saved Figure 1 to {fig1_path}")

    # 6. Generate Figure 2: Pair 32 Flank & Streamer Diagnostic Zoom
    print("\n--- Generating Figure 2: Pair 32 Flank Diagnostic Zoom ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    f32_idx = man_df[man_df["pair_index"] == 32].index[0]
    rd_f32 = rd_all[31]

    for c_idx, (method_title, masks_data, df_data) in enumerate([
        ("A4 Baseline (Midpoint Voronoi)", masks_a4, df_a4),
        ("A5b (PA-Width-Normalized CPA Distance)", masks_a5b, df_a5b),
    ]):
        ax = axes[c_idx]
        ax.imshow(rd_f32, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")

        active_g = grid4_all[f32_idx]
        for pr in range(n_patches):
            for pc in range(n_patches):
                if active_g[pr, pc]:
                    rect = mpatches.Rectangle(
                        (pc * p_size - 0.5, pr * p_size - 0.5), p_size, p_size,
                        linewidth=0.8, edgecolor="red", facecolor="none", alpha=0.8
                    )
                    ax.add_patch(rect)

        curr_mask = masks_data[f32_idx]
        for tid in np.unique(curr_mask):
            if tid == 0: continue
            col = get_track_color(tid)
            t_patch_mask = (curr_mask == tid)
            for pr in range(n_patches):
                for pc in range(n_patches):
                    if t_patch_mask[pr, pc]:
                        rect = mpatches.Rectangle(
                            (pc * p_size - 0.5, pr * p_size - 0.5), p_size, p_size,
                            linewidth=1.3, edgecolor=col, facecolor=col, alpha=0.4
                        )
                        ax.add_patch(rect)

        p32_tracks = df_data[df_data["pair_index"] == 32]
        lines = [f"{method_title} — Pair 32 States:"]
        for _, trk in p32_tracks.iterrows():
            lines.append(f"  T{int(trk['track_id'])}: N={int(trk['num_patches'])}, CPA={trk['central_pa']:.1f}°, R={trk['front_radius']:.1f} px ({trk['state']})")
        ax.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, verticalalignment="top", fontsize=9.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8, edgecolor="yellow")
        )
        ax.set_title(f"Pair 32 — {method_title}", fontsize=13, fontweight="bold")
        ax.set_xlim(20, 240)
        ax.set_ylim(20, 240)
        ax.set_aspect("equal")

    plt.tight_layout()
    fig2_path = vis_dir / "fig_02_pair32_t3_t4_flank_diagnostic.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"Saved Figure 2 to {fig2_path}")

    # 7. Generate Figure 3: Focused T4 vs T5 Encroachment Diagnostic
    print("\n--- Generating Figure 3: Focused T4 vs T5 Encroachment Diagnostic ---")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=140)
    for idx_p, p_val in enumerate([31, 33, 35]):
        fidx = man_df[man_df["pair_index"] == p_val].index[0]
        rd_f = rd_all[p_val - 1]

        for r_m, (method_title, masks_data, df_data) in enumerate([
            ("A4 Baseline", masks_a4, df_a4),
            ("A5b Normalized", masks_a5b, df_a5b),
        ]):
            ax = axes[r_m, idx_p]
            ax.imshow(rd_f, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")

            curr_mask = masks_data[fidx]
            # Plot only T4 and T5
            for tid in [4, 5]:
                col = get_track_color(tid)
                t_patch_mask = (curr_mask == tid)
                for pr in range(n_patches):
                    for pc in range(n_patches):
                        if t_patch_mask[pr, pc]:
                            rect = mpatches.Rectangle(
                                (pc * p_size - 0.5, pr * p_size - 0.5), p_size, p_size,
                                linewidth=1.5, edgecolor=col, facecolor=col, alpha=0.5
                            )
                            ax.add_patch(rect)

            sub_trks = df_data[(df_data["pair_index"] == p_val) & (df_data["track_id"].isin([4, 5]))]
            t_lines = [f"Pair {p_val} ({method_title})"]
            for _, trk in sub_trks.iterrows():
                t_lines.append(f"  T{int(trk['track_id'])}: N={int(trk['num_patches'])}, CPA={trk['central_pa']:.1f}°")
            ax.text(
                0.03, 0.97, "\n".join(t_lines), transform=ax.transAxes, verticalalignment="top",
                fontsize=9.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.8, edgecolor="cyan")
            )
            ax.set_title(f"Pair {p_val} — {method_title}", fontsize=11, fontweight="bold")
            # Zoom on northern/northeastern quadrant
            ax.set_xlim(80, 240)
            ax.set_ylim(80, 240)
            ax.set_aspect("equal")

    plt.tight_layout()
    fig3_path = vis_dir / "fig_03_t4_t5_boundary_comparison.png"
    plt.savefig(fig3_path, dpi=140)
    plt.close()
    print(f"Saved Figure 3 to {fig3_path}")

    # 8. Generate Figure 4: Patch Count & CPA Trajectories
    print("\n--- Generating Figure 4: Patch Count & CPA Trajectories ---")
    fig, (ax_p, ax_cpa) = plt.subplots(2, 1, figsize=(12, 10), dpi=140, sharex=True)
    tracks_eval = [3, 4, 5]

    for tid in tracks_eval:
        col = get_track_color(tid)
        # A4
        sub_a4 = df_a4[(df_a4["track_id"] == tid) & (df_a4["pair_index"].between(30, 35))].sort_values("pair_index")
        ax_p.plot(sub_a4["pair_index"], sub_a4["num_patches"], color=col, linestyle="--", marker="o", alpha=0.8, label=f"T{tid} (A4 Baseline)")
        ax_cpa.plot(sub_a4["pair_index"], sub_a4["central_pa"], color=col, linestyle="--", marker="o", alpha=0.8, label=f"T{tid} (A4 Baseline)")

        # A5b
        sub_a5b = df_a5b[(df_a5b["track_id"] == tid) & (df_a5b["pair_index"].between(30, 35))].sort_values("pair_index")
        ax_p.plot(sub_a5b["pair_index"], sub_a5b["num_patches"], color=col, linestyle="-", linewidth=2.2, marker="s", label=f"T{tid} (A5b Normalized)")
        ax_cpa.plot(sub_a5b["pair_index"], sub_a5b["central_pa"], color=col, linestyle="-", linewidth=2.2, marker="s", label=f"T{tid} (A5b Normalized)")

    ax_p.set_ylabel("Owned Active Patches ($N$)", fontsize=12, fontweight="bold")
    ax_p.set_title("Evolution of Active Patch Ownership (Pairs 30–35): A4 vs A5b", fontsize=14, fontweight="bold")
    ax_p.grid(True, linestyle=":", alpha=0.6)
    ax_p.legend(loc="upper left", ncol=2, fontsize=9.5)

    ax_cpa.set_xlabel("Observation Pair Index", fontsize=12, fontweight="bold")
    ax_cpa.set_ylabel("Central Position Angle CPA (°)", fontsize=12, fontweight="bold")
    ax_cpa.set_title("CPA Trajectory Stability: T4 Streamer Invariance & T5 Confinement", fontsize=14, fontweight="bold")
    ax_cpa.grid(True, linestyle=":", alpha=0.6)
    ax_cpa.legend(loc="upper right", ncol=2, fontsize=9.5)

    plt.tight_layout()
    fig4_path = vis_dir / "fig_04_patch_and_cpa_trajectories.png"
    plt.savefig(fig4_path, dpi=140)
    plt.close()
    print(f"Saved Figure 4 to {fig4_path}")

    # 9. Generate Full-Day Frame PNGs (Pairs 29–60, 32 valid frames)
    print("\n--- Generating Full-Day Frame PNGs (Pairs 29–60) ---")
    frame_filenames = []
    for fidx, row in man_df.iterrows():
        p_idx = int(row["pair_index"])
        t_str = str(row["end_time"])
        p_raw = p_idx - 1
        rd_frame = rd_all[p_raw]
        active_g = grid4_all[fidx]

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8), dpi=130)

        for ax, method_title, masks_data, df_data in [
            (ax_left, "A4 Multi-Component (Midpoint Voronoi)", masks_a4, df_a4),
            (ax_right, "A5b Multi-Component (PA-Width-Normalized CPA)", masks_a5b, df_a5b),
        ]:
            ax.imshow(rd_frame, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")

            # Layer 1: Global active patches
            for pr in range(n_patches):
                for pc in range(n_patches):
                    if active_g[pr, pc]:
                        rect = mpatches.Rectangle(
                            (pc * p_size - 0.5, pr * p_size - 0.5), p_size, p_size,
                            linewidth=0.7, edgecolor="red", facecolor="none", alpha=0.85
                        )
                        ax.add_patch(rect)

            # Layer 2: Final owned patches
            curr_mask = masks_data[fidx]
            for tid in np.unique(curr_mask):
                if tid == 0: continue
                col = get_track_color(tid)
                t_patch_mask = (curr_mask == tid)
                for pr in range(n_patches):
                    for pc in range(n_patches):
                        if t_patch_mask[pr, pc]:
                            rect = mpatches.Rectangle(
                                (pc * p_size - 0.5, pr * p_size - 0.5), p_size, p_size,
                                linewidth=1.2, edgecolor=col, facecolor=col, alpha=0.35
                            )
                            ax.add_patch(rect)

            # Layer 3: Track annotations
            f_tracks = df_data[df_data["pair_index"] == p_idx]
            t_lines = [f"{method_title}", f"Pair {p_idx:02d} | {t_str[:19]} UTC"]
            for _, trk in f_tracks.iterrows():
                tid = int(trk["track_id"])
                st = trk["state"]
                np_val = int(trk["num_patches"])
                cpa_val = float(trk["central_pa"])
                rf_val = float(trk["front_radius"])
                col = get_track_color(tid)
                t_lines.append(f"  T{tid:2d} ({st[:3]}): N={np_val:3d}, CPA={cpa_val:5.1f}°, R={rf_val:5.1f}")

                if st == "OBSERVED":
                    theta = np.deg2rad(90.0 - cpa_val)
                    cx, cy = 128.0, 128.0
                    arrow_len = rf_val * 2.0
                    ax.plot([cx, cx + arrow_len * np.cos(theta)], [cy, cy + arrow_len * np.sin(theta)],
                            color=col, linestyle="--", linewidth=1.0, alpha=0.7)
                elif st == "PREDICTED_ONLY":
                    theta = np.deg2rad(90.0 - cpa_val)
                    cx, cy = 128.0, 128.0
                    ax.scatter([cx + rf_val * 2.0 * np.cos(theta)], [cy + rf_val * 2.0 * np.sin(theta)],
                               marker="o", facecolors="none", edgecolors=col, linestyle="--", s=80, linewidth=1.5)

            ax.text(
                0.02, 0.98, "\n".join(t_lines), transform=ax.transAxes, verticalalignment="top",
                fontsize=8.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75, edgecolor="white")
            )
            ax.set_title(f"Pair {p_idx:02d} — {method_title}", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)
            ax.set_aspect("equal")

        plt.tight_layout()
        f_path = frames_dir / f"pair_{p_idx:03d}.png"
        plt.savefig(f_path, dpi=130)
        plt.close()
        frame_filenames.append(f_path)
        if p_idx % 5 == 0 or p_idx == 60:
            print(f"  Rendered frame {p_idx:02d} / 60 -> {f_path.name}")

    print(f"Archived all {len(frame_filenames)} individual frame PNGs in {frames_dir}")

    # 10. Generate Full-Day Comparison GIF
    print("\n--- Generating Full-Day Animated GIF ---")
    gif_path = anim_dir / "full_day_a4_vs_a5b_comparison.gif"
    with imageio.get_writer(gif_path, mode="I", duration=0.45, loop=0) as writer:
        for f_path in frame_filenames:
            img = imageio.v3.imread(f_path)
            writer.append_data(img)
    print(f"Generated Full-Day GIF: {gif_path} (size: {gif_path.stat().st_size / (1024*1024):.2f} MB)")

    print("\n" + "=" * 75)
    print("EXPERIMENT 06 FULL EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"All deliverables saved under: {out_base}")
    print("=" * 75)


if __name__ == "__main__":
    main()

