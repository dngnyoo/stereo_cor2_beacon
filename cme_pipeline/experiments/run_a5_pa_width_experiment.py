"""Experiment Runner for A5: PA-Width-Proportional Ownership Diagnostic.

Compares A4 Multi-Component Baseline with A5 PA-Width-Proportional Allocation
on STEREO-A COR2 Beacon d7 data (focusing on Pairs 30-35 ownership dynamics and T6 regression).
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

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v05_multi_component import MultiComponentAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v06_pa_width_proportional import PAWidthProportionalAdaptiveROITrackingStage


# Canonical track colors for consistency
TRACK_COLORS = {
    1: "#1f77b4",  # Blue
    2: "#ff7f0e",  # Orange
    3: "#2ca02c",  # Green (Eastern CME)
    4: "#d62728",  # Red (Northeastern Streamer)
    5: "#9467bd",  # Purple (Northern Streamer/Feature)
    6: "#8c564b",  # Brown (Rogue / Predicted)
    7: "#e377c2",  # Pink
    8: "#7f7f7f",  # Gray
    9: "#bcbd22",  # Olive
    10: "#17becf", # Cyan
}

def get_track_color(tid: int) -> str:
    if tid in TRACK_COLORS:
        return TRACK_COLORS[tid]
    # Fallback to colormap
    cmap = plt.get_cmap("tab20")
    return cmap((tid * 3) % 20)


def main():
    print("=" * 70)
    print("STARTING EXPERIMENT A5: PA-Width-Proportional Ownership")
    print("=" * 70)

    out_base = repo_root / "runs" / "pilot_adaptive_pa_width_a5"
    tables_dir = out_base / "tables"
    arrays_dir = out_base / "arrays"
    vis_dir = out_base / "visualizations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()
    ctx = PipelineContext(cfg)

    # 1. Run A4 Baseline
    print("\n--- Running Baseline A4 (Multi-Component Tracking) ---")
    a4_stage = MultiComponentAdaptiveROITrackingStage()
    a4_out = a4_stage.run(None, cfg, ctx)
    df_a4 = a4_out.track_states_df
    masks_a4 = getattr(a4_out, "ownership_masks")
    print(f"A4 completed: {len(df_a4)} state records, {masks_a4.shape[0]} frames.")

    # 2. Run A5 Proportional Ownership
    print("\n--- Running Variant A5 (PA-Width-Proportional Tracking) ---")
    cfg_a5 = PipelineConfig()
    cfg_a5.tracking.method = "a5"
    a5_stage = PAWidthProportionalAdaptiveROITrackingStage()
    a5_out = a5_stage.run(None, cfg_a5, ctx)
    df_a5 = a5_out.track_states_df
    masks_a5 = getattr(a5_out, "ownership_masks")
    df_alloc = getattr(a5_out, "contested_allocation_df")
    df_dec = getattr(a5_out, "decisions_df")
    df_comp = getattr(a5_out, "component_states_df")
    print(f"A5 completed: {len(df_a5)} state records, {len(df_alloc)} contested allocations.")

    # 3. Save Tables & Arrays
    print("\n--- Saving Numerical Artifacts ---")
    df_a4.to_csv(tables_dir / "a4_track_states.csv", index=False)
    df_a5.to_csv(tables_dir / "a5_track_states.csv", index=False)
    df_alloc.to_csv(tables_dir / "a5_contested_allocation_log.csv", index=False)
    df_dec.to_csv(tables_dir / "a5_decision_log.csv", index=False)
    df_comp.to_csv(tables_dir / "a5_component_states.csv", index=False)
    np.save(arrays_dir / "a5_ownership_masks.npy", masks_a5)

    # Comparison dataframe for Pairs 30-35
    p30_35_a4 = df_a4[df_a4["pair_index"].between(30, 35)].copy()
    p30_35_a4["method"] = "A4_Baseline"
    p30_35_a5 = df_a5[df_a5["pair_index"].between(30, 35)].copy()
    p30_35_a5["method"] = "A5_PA_Width_Proportional"
    df_comp_p30_35 = pd.concat([p30_35_a4, p30_35_a5], ignore_index=True)
    df_comp_p30_35.to_csv(tables_dir / "a4_vs_a5_pairs30_35_states.csv", index=False)
    print("Saved tables and arrays successfully.")

    # 4. Load Raw RD Background Data
    print("\n--- Loading Raw RD Background ---")
    exp01_dir = repo_root / "validation_20101212/experiments/01_d7_baseline_segmentation"
    rd_path = exp01_dir / "results/science_arrays/north_up_RD_DNs.npy"
    assert rd_path.exists(), f"RD array missing at {rd_path}!"
    rd_all = np.load(rd_path, mmap_mode="r")
    print(f"Loaded RD background array: shape {rd_all.shape}, range [{np.nanmin(rd_all):.1f}, {np.nanmax(rd_all):.1f}] DN")

    # Geometry & Grid
    man_df = a5_out.manifest_df
    grid4_all = a5_out.patch_activation_grid4
    geom = a5_out.geometry
    p_size = 4
    n_patches = 64

    # 5. Generate Figure 1: Pairs 30-35 Side-by-Side Comparison (A4 vs A5)
    print("\n--- Generating Figure 1: Pairs 30-35 Side-by-Side Comparison ---")
    pairs_to_plot = [30, 31, 32, 33, 34, 35]
    n_rows = len(pairs_to_plot)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5.2 * n_rows), dpi=140)

    cmap_rd = plt.get_cmap("coolwarm").copy()
    cmap_rd.set_bad("black")
    vmin_rd, vmax_rd = -600.0, 600.0

    for r_idx, pair_idx in enumerate(pairs_to_plot):
        # Frame index in manifest
        fidx = man_df[man_df["pair_index"] == pair_idx].index[0]
        t_str = str(man_df.loc[fidx, "end_time"])
        p_raw = pair_idx - 1
        rd_frame = rd_all[p_raw]

        active_g = grid4_all[fidx]

        for c_idx, (method_name, masks_data, df_data) in enumerate([
            ("A4 Multi-Component (Voronoi)", masks_a4, df_a4),
            ("A5 PA-Width-Proportional (Quota)", masks_a5, df_a5),
        ]):
            ax = axes[r_idx, c_idx]
            im = ax.imshow(rd_frame, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")

            # Layer 1: Global active patches in thin red outline
            for pr in range(n_patches):
                for pc in range(n_patches):
                    if active_g[pr, pc]:
                        rect = mpatches.Rectangle(
                            (pc * p_size - 0.5, pr * p_size - 0.5),
                            p_size, p_size,
                            linewidth=0.7, edgecolor="red", facecolor="none", alpha=0.85
                        )
                        ax.add_patch(rect)

            # Layer 2: Track ownership overlays
            curr_mask = masks_data[fidx]
            owned_tids = np.unique(curr_mask)
            owned_tids = [t for t in owned_tids if t > 0]

            for tid in owned_tids:
                col = get_track_color(tid)
                t_patch_mask = (curr_mask == tid)
                for pr in range(n_patches):
                    for pc in range(n_patches):
                        if t_patch_mask[pr, pc]:
                            rect = mpatches.Rectangle(
                                (pc * p_size - 0.5, pr * p_size - 0.5),
                                p_size, p_size,
                                linewidth=1.2, edgecolor=col, facecolor=col, alpha=0.35
                            )
                            ax.add_patch(rect)

            # Layer 3: Annotate tracks in this frame
            frame_tracks = df_data[(df_data["pair_index"] == pair_idx)]
            text_lines = []
            for _, trk_row in frame_tracks.iterrows():
                tid = int(trk_row["track_id"])
                st = trk_row["state"]
                np_val = int(trk_row["num_patches"])
                cpa_val = float(trk_row["central_pa"])
                rf_val = float(trk_row["front_radius"])
                col = get_track_color(tid)
                text_lines.append(f"T{tid} ({st[:3]}): N={np_val}, CPA={cpa_val:.1f}°, R={rf_val:.1f}")

                # Mark CPA direction with ray / marker if observed
                if st == "OBSERVED":
                    rad_center = cpa_val
                    theta = np.deg2rad(90.0 - rad_center)  # Solar North-Up
                    # Solar center in pixel coordinates
                    cx, cy = 128.0, 128.0
                    arrow_len = rf_val * 2.0
                    ax.plot([cx, cx + arrow_len * np.cos(theta)], [cy, cy + arrow_len * np.sin(theta)],
                            color=col, linestyle="--", linewidth=1.0, alpha=0.7)

            info_box_str = "\n".join(text_lines)
            ax.text(
                0.02, 0.98, f"Pair {pair_idx} | {t_str[11:19]}\n{info_box_str}",
                transform=ax.transAxes, verticalalignment="top",
                fontsize=8.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75, edgecolor="white")
            )

            ax.set_title(f"{method_name} — Pair {pair_idx} ({t_str[11:19]} UTC)", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)
            ax.set_aspect("equal")

    plt.tight_layout()
    fig1_path = vis_dir / "fig_01_a4_vs_a5_pairs30_35_comparison.png"
    plt.savefig(fig1_path, dpi=140)
    plt.close()
    print(f"Saved Figure 1 to {fig1_path}")

    # 6. Generate Figure 2: Detailed Zoom on Pair 32 Contested Component Bisection
    print("\n--- Generating Figure 2: Pair 32 Bisection & Encroachment Diagnostic ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    f32_idx = man_df[man_df["pair_index"] == 32].index[0]
    rd_f32 = rd_all[31]

    for c_idx, (method_name, masks_data, df_data) in enumerate([
        ("A4 Baseline (Midpoint Voronoi)", masks_a4, df_a4),
        ("A5 Proportional (PA-Width Quota)", masks_a5, df_a5),
    ]):
        ax = axes[c_idx]
        ax.imshow(rd_f32, cmap=cmap_rd, vmin=vmin_rd, vmax=vmax_rd, origin="lower")

        active_g = grid4_all[f32_idx]
        for pr in range(n_patches):
            for pc in range(n_patches):
                if active_g[pr, pc]:
                    rect = mpatches.Rectangle(
                        (pc * p_size - 0.5, pr * p_size - 0.5),
                        p_size, p_size,
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
                            (pc * p_size - 0.5, pr * p_size - 0.5),
                            p_size, p_size,
                            linewidth=1.2, edgecolor=col, facecolor=col, alpha=0.45
                        )
                        ax.add_patch(rect)

        p32_tracks = df_data[df_data["pair_index"] == 32]
        lines = [f"{method_name} — Pair 32 Diagnostic:"]
        for _, trk in p32_tracks.iterrows():
            lines.append(f"  T{int(trk['track_id'])}: N={int(trk['num_patches'])}, CPA={trk['central_pa']:.1f}°, R={trk['front_radius']:.1f} px ({trk['state']})")
        ax.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, verticalalignment="top",
            fontsize=9.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8, edgecolor="yellow")
        )
        ax.set_title(f"Pair 32 — {method_name}", fontsize=13, fontweight="bold")
        ax.set_xlim(20, 240)
        ax.set_ylim(20, 240)
        ax.set_aspect("equal")

    plt.tight_layout()
    fig2_path = vis_dir / "fig_02_pair32_encroachment_diagnostic.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"Saved Figure 2 to {fig2_path}")

    # 7. Generate Figure 3: Quantitative Time Series of Patch Counts & CPA Evolution
    print("\n--- Generating Figure 3: Quantitative Time Series Evolution ---")
    fig, (ax_p, ax_cpa) = plt.subplots(2, 1, figsize=(12, 10), dpi=140, sharex=True)

    tracks_to_plot = [3, 4, 5, 6]
    for tid in tracks_to_plot:
        col = get_track_color(tid)
        # A4
        sub_a4 = df_a4[(df_a4["track_id"] == tid) & (df_a4["pair_index"].between(30, 35))].sort_values("pair_index")
        if len(sub_a4) > 0:
            ax_p.plot(sub_a4["pair_index"], sub_a4["num_patches"], color=col, linestyle="--", marker="o", label=f"T{tid} (A4 Baseline)")
            ax_cpa.plot(sub_a4["pair_index"], sub_a4["central_pa"], color=col, linestyle="--", marker="o", label=f"T{tid} (A4 Baseline)")

        # A5
        sub_a5 = df_a5[(df_a5["track_id"] == tid) & (df_a5["pair_index"].between(30, 35))].sort_values("pair_index")
        if len(sub_a5) > 0:
            ax_p.plot(sub_a5["pair_index"], sub_a5["num_patches"], color=col, linestyle="-", linewidth=2.2, marker="s", label=f"T{tid} (A5 Proportional)")
            ax_cpa.plot(sub_a5["pair_index"], sub_a5["central_pa"], color=col, linestyle="-", linewidth=2.2, marker="s", label=f"T{tid} (A5 Proportional)")

    ax_p.set_ylabel("Owned Active Patches ($N$)", fontsize=12, fontweight="bold")
    ax_p.set_title("Temporal Evolution of Track Patch Ownership (Pairs 30–35): A4 vs A5", fontsize=14, fontweight="bold")
    ax_p.grid(True, linestyle=":", alpha=0.6)
    ax_p.legend(loc="upper left", ncol=2, fontsize=9)

    ax_cpa.set_xlabel("Observation Pair Index", fontsize=12, fontweight="bold")
    ax_cpa.set_ylabel("Central Position Angle CPA (°)", fontsize=12, fontweight="bold")
    ax_cpa.set_title("Central Position Angle (CPA) Drift & Stability: A4 vs A5", fontsize=14, fontweight="bold")
    ax_cpa.grid(True, linestyle=":", alpha=0.6)
    ax_cpa.legend(loc="upper right", ncol=2, fontsize=9)

    plt.tight_layout()
    fig3_path = vis_dir / "fig_03_patch_and_cpa_evolution_pairs30_35.png"
    plt.savefig(fig3_path, dpi=140)
    plt.close()
    print(f"Saved Figure 3 to {fig3_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENT A5 PHASE 2 COMPUTATION & DIAGNOSTIC VISUALIZATION COMPLETE!")
    print(f"Outputs stored in: {out_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
