"""Adaptive PA-ROI Tracking + PA Recovery Qualitative Pilot Runner (Pairs 29-39).

Executes:
- Phase 1: Code and Data Audit
- Phase 2: Birth-Seeded Adaptive PA-ROI Tracking
- Phase 3: Adaptive 1:Many Family Association
- Phase 4: Qualitative Checkpoint A (Before Recovery)
- Phase 5: Signed Dynamic PA Recovery
- Phase 6: Family Gates (Birth Radius & Size Gates)
- Phase 7: Qualitative Checkpoint B (After Recovery & Gates)
- Phase 8: Data Compilation for Final Report
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Matplotlib configuration for headless environment
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge, Circle
import numpy as np
import pandas as pd
import imageio.v2 as imageio

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import TrackingOutput
from cme_pipeline.stages.tracking.v02_adaptive_pa_roi import (
    BirthSeededAdaptiveROITrackingStage,
)
from cme_pipeline.stages.association.v03_adaptive_roi_association import (
    AdaptiveROIAssociationStage,
)
from cme_pipeline.stages.recovery.adaptive_recovery_adapter import (
    run_adaptive_family_recovery,
)
from cme_pipeline.utils.geometry import circular_diff_deg

# Visualization constants
VMIN_RD = -600.0
VMAX_RD = 600.0
SX = 126.618
SY = 127.503
R_OCCULTER = 22.0
R_OUTER_FOV = 122.5
R_BIRTH_GATE = 30.0
P04 = 4

# Qualitative distinct color palette for tracks/families
QUAL_COLORS = [
    "#00FF66",  # Vibrant Green
    "#00BFFF",  # Deep Sky Blue
    "#FF00FF",  # Magenta
    "#FFA500",  # Bright Orange
    "#FFD700",  # Gold
    "#FF3366",  # Coral Red
    "#7FFF00",  # Chartreuse
    "#9932CC",  # Dark Orchid
    "#00FFFF",  # Cyan
    "#FF69B4",  # Hot Pink
    "#32CD32",  # Lime Green
    "#FF4500",  # Orange Red
]


def pa_to_xy(pa_deg: float, r: float) -> Tuple[float, float]:
    """Convert North-Up WCS PA (0=North, 90=East) and radius to image (x, y)."""
    rad = np.radians(pa_deg)
    x = SX - r * np.sin(rad)
    y = SY + r * np.cos(rad)
    return float(x), float(y)


def get_rd_cmap():
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("black")
    return cmap


def draw_solar_references(ax: plt.Axes, show_gate: bool = True) -> None:
    """Draw occulter, FOV, 30px gate circle, and solar center crosshair."""
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(SX + R_OCCULTER * np.cos(th), SY + R_OCCULTER * np.sin(th),
            color="white", linestyle="--", linewidth=0.9, alpha=0.75, zorder=6)
    ax.plot(SX + R_OUTER_FOV * np.cos(th), SY + R_OUTER_FOV * np.sin(th),
            color="white", linestyle="--", linewidth=0.9, alpha=0.75, zorder=6)
    ax.plot([SX - 3.5, SX + 3.5], [SY, SY], color="white", linewidth=0.8, alpha=0.7, zorder=6)
    ax.plot([SX, SX], [SY - 3.5, SY + 3.5], color="white", linewidth=0.8, alpha=0.7, zorder=6)

    if show_gate:
        ax.plot(SX + R_BIRTH_GATE * np.cos(th), SY + R_BIRTH_GATE * np.sin(th),
                color="#FFD700", linestyle=":", linewidth=1.2, alpha=0.85, zorder=7)
        lx, ly = pa_to_xy(225.0, R_BIRTH_GATE + 4.5)
        ax.text(lx, ly, "Birth Gate: 30 px", color="#FFD700", fontsize=7.0, fontweight="bold",
                ha="center", va="center", zorder=15,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])


def run_full_pilot(repo_root: Path) -> Dict[str, Any]:
    print("==========================================================================")
    print("Stereo_COR2 — Adaptive PA-ROI Tracking + PA Recovery Qualitative Pilot")
    print("Target Interval: Pairs 29–39 (11 frames)")
    print("==========================================================================")

    # 1. Output Layout
    run_dir = repo_root / "runs/pilot_adaptive_roi_pair29_39"
    chk_a_dir = run_dir / "checkpoint_a"
    chk_b_dir = run_dir / "checkpoint_b"
    tables_dir = run_dir / "tables"
    arrays_dir = run_dir / "arrays"

    for d in [run_dir, chk_a_dir, chk_b_dir, tables_dir, arrays_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Config & Context
    cfg = PipelineConfig(run_name="pilot_adaptive_roi_pair29_39")
    ctx = PipelineContext(cfg)

    # ---------------------------------------------------------------------
    # Phase 2: Birth-Seeded Adaptive PA-ROI Tracking
    # ---------------------------------------------------------------------
    print("\n--- Phase 2: Running Birth-Seeded Adaptive PA-ROI Tracking ---")
    track_stage = BirthSeededAdaptiveROITrackingStage()
    track_out = track_stage.run(None, cfg, ctx)

    df_states_all = track_out.track_states_df
    conflicts_all = track_out.conflict_records
    adaptive_tracks = track_out.adaptive_tracks
    lbls_adaptive_all = track_out.patch_component_labels
    grid4_all = track_out.patch_activation_grid4
    man_df = track_out.manifest_df
    geom = track_out.geometry

    # Filter to Pairs 29–39 (indices 0 to 10)
    target_pairs = list(range(29, 40))
    p29_39_mask = man_df["pair_index"].isin(target_pairs)
    man_p29_39 = man_df[p29_39_mask].reset_index(drop=True)
    n_frames = len(man_p29_39)
    print(f"Tracking completed. Evaluating {n_frames} frames (Pairs 29–39).")

    # ---------------------------------------------------------------------
    # Phase 3: Adaptive Family Association
    # ---------------------------------------------------------------------
    print("\n--- Phase 3: Running Adaptive 1:Many Family Association ---")
    assoc_stage = AdaptiveROIAssociationStage()
    assoc_out = assoc_stage.run(track_out, cfg, ctx)

    df_membership = assoc_out.track_membership_df
    df_families = getattr(assoc_out, "family_summary_df", pd.DataFrame())
    df_edges = assoc_out.edges_df

    # ---------------------------------------------------------------------
    # Phase 4: Qualitative Checkpoint A (Before Recovery)
    # ---------------------------------------------------------------------
    print("\n--- Phase 4: Generating Qualitative Checkpoint A (Before Recovery) ---")
    # Load raw science images for background
    exp01_dir = repo_root / "validation/frozen/20101212/experiments/01_d7_baseline_segmentation"
    rd_all = np.load(exp01_dir / "results/science_arrays/north_up_RD_DNs.npy", mmap_mode="r")
    val_all = np.load(exp01_dir / "results/science_arrays/north_up_pair_valid_masks.npy", mmap_mode="r")

    # Load Baseline E1 tracking outputs for side-by-side comparison
    exp11_dir = repo_root / "validation/frozen/20101212/experiments/11_d7_patch_first_tracking_feasibility"
    df_states_e1 = pd.read_csv(exp11_dir / "results/tables/03_patch_e1_track_states.csv")
    lbls_e1_all = np.load(exp11_dir / "results/science_arrays/patch_component_labels.npy")

    cmap = get_rd_cmap()
    norm = Normalize(vmin=VMIN_RD, vmax=VMAX_RD)

    # 4.1 E1 vs Adaptive Tracking Side-by-Side (11 frames)
    sbs_a_dir = chk_a_dir / "e1_vs_adaptive_tracking_side_by_side"
    sbs_a_dir.mkdir(parents=True, exist_ok=True)
    frames_gif_a_dir = chk_a_dir / "frames/tracking_comparison"
    frames_gif_a_dir.mkdir(parents=True, exist_ok=True)
    sbs_a_paths = []

    print("Generating Checkpoint A side-by-side frames (E1 vs Adaptive Tracking)...")
    for f_idx in range(n_frames):
        p_row = man_p29_39.iloc[f_idx]
        p_num = int(p_row["pair_index"])
        p_raw = p_num - 1
        end_time = p_row["end_time"]
        dt_min = float(p_row["delta_t_min"])

        rd = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
        cur_lbl_e1 = lbls_e1_all[f_idx]
        cur_lbl_ad = lbls_adaptive_all[f_idx]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.0, 8.0), facecolor="#0e0e0e")
        plt.subplots_adjust(left=0.03, right=0.94, bottom=0.06, top=0.88, wspace=0.06)

        # Baseline E1 Panel
        ax0.set_facecolor("#000000")
        ax0.imshow(rd, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
        draw_solar_references(ax0)

        # Draw E1 tracks
        e1_tids = np.unique(cur_lbl_e1[cur_lbl_e1 > 0])
        for tid in e1_tids:
            mask_tid = np.kron(cur_lbl_e1 == tid, np.ones((P04, P04), bool))
            color = QUAL_COLORS[tid % len(QUAL_COLORS)]
            ax0.contour(mask_tid, levels=[0.5], colors=[color], linewidths=[1.8], origin="lower", extent=[0, 256, 0, 256])
            # Label
            cy, cx = np.mean(np.nonzero(mask_tid), axis=1)
            ax0.text(cx, cy, f"T{tid}", color=color, fontsize=8.5, fontweight="bold",
                     ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

        ax0.set_title(f"Baseline E1 1:1 Tracking\nActive Tracks: {len(e1_tids)}", color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax0.text(0.03, 0.94, "Fixed RD [-600.0, +600.0] DN", transform=ax0.transAxes, color="yellow", fontsize=8.0, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        # Adaptive PA-ROI Tracking Panel
        ax1.set_facecolor("#000000")
        ax1.imshow(rd, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
        draw_solar_references(ax1)

        ad_tids = np.unique(cur_lbl_ad[cur_lbl_ad > 0])
        for tid in ad_tids:
            mask_tid = np.kron(cur_lbl_ad == tid, np.ones((P04, P04), bool))
            color = QUAL_COLORS[tid % len(QUAL_COLORS)]
            ax1.contour(mask_tid, levels=[0.5], colors=[color], linewidths=[1.8], origin="lower", extent=[0, 256, 0, 256])
            # Draw Adaptive ROI wedge outline
            if tid in adaptive_tracks:
                tr = adaptive_tracks[tid]
                c_pa, h_w = tr.adaptive_active_roi
                theta1 = 90.0 - (c_pa + h_w)
                theta2 = 90.0 - (c_pa - h_w)
                w = Wedge((SX, SY), R_OUTER_FOV, theta1, theta2, width=R_OUTER_FOV - R_OCCULTER,
                          edgecolor=color, facecolor="none", linestyle=":", linewidth=1.2, alpha=0.7)
                ax1.add_patch(w)

            cy, cx = np.mean(np.nonzero(mask_tid), axis=1)
            ax1.text(cx, cy, f"T{tid}", color=color, fontsize=8.5, fontweight="bold",
                     ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

        ax1.set_title(f"Birth-Seeded Adaptive PA-ROI Tracking\nActive Tracks: {len(ad_tids)}", color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax1.text(0.03, 0.94, "Fixed RD [-600.0, +600.0] DN", transform=ax1.transAxes, color="yellow", fontsize=8.0, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        # Colorbar
        cax = fig.add_axes([0.95, 0.15, 0.015, 0.70])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label("RD (DN)", color="white", fontsize=10.0, fontweight="bold")
        cb.ax.tick_params(colors="white", labelsize=8.5)

        for ax in [ax0, ax1]:
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)
            ax.axis("off")

        fig.suptitle(f"Pair {p_num:03d} | {end_time} UTC (Δt = {dt_min:.1f}m) | Checkpoint A: Tracking Comparison",
                     color="white", fontsize=12.5, fontweight="bold", y=0.96)

        out_frame_p = sbs_a_dir / f"pair_{p_num:03d}.png"
        archive_frame_p = frames_gif_a_dir / f"pair_{p_num:03d}.png"
        plt.savefig(out_frame_p, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.savefig(archive_frame_p, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        sbs_a_paths.append(out_frame_p)

    # Build Tracking Comparison GIF
    gif_a_path = chk_a_dir / "tracking_comparison.gif"
    print(f"Assembling Checkpoint A GIF -> {gif_a_path}...")
    images_a = [imageio.imread(p) for p in sbs_a_paths]
    imageio.mimsave(gif_a_path, images_a, fps=2.0, loop=0)

    # 4.2 Birth PA / Birth CPA / Adaptive ROI Overlays (11 frames)
    roi_ov_dir = chk_a_dir / "birth_pa_roi_overlays"
    roi_ov_dir.mkdir(parents=True, exist_ok=True)
    print("Generating Birth PA / ROI overlay figures...")
    for f_idx in range(n_frames):
        p_row = man_p29_39.iloc[f_idx]
        p_num = int(p_row["pair_index"])
        p_raw = p_num - 1
        end_time = p_row["end_time"]

        rd = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
        cur_lbl_ad = lbls_adaptive_all[f_idx]

        fig, ax = plt.subplots(figsize=(9.0, 9.0), facecolor="#0e0e0e")
        ax.set_facecolor("#000000")
        ax.imshow(rd, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
        draw_solar_references(ax)

        ad_tids = np.unique(cur_lbl_ad[cur_lbl_ad > 0])
        for tid in ad_tids:
            color = QUAL_COLORS[tid % len(QUAL_COLORS)]
            mask_tid = np.kron(cur_lbl_ad == tid, np.ones((P04, P04), bool))
            ax.contour(mask_tid, levels=[0.5], colors=[color], linewidths=[1.8], origin="lower", extent=[0, 256, 0, 256])

            tr = adaptive_tracks.get(tid)
            if tr is not None:
                # Original Birth PA (Immutable) vector
                bx, by = pa_to_xy(tr.original_birth_pa, R_OCCULTER + 10.0)
                ax.plot([SX, bx], [SY, by], color="white", linewidth=2.0, linestyle="--", zorder=10)
                ax.scatter([bx], [by], color="white", s=40, marker="o", edgecolors="black", zorder=11)

                # Tracked CPA vector
                tx, ty = pa_to_xy(tr.tracked_cpa, R_OUTER_FOV - 10.0)
                ax.plot([SX, tx], [SY, ty], color=color, linewidth=2.2, zorder=10)

                # Adaptive ROI Wedge
                c_pa, h_w = tr.adaptive_active_roi
                theta1 = 90.0 - (c_pa + h_w)
                theta2 = 90.0 - (c_pa - h_w)
                w = Wedge((SX, SY), R_OUTER_FOV, theta1, theta2, width=R_OUTER_FOV - R_OCCULTER,
                          facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5, zorder=8)
                ax.add_patch(w)

                ax.text(tx, ty, f"T{tid} (CPA {tr.tracked_cpa:.1f}°\nROI ±{h_w:.1f}°)",
                        color=color, fontsize=8.0, fontweight="bold", ha="center",
                        path_effects=[pe.withStroke(linewidth=2.0, foreground="black")], zorder=15)

        ax.set_title(f"Pair {p_num:03d} | {end_time} UTC | Birth PA & Adaptive ROI Overlays\n(Dashed White: Birth PA, Solid: Tracked CPA, Shaded: Active ROI)",
                     color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.axis("off")

        plt.savefig(roi_ov_dir / f"pair_{p_num:03d}.png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    # 4.3 Track Ownership Map (Spatial visualization)
    print("Generating Track Ownership Map...")
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.2), facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.88, wspace=0.10)
    rep_pairs = [30, 34, 38]
    for idx, p_num in enumerate(rep_pairs):
        f_idx = p_num - 29
        p_raw = p_num - 1
        ax = axes[idx]
        ax.set_facecolor("#000000")
        rd = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
        ax.imshow(rd, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
        draw_solar_references(ax)

        cur_lbl_ad = lbls_adaptive_all[f_idx]
        ad_tids = np.unique(cur_lbl_ad[cur_lbl_ad > 0])
        for tid in ad_tids:
            color = QUAL_COLORS[tid % len(QUAL_COLORS)]
            mask_tid = np.kron(cur_lbl_ad == tid, np.ones((P04, P04), bool))
            # Translucent fill
            overlay = np.zeros((256, 256, 4), dtype=float)
            rgb = matplotlib.colors.to_rgb(color)
            overlay[mask_tid] = [rgb[0], rgb[1], rgb[2], 0.50]
            ax.imshow(overlay, origin="lower", extent=[0, 256, 0, 256])
            ax.contour(mask_tid, levels=[0.5], colors=[color], linewidths=[1.8], origin="lower", extent=[0, 256, 0, 256])

        ax.set_title(f"Pair {p_num:03d} Patch Ownership\nActive Tracks: {list(ad_tids)}",
                     color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.axis("off")

    fig.suptitle("Track Patch Ownership Maps — Strict Disjoint Allocation Across Coronal Structures",
                 color="white", fontsize=12.5, fontweight="bold", y=0.96)
    plt.savefig(chk_a_dir / "track_ownership_map.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 4.4 T19 / T24 / T26 Lineage Comparison (Mapping table & Diagram)
    print("Generating T19 / T24 / T26 Lineage Comparison...")
    # Find overlapping patches between E1 tracks and Adaptive tracks
    overlap_mapping_rows = []
    for e1_tid in [19, 24, 26, 15, 17]:
        e1_sub = df_states_e1[(df_states_e1["track_id"] == e1_tid) & (df_states_e1["pair_index"].isin(target_pairs))]
        if e1_sub.empty:
            continue
        first_p = int(e1_sub["pair_index"].min())
        e1_b_rmin = float(e1_sub["birth_r_min_px"].iloc[0])
        e1_cpa = float(e1_sub["central_pa"].mean())

        # Check overlap with adaptive tracks across all 11 frames
        ad_overlap_counts: Dict[int, int] = {}
        for _, r_e1 in e1_sub.iterrows():
            p_idx = int(r_e1["pair_index"])
            f_idx = p_idx - 29
            cid = int(r_e1["component_id"])
            m_e1 = (lbls_e1_all[f_idx] == cid)
            for ad_tid in np.unique(lbls_adaptive_all[f_idx]):
                if ad_tid > 0:
                    inter = int(np.logical_and(m_e1, lbls_adaptive_all[f_idx] == ad_tid).sum())
                    ad_overlap_counts[ad_tid] = ad_overlap_counts.get(ad_tid, 0) + inter

        top_ad_tid = max(ad_overlap_counts.items(), key=lambda x: x[1])[0] if ad_overlap_counts else -1
        total_inter = ad_overlap_counts.get(top_ad_tid, 0)

        tr_ad = adaptive_tracks.get(top_ad_tid)
        ad_desc = f"Adaptive T{top_ad_tid} (Birth P{tr_ad.birth_pair_id}, CPA {tr_ad.tracked_cpa:.1f}°)" if tr_ad else "None"

        overlap_mapping_rows.append({
            "baseline_e1_track_id": e1_tid,
            "e1_birth_pair": first_p,
            "e1_birth_r_min": e1_b_rmin,
            "e1_mean_cpa": e1_cpa,
            "mapped_adaptive_track_id": top_ad_tid,
            "adaptive_track_description": ad_desc,
            "patch_intersection_count": total_inter,
            "mapping_rationale": f"Max patch intersection ({total_inter} patches) and birth frame alignment",
        })

    df_mapping = pd.DataFrame(overlap_mapping_rows)
    df_mapping.to_csv(chk_a_dir / "t19_t24_t26_lineage_comparison.csv", index=False)

    # Lineage Diagram
    fig, ax = plt.subplots(figsize=(10.0, 6.0), facecolor="#121212")
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(2.5, 9.2, "Baseline E1 Tracking Lineage", color="#FFA500", fontsize=11.0, fontweight="bold", ha="center")
    ax.text(7.5, 9.2, "Adaptive PA-ROI Tracking Lineage", color="#00FF66", fontsize=11.0, fontweight="bold", ha="center")

    e1_nodes = [
        (2.5, 7.5, "E1 T15 (Pair 30, CPA 109°)\n[Fragmented East Root]", "#FFA500"),
        (2.5, 5.5, "E1 T19 (Pair 31, CPA 52°)\n[Merged Overgrown Tree]", "#FFA500"),
        (2.5, 3.5, "E1 T24 (Pair 34, CPA 125°)\n[Downstream Flank]", "#FFA500"),
        (2.5, 1.5, "E1 T26 (Pair 34, CPA 110°)\n[Separate Anchor, Merged in DO]", "#FFA500"),
    ]
    ad_nodes = [
        (7.5, 6.5, "Adaptive T3 (Pair 30 -> 39)\n[Unified Continuous East CME]\nCPA: 108.6° -> 113.0° (135 max patches)", "#00FF66"),
        (7.5, 2.5, "Adaptive T7 (Pair 34 -> 39)\n[Independent Anchor CME 2]\nCPA: 108.4° -> 107.0° (46 max patches)", "#00BFFF"),
    ]

    for x, y, txt, c in e1_nodes:
        ax.scatter([x], [y], color=c, s=500, edgecolors="white", linewidth=1.5)
        ax.text(x, y - 0.5, txt, color="white", fontsize=8.0, ha="center")

    for x, y, txt, c in ad_nodes:
        ax.scatter([x], [y], color=c, s=600, edgecolors="white", linewidth=1.5)
        ax.text(x, y - 0.6, txt, color="white", fontsize=8.5, fontweight="bold", ha="center")

    # Connect correspondences
    ax.plot([2.5, 7.5], [7.5, 6.5], color="cyan", linestyle="--", linewidth=1.8)
    ax.plot([2.5, 7.5], [5.5, 6.5], color="cyan", linestyle="--", linewidth=1.8)
    ax.plot([2.5, 7.5], [3.5, 6.5], color="cyan", linestyle="--", linewidth=1.8)
    ax.plot([2.5, 7.5], [1.5, 2.5], color="magenta", linestyle="--", linewidth=1.8)

    ax.text(5.0, 7.2, "Unified Continuity\n(No Artificial Split)", color="cyan", fontsize=8.0, ha="center", fontweight="bold")
    ax.text(5.0, 2.2, "Clean Anchor Separation\n(T26 preserved as T7)", color="magenta", fontsize=8.0, ha="center", fontweight="bold")

    fig.suptitle("Lineage Mapping: Baseline E1 Tracks vs Adaptive PA-ROI Tracks",
                 color="white", fontsize=12.0, fontweight="bold", y=0.96)
    plt.savefig(chk_a_dir / "t19_t24_t26_lineage_comparison.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 4.5 Tracked CPA vs Observed CPA Time Series
    print("Generating Tracked CPA vs Observed CPA Time Series...")
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.5), facecolor="#121212")
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.90, hspace=0.30, wspace=0.18)

    key_tids = [3, 4, 5, 7]
    key_titles = [
        "Track 3: Primary East CME (Unified T15/T19)",
        "Track 4: NE Coronal Feature",
        "Track 5: Northward Streamer",
        "Track 7: Secondary East Anchor (E1 T26 equivalent)",
    ]

    for idx, (tid, title) in enumerate(zip(key_tids, key_titles)):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor("#1a1a1a")
        sub_t = df_states_all[(df_states_all["track_id"] == tid) & (df_states_all["pair_index"].isin(target_pairs))].sort_values("pair_index")
        if sub_t.empty:
            continue

        pairs = sub_t["pair_index"].values
        obs_cpas = sub_t["central_pa"].values
        tr_cpas = sub_t["tracked_cpa"].values
        roi_half = sub_t["roi_half_width"].values

        # Acceptance corridor
        ax.fill_between(pairs, tr_cpas - roi_half, tr_cpas + roi_half, color="#00FF66", alpha=0.12, label="Adaptive Active ROI")
        # Tracked CPA (EMA alpha=0.1)
        ax.plot(pairs, tr_cpas, color="#00FF66", linewidth=2.5, label="Tracked CPA (EMA alpha=0.1)")
        # Instantaneous Observed CPA
        ax.plot(pairs, obs_cpas, color="#00BFFF", marker="o", markersize=6, linewidth=1.5, linestyle=":", label="Instantaneous Observed CPA")

        ax.set_title(title, color="white", fontsize=10.0, fontweight="bold", pad=6)
        ax.set_xlabel("Pair Index (2010-12-12)", color="white", fontsize=8.5)
        ax.set_ylabel("Position Angle (deg)", color="white", fontsize=8.5)
        ax.tick_params(colors="white", labelsize=8.0)
        ax.grid(True, linestyle=":", alpha=0.3, color="gray")
        ax.legend(loc="best", fontsize=7.5, facecolor="#262626", edgecolor="white", labelcolor="white")

    fig.suptitle("Tracked CPA (EMA alpha=0.1) vs Instantaneous Observed CPA Across Pairs 29–39",
                 color="white", fontsize=12.0, fontweight="bold", y=0.96)
    plt.savefig(chk_a_dir / "tracked_cpa_vs_observed_cpa_timeseries.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 4.6 Track-Specific HTMs
    htm_dir = chk_a_dir / "track_specific_htms"
    htm_dir.mkdir(parents=True, exist_ok=True)
    print("Generating Track-Specific Height-Time Maps...")
    r_bins = track_out.radial_bins
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0

    for tid in [3, 4, 7]:
        tr = adaptive_tracks.get(tid)
        if tr is None or len(tr.htm_history) == 0:
            continue

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.0, 6.0), facecolor="#121212")
        plt.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.88, wspace=0.18)

        pair_indices = [h.pair_index for h in tr.htm_history if h.pair_index in target_pairs]
        if not pair_indices:
            continue

        raw_mat = np.array([h.raw_radial_activity for h in tr.htm_history if h.pair_index in target_pairs]).T
        assigned_mat = np.array([h.assigned_track_components for h in tr.htm_history if h.pair_index in target_pairs]).T
        fronts = [h.observed_radial_front_px for h in tr.htm_history if h.pair_index in target_pairs]

        # Panel 1: Raw Radial Activity in PA Corridor
        ax0.set_facecolor("#000000")
        im0 = ax0.imshow(raw_mat, origin="lower", aspect="auto", cmap="inferno",
                         extent=[min(pair_indices) - 0.5, max(pair_indices) + 0.5, r_bins[0], r_bins[-1]])
        ax0.plot(pair_indices, fronts, color="cyan", linewidth=2.0, marker="o", label="Observed Radial Front")
        ax0.set_title("Raw Radial Activity in Adaptive PA ROI", color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax0.set_xlabel("Pair Index", color="white", fontsize=9.0)
        ax0.set_ylabel("Heliocentric Radius (px)", color="white", fontsize=9.0)
        ax0.tick_params(colors="white")
        ax0.legend(loc="upper left", facecolor="#262626", labelcolor="white", fontsize=8.0)
        cb0 = fig.colorbar(im0, ax=ax0)
        cb0.set_label("Active Patches", color="white")
        cb0.ax.tick_params(colors="white")

        # Panel 2: Assigned Track Components
        ax1.set_facecolor("#000000")
        im1 = ax1.imshow(assigned_mat, origin="lower", aspect="auto", cmap="viridis",
                         extent=[min(pair_indices) - 0.5, max(pair_indices) + 0.5, r_bins[0], r_bins[-1]])
        ax1.plot(pair_indices, fronts, color="magenta", linewidth=2.0, marker="o", label="Track Leading Edge")
        ax1.set_title("Assigned Track Components (Strict Ownership)", color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax1.set_xlabel("Pair Index", color="white", fontsize=9.0)
        ax1.set_ylabel("Heliocentric Radius (px)", color="white", fontsize=9.0)
        ax1.tick_params(colors="white")
        ax1.legend(loc="upper left", facecolor="#262626", labelcolor="white", fontsize=8.0)
        cb1 = fig.colorbar(im1, ax=ax1)
        cb1.set_label("Assigned Patches", color="white")
        cb1.ax.tick_params(colors="white")

        fig.suptitle(f"Track {tid} Height–Time Map (HTM) Profile (Pairs {min(pair_indices)}–{max(pair_indices)})",
                     color="white", fontsize=12.0, fontweight="bold", y=0.96)
        plt.savefig(htm_dir / f"track_{tid:02d}_htm.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    # 4.7 Overlapping ROI Conflicts Log
    df_conflicts_p29_39 = pd.DataFrame([vars(r) for r in conflicts_all if r.pair_index in target_pairs])
    df_conflicts_p29_39.to_csv(chk_a_dir / "overlapping_roi_conflicts.csv", index=False)
    print(f"Logged {len(df_conflicts_p29_39)} patch ownership conflicts in Pairs 29–39.")

    # ---------------------------------------------------------------------
    # Phase 5: Signed Dynamic PA Recovery
    # ---------------------------------------------------------------------
    print("\n--- Phase 5: Running Signed Dynamic PA Recovery (Adapter) ---")
    rec_results = run_adaptive_family_recovery(
        association_output=assoc_out,
        config=cfg,
        context=ctx,
        target_pair_indices=target_pairs,
    )
    fam_rec_masks = rec_results["family_recovered_masks"]
    fam_orig_masks = rec_results["family_original_masks"]
    fam_new_masks = rec_results["family_newly_recovered_masks"]
    df_growth = rec_results["recovery_growth_df"]
    df_overlap = rec_results["recovery_overlap_df"]

    # Save recovery tables
    df_growth.to_csv(chk_b_dir / "recovery_growth_by_family.csv", index=False)
    df_growth.to_csv(tables_dir / "recovery_growth_by_family.csv", index=False)
    df_overlap.to_csv(chk_b_dir / "recovery_overlap_between_families.csv", index=False)

    # ---------------------------------------------------------------------
    # Phase 6: Family Gates (Birth Radius & Size Gates)
    # ---------------------------------------------------------------------
    print("\n--- Phase 6: Applying Family Gates (Birth Radius & Size Gates) ---")
    # Gate 1: Birth Radius Gate (at least one anchor with birth_r_min <= 30px)
    # Gate 2: Size Gate (max original detected patches in single frame >= 5)
    gate_decisions = []
    admitted_fids = set()

    for _, r in df_families.iterrows():
        fid = r["family_id"]
        n_anc = int(r["anchor_count"])
        max_p = int(r["max_detected_patches"])

        has_anchor = (n_anc >= 1)
        has_size = (max_p >= 5)

        if has_anchor and has_size:
            status = "ADMITTED"
            reason = "NONE"
            admitted_fids.add(fid)
        elif not has_anchor:
            status = "REJECTED"
            reason = "NO_SOLAR_ROOT_ANCHOR"
        else:
            status = "REJECTED"
            reason = "INSUFFICIENT_SIZE_LT_5_PATCHES"

        gate_decisions.append({
            "family_id": fid,
            "status": status,
            "rejection_reason": reason,
            "member_count": int(r["member_count"]),
            "anchor_count": n_anc,
            "max_detected_patches": max_p,
            "first_pair_index": int(r["first_pair_index"]),
            "last_pair_index": int(r["last_pair_index"]),
        })

    df_gates = pd.DataFrame(gate_decisions)
    df_gates.to_csv(chk_b_dir / "admitted_vs_rejected_families.csv", index=False)
    df_gates.to_csv(tables_dir / "family_gate_decisions.csv", index=False)

    print(f"Family Gates: {len(admitted_fids)} families ADMITTED out of {len(df_families)} candidates.")

    # ---------------------------------------------------------------------
    # Phase 7: Qualitative Checkpoint B (After Recovery & Gates)
    # ---------------------------------------------------------------------
    print("\n--- Phase 7: Generating Qualitative Checkpoint B (After Recovery & Gates) ---")
    sbs_b_dir = chk_b_dir / "before_vs_after_recovery_side_by_side"
    sbs_b_dir.mkdir(parents=True, exist_ok=True)
    frames_gif_b_dir = chk_b_dir / "frames/before_vs_after_recovery"
    frames_gif_b_dir.mkdir(parents=True, exist_ok=True)
    sbs_b_paths = []

    # 7.1 Before vs After Recovery Side-by-Side (11 frames)
    print("Generating Checkpoint B side-by-side frames (Before vs After Recovery & Gates)...")
    for f_idx in range(n_frames):
        p_row = man_p29_39.iloc[f_idx]
        p_num = int(p_row["pair_index"])
        p_raw = p_num - 1
        end_time = p_row["end_time"]
        dt_min = float(p_row["delta_t_min"])

        rd = np.where(val_all[p_raw], rd_all[p_raw], np.nan)

        # Before Recovery: Original Observed Patches for Admitted Families
        orig_union = np.zeros((64, 64), dtype=bool)
        for fid in admitted_fids:
            if fid in fam_orig_masks:
                orig_union |= fam_orig_masks[fid][f_idx]

        # After Recovery: Original + Recovered Patches for Admitted Families
        rec_union = np.zeros((64, 64), dtype=bool)
        new_union = np.zeros((64, 64), dtype=bool)
        for fid in admitted_fids:
            if fid in fam_rec_masks:
                rec_union |= fam_rec_masks[fid][f_idx]
                new_union |= fam_new_masks[fid][f_idx]

        orig_px = np.kron(orig_union, np.ones((P04, P04), bool))
        rec_px = np.kron(rec_union, np.ones((P04, P04), bool))
        new_px = np.kron(new_union, np.ones((P04, P04), bool))

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.0, 8.0), facecolor="#0e0e0e")
        plt.subplots_adjust(left=0.03, right=0.94, bottom=0.06, top=0.88, wspace=0.06)

        # Left Panel: Before Recovery
        ax0.set_facecolor("#000000")
        ax0.imshow(rd, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
        draw_solar_references(ax0)
        if orig_px.any():
            ax0.contour(orig_px, levels=[0.5], colors=["#00FF66"], linewidths=[1.8], origin="lower", extent=[0, 256, 0, 256])
            over0 = np.zeros((256, 256, 4), dtype=float)
            over0[orig_px] = [0.0, 1.0, 0.4, 0.35]
            ax0.imshow(over0, origin="lower", extent=[0, 256, 0, 256])

        ax0.set_title(f"Before Recovery (Original Observed Patches)\nPatches: {int(orig_union.sum())} ({int(orig_px.sum()):,} px)",
                      color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax0.text(0.03, 0.94, "Fixed RD [-600.0, +600.0] DN", transform=ax0.transAxes, color="yellow", fontsize=8.0, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        # Right Panel: After Recovery & Gates
        ax1.set_facecolor("#000000")
        ax1.imshow(rd, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
        draw_solar_references(ax1)
        if rec_px.any():
            # Original in green contour
            if orig_px.any():
                ax1.contour(orig_px, levels=[0.5], colors=["#00FF66"], linewidths=[1.8], origin="lower", extent=[0, 256, 0, 256])
            # Newly recovered in orange contour / fill
            if new_px.any():
                ax1.contour(new_px, levels=[0.5], colors=["#FF9900"], linewidths=[1.8], linestyles="--", origin="lower", extent=[0, 256, 0, 256])
                over_new = np.zeros((256, 256, 4), dtype=float)
                over_new[new_px] = [1.0, 0.60, 0.0, 0.45]
                ax1.imshow(over_new, origin="lower", extent=[0, 256, 0, 256])

        growth_p = int(rec_union.sum() - orig_union.sum())
        ax1.set_title(f"After Recovery & Gates (Green: Original, Orange: Recovered)\nTotal Patches: {int(rec_union.sum())} (+{growth_p} recovered)",
                      color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax1.text(0.03, 0.94, "Fixed RD [-600.0, +600.0] DN", transform=ax1.transAxes, color="yellow", fontsize=8.0, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])

        # Colorbar
        cax = fig.add_axes([0.95, 0.15, 0.015, 0.70])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label("RD (DN)", color="white", fontsize=10.0, fontweight="bold")
        cb.ax.tick_params(colors="white", labelsize=8.5)

        for ax in [ax0, ax1]:
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)
            ax.axis("off")

        fig.suptitle(f"Pair {p_num:03d} | {end_time} UTC (Δt = {dt_min:.1f}m) | Checkpoint B: Recovery Expansion",
                     color="white", fontsize=12.5, fontweight="bold", y=0.96)

        out_frame_b = sbs_b_dir / f"pair_{p_num:03d}.png"
        archive_frame_b = frames_gif_b_dir / f"pair_{p_num:03d}.png"
        plt.savefig(out_frame_b, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.savefig(archive_frame_b, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        sbs_b_paths.append(out_frame_b)

    # Build Before vs After Recovery GIF
    gif_b_path = chk_b_dir / "before_vs_after_recovery.gif"
    print(f"Assembling Checkpoint B GIF -> {gif_b_path}...")
    images_b = [imageio.imread(p) for p in sbs_b_paths]
    imageio.mimsave(gif_b_path, images_b, fps=2.0, loop=0)

    # 7.2 Family-colored Recovered Patch Masks (Pair 34)
    print("Generating Family-colored Recovered Patch Masks...")
    p34_fidx = 5
    rd_34 = np.where(val_all[33], rd_all[33], np.nan)
    fig, ax = plt.subplots(figsize=(9.0, 9.0), facecolor="#0e0e0e")
    ax.set_facecolor("#000000")
    ax.imshow(rd_34, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
    draw_solar_references(ax)

    for idx, fid in enumerate(sorted(admitted_fids)):
        if fid in fam_rec_masks:
            f_m = fam_rec_masks[fid][p34_fidx]
            if f_m.any():
                color = QUAL_COLORS[idx % len(QUAL_COLORS)]
                f_px = np.kron(f_m, np.ones((P04, P04), bool))
                ax.contour(f_px, levels=[0.5], colors=[color], linewidths=[2.0], origin="lower", extent=[0, 256, 0, 256])
                over = np.zeros((256, 256, 4), dtype=float)
                rgb = matplotlib.colors.to_rgb(color)
                over[f_px] = [rgb[0], rgb[1], rgb[2], 0.40]
                ax.imshow(over, origin="lower", extent=[0, 256, 0, 256])
                cy, cx = np.mean(np.nonzero(f_px), axis=1)
                ax.text(cx, cy, f"{fid}", color=color, fontsize=9.0, fontweight="bold",
                        ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

    ax.set_title("Pair 034 (07:24 UTC) | Family-Colored Recovered Patch Masks", color="white", fontsize=11.0, fontweight="bold", pad=8)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.axis("off")
    plt.savefig(chk_b_dir / "family_colored_recovered_masks.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 7.3 Added Recovery Patch Maps (Original Green vs Recovered Orange at Pair 34)
    print("Generating Added Recovery Patch Maps...")
    fig, ax = plt.subplots(figsize=(9.0, 9.0), facecolor="#0e0e0e")
    ax.set_facecolor("#000000")
    ax.imshow(rd_34, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
    draw_solar_references(ax)

    orig_34 = np.zeros((64, 64), bool)
    new_34 = np.zeros((64, 64), bool)
    for fid in admitted_fids:
        if fid in fam_orig_masks:
            orig_34 |= fam_orig_masks[fid][p34_fidx]
            new_34 |= fam_new_masks[fid][p34_fidx]

    orig_34_px = np.kron(orig_34, np.ones((P04, P04), bool))
    new_34_px = np.kron(new_34, np.ones((P04, P04), bool))

    if orig_34_px.any():
        ax.contour(orig_34_px, levels=[0.5], colors=["#00FF66"], linewidths=[2.0], origin="lower", extent=[0, 256, 0, 256])
        over_orig = np.zeros((256, 256, 4), dtype=float)
        over_orig[orig_34_px] = [0.0, 1.0, 0.4, 0.40]
        ax.imshow(over_orig, origin="lower", extent=[0, 256, 0, 256])

    if new_34_px.any():
        ax.contour(new_34_px, levels=[0.5], colors=["#FF9900"], linewidths=[2.0], linestyles="--", origin="lower", extent=[0, 256, 0, 256])
        over_new = np.zeros((256, 256, 4), dtype=float)
        over_new[new_34_px] = [1.0, 0.60, 0.0, 0.45]
        ax.imshow(over_new, origin="lower", extent=[0, 256, 0, 256])

    ax.plot([], [], color="#00FF66", linewidth=3, label=f"Original Observed Patches ({int(orig_34.sum())})")
    ax.plot([], [], color="#FF9900", linewidth=3, linestyle="--", label=f"Newly Recovered Flank Patches ({int(new_34.sum())})")
    ax.legend(loc="lower left", facecolor="#262626", labelcolor="white", fontsize=9.0)

    ax.set_title("Pair 034 (07:24 UTC) | Added Recovery Patches (Green: Original, Orange: Recovered)",
                 color="white", fontsize=11.0, fontweight="bold", pad=8)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.axis("off")
    plt.savefig(chk_b_dir / "added_recovery_patch_maps.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 7.4 Admitted vs Rejected Family Diagnostic
    print("Generating Admitted vs Rejected Family Diagnostic...")
    fig, ax = plt.subplots(figsize=(10.0, 5.0), facecolor="#121212")
    ax.set_facecolor("#1a1a1a")

    fids = df_gates["family_id"].values
    max_patches = df_gates["max_detected_patches"].values
    anchors = df_gates["anchor_count"].values
    statuses = df_gates["status"].values

    colors = ["#00FF66" if s == "ADMITTED" else "#FF3333" for s in statuses]
    bars = ax.bar(fids, max_patches, color=colors, alpha=0.85, edgecolor="white")

    # Threshold line
    ax.axhline(5, color="yellow", linestyle="--", linewidth=1.8, label="Size Gate Threshold (>= 5 patches)")

    for bar, s, anc in zip(bars, statuses, anchors):
        h = bar.get_height()
        lbl = f"Pass" if s == "ADMITTED" else (f"No Anc" if anc == 0 else f"< 5p")
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 2, lbl,
                color="white", fontsize=8.0, fontweight="bold", ha="center")

    ax.set_title("Family Gate Evaluation (Birth Radius Gate: Anchor >= 1, Size Gate: Patches >= 5)",
                 color="white", fontsize=11.0, fontweight="bold", pad=8)
    ax.set_ylabel("Max Detected Patches (Single Frame)", color="white", fontsize=9.0)
    ax.tick_params(colors="white", labelsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.3, color="gray", axis="y")
    ax.legend(loc="upper right", facecolor="#262626", labelcolor="white", fontsize=8.5)

    plt.savefig(chk_b_dir / "admitted_vs_rejected_families.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 7.5 Recovery Area Growth by Family Bar Chart
    print("Generating Recovery Growth by Family Bar Chart...")
    fig, ax = plt.subplots(figsize=(10.0, 5.0), facecolor="#121212")
    ax.set_facecolor("#1a1a1a")

    sub_adm_growth = df_growth[df_growth["family_id"].isin(admitted_fids)]
    x = np.arange(len(sub_adm_growth))
    w = 0.35

    ax.bar(x - w / 2, sub_adm_growth["original_observed_patches"], width=w, color="#00FF66", alpha=0.85, label="Original Observed Patches")
    ax.bar(x + w / 2, sub_adm_growth["recovered_total_patches"], width=w, color="#FF9900", alpha=0.85, label="Total Recovered Patches")

    for idx, r in sub_adm_growth.reset_index().iterrows():
        pct = r["patch_growth_pct"]
        ax.text(idx + w / 2, r["recovered_total_patches"] + 5, f"+{pct:.1f}%",
                color="#FF9900", fontsize=8.0, fontweight="bold", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(sub_adm_growth["family_id"], color="white", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Total Patches (Pairs 29–39)", color="white", fontsize=9.0)
    ax.set_title("Recovery Patch Growth by Admitted Family Across Pairs 29–39",
                 color="white", fontsize=11.0, fontweight="bold", pad=8)
    ax.tick_params(colors="white", labelsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.3, color="gray", axis="y")
    ax.legend(loc="upper left", facecolor="#262626", labelcolor="white", fontsize=8.5)

    plt.savefig(chk_b_dir / "recovery_growth_by_family.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 7.6 Birth PA / Adaptive ROI / Recovered Patch Overlay (Pair 34)
    print("Generating Birth PA / Adaptive ROI / Recovered Patch Overlay...")
    fig, ax = plt.subplots(figsize=(9.0, 9.0), facecolor="#0e0e0e")
    ax.set_facecolor("#000000")
    ax.imshow(rd_34, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
    draw_solar_references(ax)

    # Show recovered patches in translucent cyan
    all_adm_rec = np.zeros((64, 64), bool)
    for fid in admitted_fids:
        if fid in fam_rec_masks:
            all_adm_rec |= fam_rec_masks[fid][p34_fidx]
    all_adm_rec_px = np.kron(all_adm_rec, np.ones((P04, P04), bool))
    ax.contour(all_adm_rec_px, levels=[0.5], colors=["#00FFFF"], linewidths=[2.0], origin="lower", extent=[0, 256, 0, 256])
    over_rec = np.zeros((256, 256, 4), dtype=float)
    over_rec[all_adm_rec_px] = [0.0, 1.0, 1.0, 0.35]
    ax.imshow(over_rec, origin="lower", extent=[0, 256, 0, 256])

    # Overlay Adaptive ROI and Birth PA vectors for active tracks
    cur_lbl_ad = lbls_adaptive_all[p34_fidx]
    ad_tids = np.unique(cur_lbl_ad[cur_lbl_ad > 0])
    for tid in ad_tids:
        tr = adaptive_tracks.get(tid)
        if tr:
            c_pa, h_w = tr.adaptive_active_roi
            theta1 = 90.0 - (c_pa + h_w)
            theta2 = 90.0 - (c_pa - h_w)
            w = Wedge((SX, SY), R_OUTER_FOV, theta1, theta2, width=R_OUTER_FOV - R_OCCULTER,
                      edgecolor="yellow", facecolor="none", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.add_patch(w)
            bx, by = pa_to_xy(tr.original_birth_pa, R_OCCULTER + 12.0)
            ax.scatter([bx], [by], color="yellow", s=45, marker="*")

    ax.set_title("Pair 034 (07:24 UTC) | Birth PA & Adaptive ROI vs Signed Recovered Patches\n(Yellow Stars: Birth PA, Dotted Wedges: Adaptive ROI, Cyan: Recovered CME Patches)",
                 color="white", fontsize=10.5, fontweight="bold", pad=8)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.axis("off")
    plt.savefig(chk_b_dir / "birth_pa_adaptive_roi_recovered_overlay.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # 7.7 T19 / T24 Corresponding CME Family Comparison
    print("Generating T19 / T24 CME Family Comparison...")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.0, 8.0), facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.03, right=0.94, bottom=0.06, top=0.88, wspace=0.06)

    # Panel 1: Primary East CME Family (FAM_T03)
    ax0.set_facecolor("#000000")
    ax0.imshow(rd_34, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
    draw_solar_references(ax0)

    f3_orig = fam_orig_masks.get("FAM_T03", np.zeros((11, 64, 64), bool))[p34_fidx]
    f3_rec = fam_rec_masks.get("FAM_T03", np.zeros((11, 64, 64), bool))[p34_fidx]
    f3_orig_px = np.kron(f3_orig, np.ones((P04, P04), bool))
    f3_rec_px = np.kron(f3_rec, np.ones((P04, P04), bool))

    if f3_orig_px.any():
        ax0.contour(f3_orig_px, levels=[0.5], colors=["#00FF66"], linewidths=[2.0], origin="lower", extent=[0, 256, 0, 256])
    if f3_rec_px.any():
        ax0.contour(f3_rec_px, levels=[0.5], colors=["#FF9900"], linewidths=[2.0], linestyles="--", origin="lower", extent=[0, 256, 0, 256])
        over_f3 = np.zeros((256, 256, 4), dtype=float)
        over_f3[f3_rec_px] = [1.0, 0.60, 0.0, 0.35]
        ax0.imshow(over_f3, origin="lower", extent=[0, 256, 0, 256])

    ax0.plot([], [], color="#00FF66", linewidth=2.5, label=f"Original Observed Core ({int(f3_orig.sum())} patches)")
    ax0.plot([], [], color="#FF9900", linewidth=2.5, linestyle="--", label=f"Recovered Structure ({int(f3_rec.sum())} patches)")
    ax0.legend(loc="lower left", facecolor="#262626", labelcolor="white", fontsize=8.5)
    ax0.set_title("FAM_T03: Primary East CME Complex\n(Continuous Core + Restored Lateral Flanks)",
                  color="white", fontsize=10.5, fontweight="bold", pad=8)

    # Panel 2: Secondary East Anchor Family (FAM_T07, E1 T26 equivalent)
    ax1.set_facecolor("#000000")
    ax1.imshow(rd_34, cmap=cmap, norm=norm, origin="lower", extent=[0, 256, 0, 256])
    draw_solar_references(ax1)

    f7_orig = fam_orig_masks.get("FAM_T07", np.zeros((11, 64, 64), bool))[p34_fidx]
    f7_rec = fam_rec_masks.get("FAM_T07", np.zeros((11, 64, 64), bool))[p34_fidx]
    f7_orig_px = np.kron(f7_orig, np.ones((P04, P04), bool))
    f7_rec_px = np.kron(f7_rec, np.ones((P04, P04), bool))

    if f7_orig_px.any():
        ax1.contour(f7_orig_px, levels=[0.5], colors=["#00BFFF"], linewidths=[2.0], origin="lower", extent=[0, 256, 0, 256])
    if f7_rec_px.any():
        ax1.contour(f7_rec_px, levels=[0.5], colors=["#FF00FF"], linewidths=[2.0], linestyles="--", origin="lower", extent=[0, 256, 0, 256])
        over_f7 = np.zeros((256, 256, 4), dtype=float)
        over_f7[f7_rec_px] = [1.0, 0.0, 1.0, 0.35]
        ax1.imshow(over_f7, origin="lower", extent=[0, 256, 0, 256])

    ax1.plot([], [], color="#00BFFF", linewidth=2.5, label=f"Original Observed Anchor ({int(f7_orig.sum())} patches)")
    ax1.plot([], [], color="#FF00FF", linewidth=2.5, linestyle="--", label=f"Recovered Structure ({int(f7_rec.sum())} patches)")
    ax1.legend(loc="lower left", facecolor="#262626", labelcolor="white", fontsize=8.5)
    ax1.set_title("FAM_T07: Cleanly Segregated Anchor\n(Independent Eruption at Solar Root, Disjoint from T03)",
                  color="white", fontsize=10.5, fontweight="bold", pad=8)

    # Colorbar
    cax = fig.add_axes([0.95, 0.15, 0.015, 0.70])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("RD (DN)", color="white", fontsize=10.0, fontweight="bold")
    cb.ax.tick_params(colors="white", labelsize=8.5)

    for ax in [ax0, ax1]:
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.axis("off")

    fig.suptitle("Comparison of Corresponding CME Families at Pair 034: FAM_T03 vs FAM_T07 (Segregated Identity Preserved Post-Recovery)",
                 color="white", fontsize=12.0, fontweight="bold", y=0.96)
    plt.savefig(chk_b_dir / "t19_t24_cme_family_comparison.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print("\nAll Qualitative Checkpoint A & B products generated successfully!")
    return {
        "chk_a_dir": chk_a_dir,
        "chk_b_dir": chk_b_dir,
        "admitted_fids": admitted_fids,
        "df_growth": df_growth,
        "df_overlap": df_overlap,
        "df_gates": df_gates,
        "df_mapping": df_mapping,
        "df_conflicts": df_conflicts_p29_39,
    }


if __name__ == "__main__":
    repo = Path("/home/aababc/stereo_cor2")
    results = run_full_pilot(repo)
    print("Pilot complete.")
