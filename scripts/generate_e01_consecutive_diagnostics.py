"""Stereo_COR2 Seed Research E01 — Consecutive Frame Diagnostics & Synchronization.

Generates 12-panel comprehensive diagnostic figures for Pairs 30 through 50 (inclusive)
comparing A5b, S1, S2, SPT-0, SPT-1, and SPT-2 under identical inputs.
Outputs animated GIFs for each method and a synchronized comparison animation.
"""
from __future__ import annotations

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from PIL import Image

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.stages.tracking.registry import get_tracking_stage
from cme_pipeline.stages.tracking.seed_component_pilot import (
    identify_and_rank_seed_candidates,
    determine_connectivity_and_grow,
    extract_continuous_scores_from_detection,
)
from cme_pipeline.stages.tracking.spt_downstream_adapter import run_spt_downstream_pipeline


# Discrete distinct palette for Track IDs 1 to 20
TRACK_COLORS = [
    "#000000",  # 0: background
    "#1f77b4",  # 1: blue
    "#ff7f0e",  # 2: orange
    "#2ca02c",  # 3: green
    "#d62728",  # 4: red
    "#9467bd",  # 5: purple
    "#8c564b",  # 6: brown
    "#e377c2",  # 7: pink
    "#7f7f7f",  # 8: gray
    "#bcbd22",  # 9: olive
    "#17becf",  # 10: cyan
    "#aec7e8",  # 11: light blue
    "#ffbb78",  # 12: light orange
    "#98df8a",  # 13: light green
    "#ff9896",  # 14: light red
    "#c5b0d5",  # 15: light purple
    "#c49c94",  # 16: light brown
    "#f7b6d2",  # 17: light pink
    "#c7c7c7",  # 18: light gray
    "#dbdb8d",  # 19: light olive
    "#9edae5",  # 20: light cyan
]
TRACK_CMAP = ListedColormap(TRACK_COLORS)


def get_color_for_track(track_id: int) -> str:
    """Returns stable hex color for a given track ID."""
    if track_id <= 0:
        return "#000000"
    idx = 1 + ((track_id - 1) % (len(TRACK_COLORS) - 1))
    return TRACK_COLORS[idx]


def build_track_overlay(
    track_mask: np.ndarray,
    max_id: int = 20,
) -> np.ndarray:
    """Maps track mask (64x64) to RGBA image (256x256) for crisp display."""
    rgba = np.zeros((64, 64, 4), dtype=np.float32)
    for tid in range(1, max_id + 1):
        pts = (track_mask == tid)
        if pts.any():
            hex_c = get_color_for_track(tid)
            r = int(hex_c[1:3], 16) / 255.0
            g = int(hex_c[3:5], 16) / 255.0
            b = int(hex_c[5:7], 16) / 255.0
            rgba[pts] = [r, g, b, 0.85]
    return rgba


def generate_single_frame_panel(
    pair_idx: int,
    det: Any,
    scores_64: np.ndarray,
    tracking_outputs: Dict[str, Any],
    spt2_downstream: Any,
    output_png: Path,
) -> None:
    """Renders a 12-panel comprehensive diagnostic figure for one pair."""
    f_idx_matches = det.manifest_df[det.manifest_df["pair_index"] == pair_idx].index
    if len(f_idx_matches) == 0:
        print(f"Warning: Pair {pair_idx} not found in manifest.")
        return
    f_idx = int(f_idx_matches[0])
    row = det.manifest_df.iloc[f_idx]
    ts = str(row["end_time"])[:19]
    is_usable = bool(row.get("usable_detection", True))
    status = str(row.get("observation_status", "UNKNOWN"))

    geom = det.geometry
    sx, sy = float(geom.sx), float(geom.sy)
    fc = det.frame_cache[f_idx]
    rd = np.asarray(fc.get("rd", np.zeros((256, 256))), dtype=float)
    usable_mask = np.asarray(fc.get("usable", np.ones((256, 256), dtype=bool)), dtype=bool)

    # D4 continuous score & active patches
    cur_score = scores_64[f_idx]
    active_grid = det.patch_activation_grid4[f_idx]
    n_active = int(np.sum(active_grid))

    # Inner corona boundaries
    r_min_px = 22.0
    r_seed_px = 30.0

    # Seed candidates
    candidates = identify_and_rank_seed_candidates(
        frame_scores=cur_score,
        active_mask=active_grid,
        geometry=geom,
        r_min=r_min_px,
        r_seed=r_seed_px,
        pair_index=pair_idx,
        frame_idx=f_idx,
        timestamp=ts,
    )
    rep_seeds = [c for c in candidates if c.score_rank == 1]

    # Create 3x4 figure
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), facecolor="black")
    plt.subplots_adjust(wspace=0.15, hspace=0.22, left=0.04, right=0.96, top=0.94, bottom=0.04)

    fig.suptitle(
        f"Stereo_COR2 Seed Research E01 — Pair {pair_idx:03d} | Timestamp: {ts} UTC | Status: {status} | Active Patches: {n_active}",
        color="white", fontsize=18, fontweight="bold", y=0.98
    )

    def style_ax(ax: plt.Axes, title: str):
        ax.set_facecolor("black")
        ax.set_title(title, color="cyan", fontsize=13, fontweight="bold", pad=6)
        ax.set_xlim(-0.5, 255.5)
        ax.set_ylim(255.5, -0.5)  # North-up image coords
        ax.tick_params(colors="gray", labelsize=8)
        # Solar center and occulter circles
        occ = Circle((sx, sy), 16.326, edgecolor="white", facecolor="none", linestyle=":", linewidth=1.0)
        ann_min = Circle((sx, sy), r_min_px, edgecolor="red", facecolor="none", linestyle="--", linewidth=1.0, alpha=0.7)
        ann_max = Circle((sx, sy), r_seed_px, edgecolor="yellow", facecolor="none", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.add_patch(occ)
        ax.add_patch(ann_min)
        ax.add_patch(ann_max)

    # -------------------------------------------------------------
    # Panel 1: Original North-Up Running-Difference (RD)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    style_ax(ax1, "1. North-Up Running-Difference (RD)")
    if is_usable:
        rd_disp = np.clip(rd, -300.0, 300.0)
        im1 = ax1.imshow(rd_disp, cmap="gray", origin="upper", vmin=-150, vmax=150)
    else:
        ax1.text(128, 128, f"OBSERVATION INVALID\n({status})", color="red", ha="center", va="center", fontsize=14, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 2: Continuous D4 Score Heatmap (64x64)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    style_ax(ax2, "2. Continuous D4 Detection Score")
    if is_usable and not np.isnan(cur_score).all():
        # Upsample 64x64 score to 256x256 for alignment
        score_up = np.repeat(np.repeat(np.nan_to_num(cur_score, nan=0.0), 4, axis=0), 4, axis=1)
        ax2.imshow(score_up, cmap="inferno", origin="upper", vmin=0.0, vmax=12.0)
    else:
        ax2.text(128, 128, "No Continuous Scores", color="gray", ha="center", va="center")

    # -------------------------------------------------------------
    # Panel 3: T24-Active Patch Activation Map
    # -------------------------------------------------------------
    ax3 = axes[0, 2]
    style_ax(ax3, f"3. T24 Active Patches ({n_active} Active)")
    if is_usable and n_active > 0:
        active_up = np.repeat(np.repeat(active_grid.astype(float), 4, axis=0), 4, axis=1)
        ax3.imshow(active_up, cmap="magma", origin="upper", vmin=0, vmax=1.5)
    else:
        ax3.text(128, 128, "No Active Patches", color="gray", ha="center", va="center")

    # -------------------------------------------------------------
    # Panel 4: Ranked Seed Candidates & Representative Seeds
    # -------------------------------------------------------------
    ax4 = axes[0, 3]
    style_ax(ax4, f"4. Seed Candidates ({len(candidates)} Cands, {len(rep_seeds)} Seeds)")
    if is_usable:
        # Show faint RD background
        ax4.imshow(np.clip(rd, -150, 150), cmap="gray", origin="upper", alpha=0.4)
        for cand in candidates:
            px = cand.patch_c * 4 + 2
            py = cand.patch_r * 4 + 2
            if cand.score_rank == 1:
                ax4.plot(px, py, marker="*", markersize=14, color="gold", markeredgecolor="black")
                ax4.text(px + 4, py - 4, f"R1:{cand.raw_score:.1f}", color="yellow", fontsize=9, fontweight="bold")
            else:
                ax4.plot(px, py, marker="o", markersize=5, color="cyan", markeredgecolor="blue")
                if cand.score_rank <= 3:
                    ax4.text(px + 3, py + 3, f"R{cand.score_rank}", color="cyan", fontsize=7)

    # -------------------------------------------------------------
    # Panel 5: A5b Track Ownership
    # -------------------------------------------------------------
    ax5 = axes[1, 0]
    style_ax(ax5, "5. A5b Track Ownership")
    if is_usable and "a5b" in tracking_outputs:
        mask = tracking_outputs["a5b"].track_ownership_masks[f_idx]
        rgba = build_track_overlay(mask)
        rgba_up = np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1)
        ax5.imshow(np.clip(rd, -100, 100), cmap="gray", origin="upper", alpha=0.35)
        ax5.imshow(rgba_up, origin="upper")
        # Annotate active tracks
        u = np.unique(mask)
        ann = [f"T{tid}({np.sum(mask == tid)})" for tid in u if tid > 0]
        ax5.text(8, 245, " ".join(ann) if ann else "No Active Tracks", color="white", fontsize=9, backgroundcolor="#00000088")
    else:
        ax5.text(128, 128, "No A5b Data", color="gray", ha="center", va="center")

    # -------------------------------------------------------------
    # Panel 6: S1 Seed-Guided Birth Ownership
    # -------------------------------------------------------------
    ax6 = axes[1, 1]
    style_ax(ax6, "6. S1 Seed-Guided Birth")
    if is_usable and "s1" in tracking_outputs:
        mask = tracking_outputs["s1"].track_ownership_masks[f_idx]
        rgba = build_track_overlay(mask)
        rgba_up = np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1)
        ax6.imshow(np.clip(rd, -100, 100), cmap="gray", origin="upper", alpha=0.35)
        ax6.imshow(rgba_up, origin="upper")
        u = np.unique(mask)
        ann = [f"T{tid}({np.sum(mask == tid)})" for tid in u if tid > 0]
        ax6.text(8, 245, " ".join(ann) if ann else "No Active Tracks", color="white", fontsize=9, backgroundcolor="#00000088")
    else:
        ax6.text(128, 128, "No S1 Data", color="gray", ha="center", va="center")

    # -------------------------------------------------------------
    # Panel 7: S2 Seed-Aware Association Ownership
    # -------------------------------------------------------------
    ax7 = axes[1, 2]
    style_ax(ax7, "7. S2 Seed-Aware Association")
    if is_usable and "s2" in tracking_outputs:
        mask = tracking_outputs["s2"].track_ownership_masks[f_idx]
        rgba = build_track_overlay(mask)
        rgba_up = np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1)
        ax7.imshow(np.clip(rd, -100, 100), cmap="gray", origin="upper", alpha=0.35)
        ax7.imshow(rgba_up, origin="upper")
        u = np.unique(mask)
        ann = [f"T{tid}({np.sum(mask == tid)})" for tid in u if tid > 0]
        ax7.text(8, 245, " ".join(ann) if ann else "No Active Tracks", color="white", fontsize=9, backgroundcolor="#00000088")
    else:
        ax7.text(128, 128, "No S2 Data", color="gray", ha="center", va="center")

    # -------------------------------------------------------------
    # Panel 8: Diagnostic Summary / Benchmark Info
    # -------------------------------------------------------------
    ax8 = axes[1, 3]
    ax8.set_facecolor("#111111")
    ax8.set_title("8. Frame Diagnostic Metadata", color="cyan", fontsize=13, fontweight="bold", pad=6)
    ax8.axis("off")
    
    info_lines = [
        f"Pair Index: {pair_idx}  (Frame: {f_idx})",
        f"Timestamp: {ts} UTC",
        f"Observation Status: {status}",
        f"Usable Detection: {is_usable}",
        f"Active T24 Patches: {n_active}",
        f"Seed Candidates (22-30px): {len(candidates)}",
        f"Representative Seeds: {len(rep_seeds)}",
        "",
        "--- Track Patch Counts ---",
    ]
    for m in ["a5b", "s1", "s2", "spt0", "spt1", "spt2"]:
        if m in tracking_outputs and is_usable:
            m_mask = tracking_outputs[m].track_ownership_masks[f_idx]
            u_tids = [tid for tid in np.unique(m_mask) if tid > 0]
            counts_str = ", ".join([f"T{tid}:{np.sum(m_mask == tid)}" for tid in u_tids]) if u_tids else "None"
            info_lines.append(f"{m.upper():4s}: {counts_str}")
    ax8.text(0.05, 0.95, "\n".join(info_lines), color="white", fontsize=10, family="monospace", va="top")

    # -------------------------------------------------------------
    # Panel 9: SPT-0 (Spatial Overlap) Track Ownership
    # -------------------------------------------------------------
    ax9 = axes[2, 0]
    style_ax(ax9, "9. SPT-0: Spatial Overlap Tracking")
    if is_usable and "spt0" in tracking_outputs:
        mask = tracking_outputs["spt0"].track_ownership_masks[f_idx]
        rgba = build_track_overlay(mask)
        rgba_up = np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1)
        ax9.imshow(np.clip(rd, -100, 100), cmap="gray", origin="upper", alpha=0.35)
        ax9.imshow(rgba_up, origin="upper")
        u = np.unique(mask)
        ann = [f"T{tid}({np.sum(mask == tid)})" for tid in u if tid > 0]
        ax9.text(8, 245, " ".join(ann) if ann else "No Active Tracks", color="white", fontsize=9, backgroundcolor="#00000088")
    else:
        ax9.text(128, 128, "No SPT-0 Data", color="gray", ha="center", va="center")

    # -------------------------------------------------------------
    # Panel 10: SPT-1 (Motion Prediction vr) Track Ownership
    # -------------------------------------------------------------
    ax10 = axes[2, 1]
    style_ax(ax10, "10. SPT-1: Kinematic Motion Prediction")
    if is_usable and "spt1" in tracking_outputs:
        mask = tracking_outputs["spt1"].track_ownership_masks[f_idx]
        rgba = build_track_overlay(mask)
        rgba_up = np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1)
        ax10.imshow(np.clip(rd, -100, 100), cmap="gray", origin="upper", alpha=0.35)
        ax10.imshow(rgba_up, origin="upper")
        # Overplot predicted front radius circles for active tracks
        df_spt1 = tracking_outputs["spt1"].track_states_df
        cur_states = df_spt1[df_spt1["pair_index"] == pair_idx]
        for _, r in cur_states.iterrows():
            tid = int(r["track_id"])
            rf_pred = r.get("predicted_front_radius", np.nan)
            if np.isfinite(rf_pred) and rf_pred > 0:
                c_pred = Circle((sx, sy), rf_pred, edgecolor=get_color_for_track(tid), facecolor="none", linestyle=":", linewidth=1.2)
                ax10.add_patch(c_pred)
        u = np.unique(mask)
        ann = [f"T{tid}({np.sum(mask == tid)})" for tid in u if tid > 0]
        ax10.text(8, 245, " ".join(ann) if ann else "No Active Tracks", color="white", fontsize=9, backgroundcolor="#00000088")
    else:
        ax10.text(128, 128, "No SPT-1 Data", color="gray", ha="center", va="center")

    # -------------------------------------------------------------
    # Panel 11: SPT-2 (Geodesic Arbitration) Track Ownership
    # -------------------------------------------------------------
    ax11 = axes[2, 2]
    style_ax(ax11, "11. SPT-2: Geodesic Patch Arbitration")
    if is_usable and "spt2" in tracking_outputs:
        mask = tracking_outputs["spt2"].track_ownership_masks[f_idx]
        rgba = build_track_overlay(mask)
        rgba_up = np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1)
        ax11.imshow(np.clip(rd, -100, 100), cmap="gray", origin="upper", alpha=0.35)
        ax11.imshow(rgba_up, origin="upper")
        # Overplot leading fronts
        df_spt2 = tracking_outputs["spt2"].track_states_df
        cur_states = df_spt2[df_spt2["pair_index"] == pair_idx]
        for _, r in cur_states.iterrows():
            tid = int(r["track_id"])
            rf = r.get("front_radius", np.nan)
            if np.isfinite(rf) and rf > 0 and r["state"] == "OBSERVED":
                c_front = Circle((sx, sy), rf, edgecolor=get_color_for_track(tid), facecolor="none", linestyle="-", linewidth=1.5)
                ax11.add_patch(c_front)
        u = np.unique(mask)
        ann = [f"T{tid}({np.sum(mask == tid)})" for tid in u if tid > 0]
        ax11.text(8, 245, " ".join(ann) if ann else "No Active Tracks", color="white", fontsize=9, backgroundcolor="#00000088")
    else:
        ax11.text(128, 128, "No SPT-2 Data", color="gray", ha="center", va="center")

    # -------------------------------------------------------------
    # Panel 12: SPT-2 Final Downstream Pixel Segmentation (256x256)
    # -------------------------------------------------------------
    ax12 = axes[2, 3]
    style_ax(ax12, "12. SPT-2: Downstream Pixel Segmentation")
    if is_usable and spt2_downstream is not None:
        seg_out = spt2_downstream.segmentation_output
        fam_masks = seg_out.family_pixel_masks
        ax12.imshow(np.clip(rd, -100, 100), cmap="gray", origin="upper", alpha=0.35)
        seg_rgba = np.zeros((256, 256, 4), dtype=np.float32)
        active_fams = []
        for fid, fmask_all in fam_masks.items():
            f_frame = fmask_all[f_idx]
            if f_frame.any():
                active_fams.append(f"{fid}({np.sum(f_frame)}px)")
                # Get track ID from family ID (e.g. FAM_T03 -> 3)
                try:
                    tid = int(fid.replace("FAM_T", "").replace("FAM_", ""))
                except ValueError:
                    tid = 1
                hex_c = get_color_for_track(tid)
                r = int(hex_c[1:3], 16) / 255.0
                g = int(hex_c[3:5], 16) / 255.0
                b = int(hex_c[5:7], 16) / 255.0
                seg_rgba[f_frame] = [r, g, b, 0.85]
        ax12.imshow(seg_rgba, origin="upper")
        ax12.text(8, 245, " ".join(active_fams) if active_fams else "No Segmented Families", color="white", fontsize=9, backgroundcolor="#00000088")
    else:
        ax12.text(128, 128, "No Segmentation Data", color="gray", ha="center", va="center")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_png.name}")


def generate_animated_gif(image_paths: List[Path], output_gif: Path, duration_ms: int = 400) -> None:
    """Combines a list of PNG image paths into an animated GIF."""
    if not image_paths:
        return
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for p in image_paths:
        if p.exists():
            im = Image.open(p)
            images.append(im.convert("RGB"))
    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(f"Generated animation: {output_gif.name} ({len(images)} frames, {output_gif.stat().st_size / 1024:.1f} KB)")


def generate_method_only_gif(
    method_name: str,
    pair_indices: List[int],
    det: Any,
    tracking_out: Any,
    output_gif: Path,
    duration_ms: int = 350,
) -> None:
    """Generates a clean single-panel animated GIF for a specific tracking method across pairs."""
    geom = det.geometry
    sx, sy = float(geom.sx), float(geom.sy)
    temp_frames = []

    tmp_dir = output_gif.parent / f"_tmp_{method_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for p_idx in pair_indices:
        f_idx_matches = det.manifest_df[det.manifest_df["pair_index"] == p_idx].index
        if len(f_idx_matches) == 0:
            continue
        f_idx = int(f_idx_matches[0])
        row = det.manifest_df.iloc[f_idx]
        ts = str(row["end_time"])[:19]
        is_usable = bool(row.get("usable_detection", True))
        status = str(row.get("observation_status", "UNKNOWN"))

        fc = det.frame_cache[f_idx]
        rd = np.asarray(fc.get("rd", np.zeros((256, 256))), dtype=float)

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
        ax.set_facecolor("black")
        ax.set_title(f"{method_name.upper()} Tracking | Pair {p_idx:02d} | {ts} UTC", color="cyan", fontsize=14, fontweight="bold", pad=8)
        ax.set_xlim(-0.5, 255.5)
        ax.set_ylim(255.5, -0.5)
        ax.tick_params(colors="gray", labelsize=8)

        occ = Circle((sx, sy), 16.326, edgecolor="white", facecolor="none", linestyle=":", linewidth=1.0)
        ann_min = Circle((sx, sy), 22.0, edgecolor="red", facecolor="none", linestyle="--", linewidth=1.0, alpha=0.7)
        ann_max = Circle((sx, sy), 30.0, edgecolor="yellow", facecolor="none", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.add_patch(occ)
        ax.add_patch(ann_min)
        ax.add_patch(ann_max)

        if is_usable:
            ax.imshow(np.clip(rd, -120, 120), cmap="gray", origin="upper", alpha=0.4)
            mask = tracking_out.track_ownership_masks[f_idx]
            rgba = build_track_overlay(mask)
            rgba_up = np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1)
            ax.imshow(rgba_up, origin="upper")

            u = np.unique(mask)
            active_info = [f"T{tid}:{np.sum(mask == tid)}" for tid in u if tid > 0]
            txt = "Active: " + (", ".join(active_info) if active_info else "None")
            ax.text(8, 245, txt, color="white", fontsize=10, backgroundcolor="#000000aa")
        else:
            ax.text(128, 128, f"OBSERVATION INVALID\n({status})", color="red", ha="center", va="center", fontsize=16, fontweight="bold")

        frame_path = tmp_dir / f"frame_{p_idx:03d}.png"
        plt.savefig(frame_path, dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        temp_frames.append(frame_path)

    generate_animated_gif(temp_frames, output_gif, duration_ms=duration_ms)
    # Cleanup temp
    for p in temp_frames:
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()


def generate_synchronized_comparison_gif(
    pair_indices: List[int],
    det: Any,
    tracking_outputs: Dict[str, Any],
    output_gif: Path,
    duration_ms: int = 400,
) -> None:
    """Generates a synchronized 2x3 animated GIF showing A5b, S1, S2, SPT-0, SPT-1, SPT-2 side-by-side."""
    geom = det.geometry
    sx, sy = float(geom.sx), float(geom.sy)
    temp_frames = []

    tmp_dir = output_gif.parent / "_tmp_sync_comp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    methods = ["a5b", "s1", "s2", "spt0", "spt1", "spt2"]
    titles = [
        "A5b: Baseline Weighted CPA",
        "S1: Seed-Guided Birth",
        "S2: Seed-Aware Association",
        "SPT-0: Spatial Overlap",
        "SPT-1: Kinematic Prediction",
        "SPT-2: Geodesic Arbitration",
    ]

    for p_idx in pair_indices:
        f_idx_matches = det.manifest_df[det.manifest_df["pair_index"] == p_idx].index
        if len(f_idx_matches) == 0:
            continue
        f_idx = int(f_idx_matches[0])
        row = det.manifest_df.iloc[f_idx]
        ts = str(row["end_time"])[:19]
        is_usable = bool(row.get("usable_detection", True))
        status = str(row.get("observation_status", "UNKNOWN"))

        fc = det.frame_cache[f_idx]
        rd = np.asarray(fc.get("rd", np.zeros((256, 256))), dtype=float)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="black")
        plt.subplots_adjust(wspace=0.12, hspace=0.16, left=0.04, right=0.96, top=0.92, bottom=0.04)
        fig.suptitle(
            f"Synchronized Tracking Comparison | Pair {p_idx:02d} | {ts} UTC",
            color="white", fontsize=18, fontweight="bold", y=0.97
        )

        for ax_idx, (m, title) in enumerate(zip(methods, titles)):
            ax = axes.ravel()[ax_idx]
            ax.set_facecolor("black")
            ax.set_title(title, color="cyan", fontsize=13, fontweight="bold", pad=6)
            ax.set_xlim(-0.5, 255.5)
            ax.set_ylim(255.5, -0.5)
            ax.tick_params(colors="gray", labelsize=8)

            occ = Circle((sx, sy), 16.326, edgecolor="white", facecolor="none", linestyle=":", linewidth=1.0)
            ann_min = Circle((sx, sy), 22.0, edgecolor="red", facecolor="none", linestyle="--", linewidth=1.0, alpha=0.7)
            ann_max = Circle((sx, sy), 30.0, edgecolor="yellow", facecolor="none", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.add_patch(occ)
            ax.add_patch(ann_min)
            ax.add_patch(ann_max)

            if is_usable and m in tracking_outputs:
                ax.imshow(np.clip(rd, -120, 120), cmap="gray", origin="upper", alpha=0.4)
                mask = tracking_outputs[m].track_ownership_masks[f_idx]
                rgba = build_track_overlay(mask)
                rgba_up = np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1)
                ax.imshow(rgba_up, origin="upper")

                u = np.unique(mask)
                active_info = [f"T{tid}:{np.sum(mask == tid)}" for tid in u if tid > 0]
                txt = " ".join(active_info) if active_info else "No Active Tracks"
                ax.text(8, 245, txt, color="white", fontsize=9, backgroundcolor="#000000aa")
            else:
                ax.text(128, 128, f"OBSERVATION INVALID\n({status})", color="red", ha="center", va="center", fontsize=14, fontweight="bold")

        frame_path = tmp_dir / f"sync_frame_{p_idx:03d}.png"
        plt.savefig(frame_path, dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        temp_frames.append(frame_path)

    generate_animated_gif(temp_frames, output_gif, duration_ms=duration_ms)
    for p in temp_frames:
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()


def main():
    print("=" * 80)
    print("STEREO_COR2 SEED RESEARCH E01 — CONSECUTIVE DIAGNOSTICS & ANIMATIONS")
    print("=" * 80)

    # 1. Load standardized detection input
    print("\n[1/6] Loading detection inputs for 2010-12-12...")
    det = load_live_tracking_input(repo_root / "runs/test_d4_20101212")
    cfg = PipelineConfig(date_str="20101212")
    ctx = PipelineContext(cfg)

    # Extract continuous scores
    scores_64 = extract_continuous_scores_from_detection(det, score_mode="abs")

    # 2. Run all 6 tracking methods
    methods = ["a5b", "s1", "s2", "spt0", "spt1", "spt2"]
    tracking_outputs = {}
    print("\n[2/6] Executing all 6 tracking methods...")
    for m in methods:
        stage = get_tracking_stage(m)
        print(f"  Running {m.upper()} ({stage.__class__.__name__})...")
        tracking_outputs[m] = stage.run(det, cfg, ctx)

    # 3. Run SPT-2 downstream pipeline
    print("\n[3/6] Running SPT-2 downstream pipeline (Stages 04-08)...")
    spt2_downstream = run_spt_downstream_pipeline(tracking_outputs["spt2"], cfg, ctx)
    print(f"  SPT-2 Downstream admitted {len(spt2_downstream.admitted_track_ids)} tracks.")

    # 4. Target output directories
    e01_out_dir = repo_root / "outputs/seed_research/e01_track_identity_audit"
    fig_dir = e01_out_dir / "figures"
    anim_dir = e01_out_dir / "animations"
    fig_dir.mkdir(parents=True, exist_ok=True)
    anim_dir.mkdir(parents=True, exist_ok=True)

    # 5. Render individual 12-panel diagnostic PNGs for Pairs 30 through 50
    print("\n[4/6] Rendering 12-panel diagnostic figures for Pairs 30 through 50 (21 pairs)...")
    pair_indices = list(range(30, 51))
    rendered_pngs = []
    for p_idx in pair_indices:
        out_png = fig_dir / f"pair_{p_idx:03d}_diagnostic.png"
        generate_single_frame_panel(
            pair_idx=p_idx,
            det=det,
            scores_64=scores_64,
            tracking_outputs=tracking_outputs,
            spt2_downstream=spt2_downstream,
            output_png=out_png,
        )
        rendered_pngs.append(out_png)

    # 6. Generate Method GIFs & Synchronized Comparison GIF
    print("\n[5/6] Generating individual method animations (GIFs)...")
    for m in methods:
        out_gif = anim_dir / f"{m}_consecutive_30_50.gif"
        print(f"  Creating {out_gif.name}...")
        generate_method_only_gif(
            method_name=m,
            pair_indices=pair_indices,
            det=det,
            tracking_out=tracking_outputs[m],
            output_gif=out_gif,
            duration_ms=400,
        )

    print("\n[6/6] Generating synchronized comparison animation (GIF)...")
    sync_gif = anim_dir / "synchronized_comparison_30_50.gif"
    generate_synchronized_comparison_gif(
        pair_indices=pair_indices,
        det=det,
        tracking_outputs=tracking_outputs,
        output_gif=sync_gif,
        duration_ms=450,
    )

    print("\n" + "=" * 80)
    print(f"SUCCESS: Generated {len(rendered_pngs)} diagnostic panels and 7 animations in {e01_out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
