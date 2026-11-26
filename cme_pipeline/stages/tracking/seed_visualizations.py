"""Diagnostic visualizations for Score-Ranked Seed-Based CME Component Identification.

Generates comprehensive multi-panel diagnostic figures satisfying Section 7 requirements:
1. Original 256x256 north-up RD image
2. Continuous patch detection score heatmap (64x64)
3. D4/T24 64x64 patch activation map with inner-corona seed-search boundary (r_min, R_seed)
4. Seed candidates labeled by score rank (with representative seeds highlighted)
5. Merged seed groups and 8-connected growth results
6. Seeded versus unseeded components partition
7. A5b baseline tracking ownership vs Seeded components comparison
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

from cme_pipeline.core.stage import DetectionOutput, GeometryContext
from cme_pipeline.stages.segmentation.visualization import get_rd_cmap
from cme_pipeline.stages.tracking.seed_component_pilot import FrameSeedResult


def generate_seed_diagnostic_figure(
    frame_idx: int,
    frame_seed_result: FrameSeedResult,
    det_out: DetectionOutput,
    a5b_ownership_frame: Optional[np.ndarray] = None,
    r_min: float = 22.0,
    r_seed: float = 30.0,
    output_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Renders a 6-panel comprehensive diagnostic figure for one frame."""
    res = frame_seed_result
    p_idx = res.pair_index
    ts = res.timestamp
    geom = det_out.geometry
    fc = det_out.frame_cache[frame_idx]

    rd = np.asarray(fc.get("rd", np.zeros((256, 256))), dtype=float)
    val = np.asarray(fc.get("usable", np.ones((256, 256), dtype=bool)), dtype=bool)
    masked_rd = np.ma.masked_where(~val, rd)

    sx, sy = geom.sx, geom.sy
    scores = res.continuous_scores
    active_mask = det_out.patch_activation_grid4[frame_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

    # Standard styling for all image panels
    for ax in axes.ravel():
        ax.set_xlim(-0.5, 255.5)
        ax.set_ylim(-0.5, 255.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (pixels, North-Up)", fontsize=9)
        ax.set_ylabel("Y (pixels, North-Up)", fontsize=9)

    # Helper to draw solar center & radial circles
    def add_radial_circles(ax, color_seed="gold", color_occ="red"):
        # Solar center
        ax.plot(sx, sy, marker="+", color="yellow", markersize=8, markeredgewidth=1.5)
        # Occulter inner boundary r_min
        ax.add_patch(Circle((sx, sy), r_min, fill=False, linestyle=":", edgecolor=color_occ, linewidth=1.2, label=f"Occulter r_min={r_min:.1f}px"))
        # Seed search outer boundary R_seed
        ax.add_patch(Circle((sx, sy), r_seed, fill=False, linestyle="--", edgecolor=color_seed, linewidth=1.4, label=f"R_seed={r_seed:.1f}px"))
        # Outer FOV boundary
        ax.add_patch(Circle((sx, sy), geom.fov_r_max_px, fill=False, linestyle="-.", edgecolor="gray", linewidth=0.8, alpha=0.7))

    # -------------------------------------------------------------------------
    # Panel 1: Original 256x256 North-Up RD Image
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    im1 = ax1.imshow(masked_rd, origin="lower", cmap=get_rd_cmap(), vmin=-500, vmax=500, extent=(-0.5, 255.5, -0.5, 255.5))
    fig.colorbar(im1, ax=ax1, shrink=0.7, label="RD Intensity (DN/s)")
    add_radial_circles(ax1)
    ax1.set_title(f"1. North-Up RD Image\nPair {p_idx} | {ts[:19]}", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 2: Continuous Patch Detection Score Heatmap (64x64)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    finite_scores = scores[np.isfinite(scores)]
    vmax_score = float(np.percentile(finite_scores, 99.5)) if len(finite_scores) > 0 else 10.0
    im2 = ax2.imshow(scores, origin="lower", extent=(-0.5, 255.5, -0.5, 255.5), cmap="magma", vmin=0, vmax=max(5.0, vmax_score), interpolation="nearest")
    fig.colorbar(im2, ax=ax2, shrink=0.7, label="D4 Continuous Score S_p (P95 |Z|)")
    add_radial_circles(ax2, color_seed="cyan")
    ax2.set_title(f"2. Continuous D4 Patch Score Heatmap\nEffective Threshold T_eff={res.effective_threshold:.2f}", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: D4/T24 Patch Activation Map & Seed Search Corridor
    # -------------------------------------------------------------------------
    ax3 = axes[0, 2]
    ax3.imshow(masked_rd, origin="lower", cmap="gray", vmin=-300, vmax=300, extent=(-0.5, 255.5, -0.5, 255.5), alpha=0.55)
    # Draw active patches
    yy, xx = np.nonzero(active_mask)
    if len(yy):
        patches = [Rectangle((int(c) * 4 - 0.5, int(r) * 4 - 0.5), 4, 4) for r, c in zip(yy, xx)]
        ax3.add_collection(PatchCollection(patches, facecolor="#00FFCC", edgecolor="none", alpha=0.75))
    add_radial_circles(ax3)
    ax3.set_title(f"3. D4/T24 Patch Activation Map\n{int(active_mask.sum())} Active Patches (Cyan)", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 4: Seed Candidates Labeled by Score Rank & Representatives
    # -------------------------------------------------------------------------
    ax4 = axes[1, 0]
    ax4.imshow(masked_rd, origin="lower", cmap="gray", vmin=-300, vmax=300, extent=(-0.5, 255.5, -0.5, 255.5), alpha=0.5)
    add_radial_circles(ax4)

    # Plot all candidates and highlight representative seeds
    if res.candidates:
        c_ranks = [c.score_rank for c in res.candidates]
        c_rows = [c.patch_r for c in res.candidates]
        c_cols = [c.patch_c for c in res.candidates]
        c_rep = [c.is_representative for c in res.candidates]

        # Draw candidate boxes
        cand_patches = [Rectangle((c * 4 - 0.5, r * 4 - 0.5), 4, 4) for r, c in zip(c_rows, c_cols)]
        ax4.add_collection(PatchCollection(cand_patches, facecolor="orange", edgecolor="gold", alpha=0.6, linewidth=0.8))

        # Annotate representative seeds
        for c in res.candidates:
            px = c.patch_c * 4 + 1.5
            py = c.patch_r * 4 + 1.5
            if c.is_representative:
                ax4.plot(px, py, marker="*", color="red", markersize=9, markeredgecolor="white", markeredgewidth=1)
                ax4.text(px + 2, py + 2, f"R{c.score_rank}\n({c.raw_score:.1f})", color="yellow", fontsize=7.5, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.7, edgecolor="red"))
            elif c.score_rank <= 10:
                ax4.text(px + 1, py + 1, f"{c.score_rank}", color="white", fontsize=6, fontweight="bold")

    ax4.set_title(f"4. Ranked Seed Candidates (R1..R{len(res.candidates)})\n{len(res.seed_groups)} Seed Groups (* = Rep Seed)", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 5: Merged Seed Groups and 8-Connected Component Growth
    # -------------------------------------------------------------------------
    ax5 = axes[1, 1]
    ax5.imshow(masked_rd, origin="lower", cmap="gray", vmin=-300, vmax=300, extent=(-0.5, 255.5, -0.5, 255.5), alpha=0.5)
    add_radial_circles(ax5)

    # Colormap for seed groups
    group_cmap = cm.get_cmap("tab10")
    if res.seed_groups:
        for idx, g in enumerate(res.seed_groups):
            gid = g.seed_group_id
            col = group_cmap(idx % 10)
            gm = (res.seeded_component_mask == gid)
            gyy, gxx = np.nonzero(gm)
            if len(gyy):
                gpatches = [Rectangle((int(c) * 4 - 0.5, int(r) * 4 - 0.5), 4, 4) for r, c in zip(gyy, gxx)]
                ax5.add_collection(PatchCollection(gpatches, facecolor=col, edgecolor="black", alpha=0.7, linewidth=0.5))

            # Mark representative seed
            rep = g.representative_seed
            rpx = rep.patch_c * 4 + 1.5
            rpy = rep.patch_r * 4 + 1.5
            ax5.plot(rpx, rpy, marker="*", color="yellow", markersize=10, markeredgecolor="black")
            ax5.text(rpx + 2, rpy + 2, f"G{gid}: {g.total_grown_patches}px\nCPA={g.cpa_deg:.0f}°", color="white", fontsize=7.5,
                     fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor=col, alpha=0.85, edgecolor="white"))

    ax5.set_title(f"5. Seeded Component Growth\n{len(res.seed_groups)} Seeded Structures", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 6: Seeded vs. Unseeded Components Partition (or A5b Comparison)
    # -------------------------------------------------------------------------
    ax6 = axes[1, 2]
    ax6.imshow(masked_rd, origin="lower", cmap="gray", vmin=-300, vmax=300, extent=(-0.5, 255.5, -0.5, 255.5), alpha=0.5)
    add_radial_circles(ax6)

    # Seeded components in color
    if res.seed_groups:
        for idx, g in enumerate(res.seed_groups):
            gid = g.seed_group_id
            col = group_cmap(idx % 10)
            gm = (res.seeded_component_mask == gid)
            gyy, gxx = np.nonzero(gm)
            if len(gyy):
                ax6.add_collection(PatchCollection(
                    [Rectangle((int(c) * 4 - 0.5, int(r) * 4 - 0.5), 4, 4) for r, c in zip(gyy, gxx)],
                    facecolor=col, edgecolor="white", alpha=0.8, linewidth=0.5
                ))

    # Unseeded components in hatched gray/orange
    unseeded_m = (res.unseeded_component_mask > 0)
    uyy, uxx = np.nonzero(unseeded_m)
    if len(uyy):
        ax6.add_collection(PatchCollection(
            [Rectangle((int(c) * 4 - 0.5, int(r) * 4 - 0.5), 4, 4) for r, c in zip(uyy, uxx)],
            facecolor="none", edgecolor="red", hatch="//", alpha=0.9, linewidth=0.8
        ))

    # If A5b ownership is provided, overlay A5b track outlines or fronts
    if a5b_ownership_frame is not None:
        a5b_tracks = [tid for tid in np.unique(a5b_ownership_frame) if tid > 0]
        for tid in a5b_tracks:
            tm = (a5b_ownership_frame == tid)
            # Find centroid
            tyy, txx = np.nonzero(tm)
            if len(tyy):
                cx = float(np.mean(txx) * 4 + 2)
                cy = float(np.mean(tyy) * 4 + 2)
                ax6.text(cx, cy, f"A5b T{tid}", color="cyan", fontsize=8, fontweight="bold",
                         bbox=dict(boxstyle="square,pad=0.1", facecolor="darkblue", alpha=0.7))

    n_seeded = sum(1 for d in res.component_diagnostics if d.is_seeded)
    n_unseeded = sum(1 for d in res.component_diagnostics if not d.is_seeded)
    ax6.set_title(f"6. Seeded (Colors) vs Unseeded (Red //)\nSeeded: {n_seeded} | Unseeded: {n_unseeded}", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"STEREO-A COR2 Seed-Based CME Pilot | Pair {p_idx:03d} | {ts} | Status: {res.observation_status}\n"
        f"Inner Corona: [{r_min:.1f}, {r_seed:.1f}] px | Candidates: {len(res.candidates)} | Seed Groups: {len(res.seed_groups)} | Equivalence Verified: {res.equivalence_verified}",
        fontsize=13, fontweight="bold"
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    return fig


def save_pilot_diagnostics_gallery(
    pilot_results: Dict[str, Any],
    det_out: DetectionOutput,
    output_dir: Path,
    a5b_ownership_masks: Optional[np.ndarray] = None,
    r_min: float = 22.0,
    r_seed: float = 30.0,
    sample_pair_indices: Optional[List[int]] = None,
) -> List[Path]:
    """Generates and saves diagnostic figures for sampled evaluation frames."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_results: List[FrameSeedResult] = pilot_results["frame_results"]
    saved_paths: List[Path] = []

    for i, res in enumerate(frame_results):
        if not res.usable_detection:
            continue
        if sample_pair_indices is not None and res.pair_index not in sample_pair_indices:
            continue

        a5b_frame = a5b_ownership_masks[i] if a5b_ownership_masks is not None else None
        p_str = f"pair_{res.pair_index:03d}_{res.timestamp.replace(':', '').replace('-', '')[:15]}"
        fig_path = output_dir / f"{p_str}_seed_diagnostics.png"

        generate_seed_diagnostic_figure(
            frame_idx=i,
            frame_seed_result=res,
            det_out=det_out,
            a5b_ownership_frame=a5b_frame,
            r_min=r_min,
            r_seed=r_seed,
            output_path=fig_path,
        )
        saved_paths.append(fig_path)

    return saved_paths
