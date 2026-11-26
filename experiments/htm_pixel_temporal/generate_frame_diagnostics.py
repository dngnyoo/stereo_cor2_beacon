"""Generates diagnostic frame panels showing temporal evidence and tracking assignments."""

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
from cme_pipeline.stages.tracking.temporal_evidence_model import (
    PixelTemporalEvidenceModel,
    TemporalEvidenceConfig,
)
from cme_pipeline.utils.geometry import circular_diff_deg


def render_diagnostic_frame(
    pair_index: int,
    det_out,
    h3_states_df: pd.DataFrame,
    h4_states_df: pd.DataFrame,
    a5b_states_df: pd.DataFrame,
    model: PixelTemporalEvidenceModel,
    out_path: Path,
):
    man = det_out.manifest_df.reset_index(drop=True)
    match_rows = man[man["pair_index"] == pair_index]
    if match_rows.empty:
        print(f"Pair {pair_index} not in manifest.")
        return
    f_idx = int(match_rows.index[0])

    f_cache = det_out.frame_cache[f_idx]
    z_map = f_cache["z_global"]
    val_map = f_cache["val"]
    geom = det_out.geometry
    ts_str = match_rows.iloc[0]["end_time"]

    # Compute spatial evidence and accumulated evidence
    analysis_mask = val_map & (geom.radius_raster >= geom.occulter_r_min_px) & (geom.radius_raster <= geom.fov_r_max_px)
    accum_ev = model.get_accumulated_evidence()
    spat_ev = model.compute_spatial_evidence(z_map, analysis_mask)
    raw_prof, norm_prof, sm_prof = model.compute_directional_profile(accum_ev, analysis_mask)
    peaks = model.find_directional_peaks(sm_prof)

    # Patch grid and component labels
    patch_labels = det_out.raw_component_labels[f_idx]

    # Filter states for this pair
    h3_p = h3_states_df[h3_states_df["pair_index"] == pair_index]
    a5b_p = a5b_states_df[a5b_states_df["pair_index"] == pair_index]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Standardized z-score
    im0 = axes[0, 0].imshow(np.where(val_map, z_map, np.nan), cmap="gray", origin="lower", vmin=-1.0, vmax=5.0)
    axes[0, 0].set_title(f"Pair {pair_index} ({ts_str})\nPixel Standardized z-score", fontsize=10, fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Pixel Spatial Relative Evidence
    im1 = axes[0, 1].imshow(np.where(analysis_mask, spat_ev, np.nan), cmap="inferno", origin="lower")
    axes[0, 1].set_title("Current-Frame Spatial Relative Evidence\n(Analysis Masked)", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Accumulated Temporal Evidence (H2 Motion Compensated)
    im2 = axes[0, 2].imshow(np.where(analysis_mask, accum_ev, np.nan), cmap="plasma", origin="lower")
    axes[0, 2].set_title("Accumulated Temporal Evidence (H2)\nOutward Warped & Decayed", fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. Directional PA Profile on S^1 & Multi-Peaks
    pa_axis = np.linspace(0, 360, len(sm_prof))
    axes[1, 0].plot(pa_axis, norm_prof, color="gray", alpha=0.5, label="Normalized Profile")
    axes[1, 0].plot(pa_axis, sm_prof, color="crimson", lw=2, label="Gaussian Smoothed")
    for pk in peaks:
        axes[1, 0].axvline(pk.cpa_deg, color="blue", linestyle="--", alpha=0.8)
        axes[1, 0].text(pk.cpa_deg, pk.amplitude * 1.05, f"{pk.cpa_deg:.1f}°", color="blue", fontsize=8, ha="center")
    axes[1, 0].set_xlabel("Position Angle [deg]")
    axes[1, 0].set_ylabel("Evidence Amplitude")
    axes[1, 0].set_title("Directional Evidence Profile & Peaks", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlim(0, 360)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right", fontsize=8)

    # 5. Detected Components (64x64 grid)
    comp_display = np.where(patch_labels > 0, patch_labels % 10 + 1, 0)
    im4 = axes[1, 1].imshow(comp_display, cmap="tab10", origin="lower", interpolation="nearest")
    axes[1, 1].set_title(f"D4 Detection Components (64×64)\nUnique Spatial Components: {len(np.unique(patch_labels[patch_labels > 0]))}", fontsize=10, fontweight="bold")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 6. Tracking Comparison: A5b vs H3 Assignments
    axes[1, 2].imshow(np.where(val_map, z_map, np.nan), cmap="gray", origin="lower", vmin=-1.0, vmax=3.0, alpha=0.6)
    # Overlay peak directions as radial lines from center
    cx, cy = 128.0, 128.0
    for pk in peaks:
        rad = np.radians(pk.cpa_deg)
        dx = -np.sin(rad) * 110.0
        dy = np.cos(rad) * 110.0
        axes[1, 2].plot([cx, cx + dx], [cy, cy + dy], color="cyan", lw=2, linestyle="--", label=f"Temporal Peak {pk.cpa_deg:.0f}°")

    axes[1, 2].set_title("Temporal CPA Prior Sectors\nOverlaid on Coronagraph Field", fontsize=10, fontweight="bold")
    axes[1, 2].set_xlim(0, 256)
    axes[1, 2].set_ylim(0, 256)
    axes[1, 2].legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Generated diagnostic panel: {out_path}")


def main():
    det_out = load_live_tracking_input("runs/clean_cor2a_verify/detection")
    if not np.isfinite(det_out.geometry.rsun_px) or det_out.geometry.rsun_px <= 0:
        det_out.geometry.rsun_px = 8.163
        det_out.geometry.occulter_inner_radius_px = 2.0 * 8.163

    results_dir = repo_root / "experiments/htm_pixel_temporal/results"
    vis_dir = repo_root / "experiments/htm_pixel_temporal/visualizations/tracking"
    vis_dir.mkdir(parents=True, exist_ok=True)

    h3_states = pd.read_csv(results_dir / "track_states_h3_h2.csv") if (results_dir / "track_states_h3_h2.csv").exists() else pd.DataFrame()
    h4_states = pd.read_csv(results_dir / "track_states_h4_h2_direct.csv") if (results_dir / "track_states_h4_h2_direct.csv").exists() else pd.DataFrame()
    a5b_states = pd.read_csv(results_dir / "track_states_a5b_baseline.csv") if (results_dir / "track_states_a5b_baseline.csv").exists() else pd.DataFrame()

    geom = det_out.geometry
    cfg = TemporalEvidenceConfig(variant="H2", half_life_min=45.0, radial_speed_px_30m=8.0)
    model = PixelTemporalEvidenceModel(
        sx=geom.sx, sy=geom.sy,
        pa_raster=geom.pa_raster, radius_raster=geom.radius_raster,
        r_min_px=geom.occulter_r_min_px, r_max_px=geom.fov_r_max_px,
        config=cfg,
    )

    man = det_out.manifest_df.reset_index(drop=True)
    target_pairs = [35, 38, 41, 46, 52, 61]

    # Replay frames sequentially up to max target pair
    max_pair = max(target_pairs)
    for idx, row in man.iterrows():
        p_idx = int(row["pair_index"])
        f_cache = det_out.frame_cache[idx]
        z_map = f_cache["z_global"]
        val_map = f_cache["val"]
        ts = pd.to_datetime(row["end_time"])

        model.update(z_map, val_map, ts)

        if p_idx in target_pairs:
            out_p = vis_dir / f"diag_tracking_pair_{p_idx:02d}.png"
            render_diagnostic_frame(p_idx, det_out, h3_states, h4_states, a5b_states, model, out_p)


if __name__ == "__main__":
    main()
