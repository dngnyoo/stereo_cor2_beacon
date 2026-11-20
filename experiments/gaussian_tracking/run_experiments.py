"""Real-Data Experimental Runner & Sensitivity Analysis for Standalone Gaussian Evidence Tracking.

Evaluates 2010-12-12 STEREO-A COR2 Running Difference sequence (63 frames):
- G0: Raw pixel evidence (sigma_s = 0)
- G1: Spatial Gaussian spreading (sigma_s = 2.0 px)
- G2: Causal temporal accumulation memory (tau_1/2 = 40 min)
- G3: Motion-aware association (trajectory prediction discovering v_r)
- Sensitivity analysis over sigma_s in {1, 2, 4, 6} px
- Sensitivity analysis over formulation (positive vs compressed)
- Sensitivity analysis over evidence threshold tau in {0.3, 0.5, 0.8, 1.2}
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from cme_pipeline.cadence_selection import parse_timestamp
from cme_pipeline.utils.geometry import compute_wcs_rasters, circular_diff_deg
from experiments.gaussian_tracking.gaussian_tracker import (
    TrackerConfig,
    GaussianEvidenceTracker,
    GaussianTrack,
    FrameTrackingResult,
)


class RealDataPipelineRunner:
    """Orchestrates loading, preprocessing, and tracker execution on 2010-12-12 dataset."""

    def __init__(
        self,
        data_dir: str | Path = "runs/dual_archive_20101212/preprocessed",
        output_dir: str | Path = "experiments/gaussian_tracking/results",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load manifest and raw files
        self.manifest = pd.read_csv(self.data_dir / "01_rd_manifest.csv")
        self.n_frames = len(self.manifest)
        self.rd_images = np.load(self.data_dir / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
        self.valid_masks = np.load(self.data_dir / "north_up_pair_valid_masks.npy", mmap_mode="r")

        # 2. WCS and solar geometry
        hdr_str = (self.data_dir / "target_wcs_header.txt").read_text(encoding="utf-8")
        self.wcs = WCS(fits.Header.fromstring(hdr_str, sep="\n"))
        sx, sy = self.wcs.world_to_pixel_values(0.0, 0.0)
        self.sx = float(sx)
        self.sy = float(sy)

        rasters = compute_wcs_rasters(wcs=self.wcs, center_xy=(self.sx, self.sy), patch_size=4)
        self.radius_raster = rasters["radius_raster"]
        self.pa_raster = rasters["pa_raster"]

        # Parse timestamps
        self.timestamps = [parse_timestamp(t) for t in self.manifest["end_time"]]

        # Precompute standardized smoothed z-scores (reproducing cme_pipeline/live_tracking_input.py)
        print("Precomputing frame-level Gaussian-smoothed z-scores...")
        self.z_maps = self._precompute_z_maps()
        print(f"Loaded {self.n_frames} frames. Geometry center: ({self.sx:.2f}, {self.sy:.2f})")

    def _precompute_z_maps(self, gaussian_sigma: float = 1.0) -> np.ndarray:
        """Precomputes standardized z-score maps identically to baseline pipeline."""
        z_maps = np.zeros((self.n_frames, 256, 256), dtype=np.float32)
        for idx in range(self.n_frames):
            image = np.asarray(self.rd_images[idx], dtype=np.float32)
            val = np.asarray(self.valid_masks[idx], dtype=bool)

            num = gaussian_filter(np.where(val, image, 0.0), gaussian_sigma, mode="constant", cval=0.0)
            den = gaussian_filter(val.astype(float), gaussian_sigma, mode="constant", cval=0.0)

            good = val & (den > 0)
            smoothed = np.full(image.shape, np.nan, dtype=np.float32)
            smoothed[good] = (num[good] / den[good]).astype(np.float32)

            fin = smoothed[good & np.isfinite(smoothed)]
            if fin.size and float(fin.std()) > 0:
                z_maps[idx, good] = (smoothed[good] - float(fin.mean())) / float(fin.std())

        return z_maps

    def run_configuration(
        self,
        cfg: TrackerConfig,
        run_name: str,
        collect_frame_results: bool = False,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, List[FrameTrackingResult]]:
        """Runs single tracker configuration over all frames."""
        tracker = GaussianEvidenceTracker(
            config=cfg,
            sx=self.sx,
            sy=self.sy,
            radius_raster=self.radius_raster,
            pa_raster=self.pa_raster,
        )

        t0 = time.time()
        frame_results: List[FrameTrackingResult] = []

        for idx in range(self.n_frames):
            res = tracker.step(
                frame_idx=idx,
                timestamp=self.timestamps[idx],
                z_map=self.z_maps[idx],
                valid_mask=self.valid_masks[idx],
            )
            if collect_frame_results:
                frame_results.append(res)

        elapsed_sec = time.time() - t0
        all_tracks = tracker.get_all_tracks()
        confirmed_tracks = tracker.get_confirmed_tracks()
        tentative_tracks = [t for t in all_tracks if t.status == "tentative"]

        # Track-level metrics
        if confirmed_tracks:
            lifespans = [t.hits for t in confirmed_tracks]
            mean_life = float(np.mean(lifespans))
            max_life = int(np.max(lifespans))
            jitters = [t.cpa_jitter for t in confirmed_tracks if t.hits >= 3]
            mean_jitter = float(np.mean(jitters)) if jitters else 0.0
            v_rads = [t.mean_radial_velocity for t in confirmed_tracks if t.hits >= 3]
            mean_vr = float(np.mean(v_rads)) if v_rads else 0.0
            consistencies = [t.outward_consistency for t in confirmed_tracks if t.hits >= 2]
            mean_consist = float(np.mean(consistencies)) if consistencies else 1.0
        else:
            mean_life, max_life, mean_jitter, mean_vr, mean_consist = 0.0, 0, 0.0, 0.0, 0.0

        summary = {
            "run_name": run_name,
            "variant": cfg.variant,
            "sigma_s": cfg.sigma_s,
            "formulation": cfg.evidence_formulation,
            "threshold": cfg.evidence_threshold,
            "birth_mode": cfg.birth_mode,
            "total_tracks_born": len(all_tracks),
            "confirmed_tracks": len(confirmed_tracks),
            "tentative_tracks": len(tentative_tracks),
            "mean_lifespan_frames": round(mean_life, 2),
            "max_lifespan_frames": max_life,
            "mean_cpa_jitter_deg": round(mean_jitter, 2),
            "mean_radial_velocity_px_hr": round(mean_vr, 2),
            "mean_outward_consistency": round(mean_consist, 3),
            "runtime_sec": round(elapsed_sec, 2),
        }

        df_tracks = tracker.to_dataframe()
        return summary, df_tracks, frame_results


def run_all_experiments():
    runner = RealDataPipelineRunner()
    results_dir = Path("experiments/gaussian_tracking/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # 1. Primary Benchmark: G0, G1, G2, G3
    primary_configs = [
        ("G0_Raw", TrackerConfig(variant="G0", sigma_s=0.0)),
        ("G1_Spatial_sigma2", TrackerConfig(variant="G1", sigma_s=2.0)),
        ("G2_Temporal_sigma2", TrackerConfig(variant="G2", sigma_s=2.0, tau_half_life_min=40.0)),
        ("G3_MotionAware_sigma2", TrackerConfig(variant="G3", sigma_s=2.0)),
    ]

    print("\n=== Running Primary Architecture Benchmark (G0, G1, G2, G3) ===")
    key_frame_results = {}
    for name, cfg in primary_configs:
        print(f"Running {name}...")
        collect_diag = (name in ("G0_Raw", "G1_Spatial_sigma2", "G3_MotionAware_sigma2"))
        summ, df_trk, frame_res = runner.run_configuration(cfg, name, collect_frame_results=collect_diag)
        summary_rows.append(summ)
        # Export track CSV
        df_trk.to_csv(results_dir / f"tracks_{name}.csv", index=False)
        if collect_diag:
            key_frame_results[name] = frame_res
        print(f"  -> Confirmed: {summ['confirmed_tracks']}, Born: {summ['total_tracks_born']}, Max Life: {summ['max_lifespan_frames']}, CPA Jitter: {summ['mean_cpa_jitter_deg']} deg")

    # 2. Gaussian Width Sensitivity Sweep on G1: sigma_s in {1, 2, 4, 6}
    print("\n=== Running Gaussian Scale Sensitivity Sweep (sigma_s in {1.0, 2.0, 4.0, 6.0}) ===")
    for s in [1.0, 2.0, 4.0, 6.0]:
        name = f"G1_sigma_{int(s)}px" if s.is_integer() else f"G1_sigma_{s}px"
        cfg = TrackerConfig(variant="G1", sigma_s=s)
        summ, df_trk, _ = runner.run_configuration(cfg, name)
        summary_rows.append(summ)
        df_trk.to_csv(results_dir / f"tracks_{name}.csv", index=False)
        print(f"  -> sigma_s={s}: Confirmed={summ['confirmed_tracks']}, Max Life={summ['max_lifespan_frames']}, Jitter={summ['mean_cpa_jitter_deg']} deg")

    # 3. Evidence Formulation Sensitivity: Positive vs Compressed
    print("\n=== Running Evidence Formulation Sensitivity (Positive vs Compressed) ===")
    cfg_comp = TrackerConfig(variant="G1", sigma_s=2.0, evidence_formulation="compressed", evidence_threshold=0.35)
    summ, df_trk, _ = runner.run_configuration(cfg_comp, "G1_Compressed_tau0.35")
    summary_rows.append(summ)
    df_trk.to_csv(results_dir / "tracks_G1_Compressed.csv", index=False)
    print(f"  -> Compressed: Confirmed={summ['confirmed_tracks']}, Max Life={summ['max_lifespan_frames']}, Jitter={summ['mean_cpa_jitter_deg']} deg")

    # 4. Threshold Sensitivity Sweep: tau in {0.3, 0.5, 0.8, 1.2}
    print("\n=== Running Evidence Threshold Sensitivity (tau in {0.3, 0.5, 0.8, 1.2}) ===")
    for tau in [0.3, 0.5, 0.8, 1.2]:
        name = f"G1_threshold_{tau}"
        cfg = TrackerConfig(variant="G1", sigma_s=2.0, evidence_threshold=tau)
        summ, df_trk, _ = runner.run_configuration(cfg, name)
        summary_rows.append(summ)
        print(f"  -> tau={tau}: Confirmed={summ['confirmed_tracks']}, Max Life={summ['max_lifespan_frames']}, Jitter={summ['mean_cpa_jitter_deg']} deg")

    # Export overall summary
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "track_summary_table.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved overall experimental summary to {summary_csv}")
    print("\nSummary Table:")
    print(df_summary[["run_name", "confirmed_tracks", "total_tracks_born", "mean_lifespan_frames", "max_lifespan_frames", "mean_cpa_jitter_deg", "mean_outward_consistency"]])


if __name__ == "__main__":
    run_all_experiments()
