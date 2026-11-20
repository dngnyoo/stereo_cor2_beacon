r"""Real-Data Experimental Runner & Sensitivity Analysis for Radial Evidence Tracking (RET).

Evaluates 2010-12-12 STEREO-A COR2 Running Difference sequence (63 frames):
- Primary benchmark: RET-0, RET-1, RET-2, RET-3
- Sensitivity sweeps:
  - Gaussian width sigma_s in {0.0, 1.0, 2.0, 4.0, 6.0} px
  - Evidence formulation: positive vs compressed
  - Confirmation persistence: N_confirm in {2, 3}
  - Angular smoothing: sigma_theta in {1.0, 2.0, 4.0} deg
  - Front detection threshold quantile in {0.15, 0.20, 0.30}
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
from experiments.ret_tracking.ret_evidence_engine import PolarGridGeometry
from experiments.ret_tracking.ret_tracker import (
    RETConfig,
    RadialEvidenceTracker,
    RETTrack,
    FrameRETResult,
)


class RETPipelineRunner:
    """Manages data loading, preprocessing, and execution of RET experiments."""

    def __init__(
        self,
        data_dir: str | Path = "runs/dual_archive_20101212/preprocessed",
        output_dir: str | Path = "experiments/ret_tracking/results",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load manifest and RD data
        self.manifest = pd.read_csv(self.data_dir / "01_rd_manifest.csv")
        self.n_frames = len(self.manifest)
        self.rd_images = np.load(self.data_dir / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
        self.valid_masks = np.load(self.data_dir / "north_up_pair_valid_masks.npy", mmap_mode="r")

        # 2. Solar geometry from WCS
        hdr_str = (self.data_dir / "target_wcs_header.txt").read_text(encoding="utf-8")
        self.wcs = WCS(fits.Header.fromstring(hdr_str, sep="\n"))
        sx, sy = self.wcs.world_to_pixel_values(0.0, 0.0)
        self.sx = float(sx)
        self.sy = float(sy)

        self.grid = PolarGridGeometry(sx=self.sx, sy=self.sy)
        self.timestamps = [parse_timestamp(t) for t in self.manifest["end_time"]]

        # Precompute standardized smoothed z-score maps
        print(f"Precomputing z-scores for {self.n_frames} frames (solar center: {self.sx:.2f}, {self.sy:.2f})...")
        self.z_maps = self._precompute_z_maps()

    def _precompute_z_maps(self, gaussian_sigma: float = 1.0) -> np.ndarray:
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
        cfg: RETConfig,
        run_name: str,
        collect_frame_results: bool = False,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, List[FrameRETResult]]:
        tracker = RadialEvidenceTracker(config=cfg, grid=self.grid)

        t0 = time.time()
        frame_results: List[FrameRETResult] = []
        ret0_candidate_rows = []

        for idx in range(self.n_frames):
            res = tracker.step(
                frame_idx=idx,
                timestamp=self.timestamps[idx],
                z_map=self.z_maps[idx],
                valid_mask=self.valid_masks[idx],
            )
            if collect_frame_results:
                frame_results.append(res)

            if cfg.variant == "RET-0":
                for cand in res.evidence_result.candidates:
                    ret0_candidate_rows.append({
                        "frame_idx": idx,
                        "timestamp": self.timestamps[idx].isoformat(),
                        "candidate_id": cand.candidate_id,
                        "peak_pa_deg": cand.peak_pa_deg,
                        "weighted_cpa_deg": cand.weighted_cpa_deg,
                        "pa_width_deg": cand.pa_width_deg,
                        "r_front_px": cand.r_front_px,
                        "r_peak_px": cand.r_peak_px,
                        "total_lobe_evidence": cand.total_lobe_evidence,
                        "peak_evidence": cand.peak_evidence,
                        "prominence": cand.prominence,
                    })

        elapsed_sec = time.time() - t0

        if cfg.variant == "RET-0":
            df_ret0 = pd.DataFrame(ret0_candidate_rows)
            summary = {
                "run_name": run_name,
                "variant": cfg.variant,
                "sigma_s": cfg.sigma_s,
                "formulation": cfg.evidence_formulation,
                "min_confirm_hits": cfg.min_confirm_hits,
                "total_tracks_born": 0,
                "confirmed_tracks": 0,
                "tentative_tracks": 0,
                "total_candidates": len(df_ret0),
                "mean_lifespan_frames": 0.0,
                "max_lifespan_frames": 0,
                "mean_cpa_jitter_deg": 0.0,
                "mean_radial_velocity_px_hr": 0.0,
                "mean_outward_consistency": 0.0,
                "runtime_sec": round(elapsed_sec, 2),
            }
            return summary, df_ret0, frame_results

        all_tracks = tracker.get_all_tracks()
        confirmed_tracks = tracker.get_confirmed_tracks()
        tentative_tracks = [t for t in all_tracks if t.status == "tentative"]

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
            "min_confirm_hits": cfg.min_confirm_hits,
            "total_tracks_born": len(all_tracks),
            "confirmed_tracks": len(confirmed_tracks),
            "tentative_tracks": len(tentative_tracks),
            "total_candidates": 0,
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
    runner = RETPipelineRunner()
    results_dir = Path("experiments/ret_tracking/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # 1. Primary Benchmark: RET-0, RET-1, RET-2, RET-3
    primary_configs = [
        ("RET0_EvidenceOnly", RETConfig(variant="RET-0", sigma_s=2.0)),
        ("RET1_OutwardConfirm", RETConfig(variant="RET-1", sigma_s=2.0)),
        ("RET2_MotionPred", RETConfig(variant="RET-2", sigma_s=2.0)),
        ("RET3_MultiLobe", RETConfig(variant="RET-3", sigma_s=2.0)),
    ]

    print("\n=== Running Primary Architecture Benchmark (RET-0, RET-1, RET-2, RET-3) ===")
    for name, cfg in primary_configs:
        print(f"Running {name}...")
        summ, df_out, _ = runner.run_configuration(cfg, name)
        summary_rows.append(summ)
        df_out.to_csv(results_dir / f"tracks_{name}.csv", index=False)
        print(f"  -> Confirmed: {summ['confirmed_tracks']}, Born: {summ['total_tracks_born']}, Max Life: {summ['max_lifespan_frames']}, CPA Jitter: {summ['mean_cpa_jitter_deg']} deg")

    # 2. Gaussian Width Sensitivity Sweep: sigma_s in {0, 1, 2, 4, 6} px on RET-2
    print("\n=== Running Gaussian Scale Sensitivity Sweep (sigma_s in {0, 1, 2, 4, 6} px) ===")
    for s in [0.0, 1.0, 2.0, 4.0, 6.0]:
        name = f"RET2_sigma_{int(s)}px" if s.is_integer() else f"RET2_sigma_{s}px"
        cfg = RETConfig(variant="RET-2", sigma_s=s)
        summ, df_out, _ = runner.run_configuration(cfg, name)
        summary_rows.append(summ)
        df_out.to_csv(results_dir / f"tracks_{name}.csv", index=False)
        print(f"  -> sigma_s={s}: Confirmed={summ['confirmed_tracks']}, Born={summ['total_tracks_born']}, Jitter={summ['mean_cpa_jitter_deg']} deg")

    # 3. Evidence Formulation Sensitivity: Positive vs Compressed
    print("\n=== Running Evidence Formulation Sensitivity (Positive vs Compressed) ===")
    cfg_comp = RETConfig(variant="RET-2", sigma_s=2.0, evidence_formulation="compressed", min_prominence=1.5)
    summ, df_out, _ = runner.run_configuration(cfg_comp, "RET2_Compressed")
    summary_rows.append(summ)
    df_out.to_csv(results_dir / "tracks_RET2_Compressed.csv", index=False)
    print(f"  -> Compressed: Confirmed={summ['confirmed_tracks']}, Max Life={summ['max_lifespan_frames']}, Jitter={summ['mean_cpa_jitter_deg']} deg")

    # 4. Confirmation Persistence Sensitivity: N_confirm in {2, 3}
    print("\n=== Running Confirmation Persistence Sensitivity (N_confirm in {2, 3}) ===")
    cfg_conf3 = RETConfig(variant="RET-2", sigma_s=2.0, min_confirm_hits=3)
    summ, df_out, _ = runner.run_configuration(cfg_conf3, "RET2_ConfirmHits_3")
    summary_rows.append(summ)
    df_out.to_csv(results_dir / "tracks_RET2_ConfirmHits_3.csv", index=False)
    print(f"  -> N_confirm=3: Confirmed={summ['confirmed_tracks']}, Max Life={summ['max_lifespan_frames']}, Jitter={summ['mean_cpa_jitter_deg']} deg")

    # 5. Angular Smoothing Sensitivity: sigma_theta in {1.0, 2.0, 4.0} deg
    print("\n=== Running Angular Smoothing Sensitivity (sigma_theta in {1.0, 2.0, 4.0} deg) ===")
    for stheta in [1.0, 2.0, 4.0]:
        name = f"RET2_smooth_{int(stheta)}deg"
        cfg = RETConfig(variant="RET-2", sigma_s=2.0, angular_smooth_sigma_deg=stheta)
        summ, df_out, _ = runner.run_configuration(cfg, name)
        summary_rows.append(summ)
        print(f"  -> sigma_theta={stheta}deg: Confirmed={summ['confirmed_tracks']}, Jitter={summ['mean_cpa_jitter_deg']} deg")

    # Export overall summary
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "track_summary_table.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved overall experimental summary to {summary_csv}")
    print("\nSummary Table:")
    print(df_summary[["run_name", "confirmed_tracks", "total_tracks_born", "mean_lifespan_frames", "max_lifespan_frames", "mean_cpa_jitter_deg", "mean_radial_velocity_px_hr", "mean_outward_consistency"]])


if __name__ == "__main__":
    run_all_experiments()
