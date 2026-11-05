"""E1 1:1 Patch Tracking Stage (Baseline Gate OFF)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import GeometryContext, PipelineStage, TrackingOutput
from cme_pipeline.utils.geometry import compute_wcs_rasters
from cme_pipeline.utils.io import compute_dict_hash


class E1TrackingStage(PipelineStage):
    """Executes or loads baseline E1 1:1 Patch Tracking and frame preprocessing."""
    name: str = "tracking"
    version: str = "1.0.0"

    def compute_fingerprint(self, inputs: Any, config: PipelineConfig, context: PipelineContext) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "date_str": config.date_str,
            "method": config.tracking.method,
            "alpha": config.tracking.alpha,
            "patch_size": config.tracking.patch_size,
        }
        return compute_dict_hash(desc)

    def run(self, inputs: Any, config: PipelineConfig, context: PipelineContext) -> TrackingOutput:
        context.info("Running E1TrackingStage (2010-12-12 Baseline)...")

        repo_root = context.repo_root
        sample_dir = repo_root / "sample_data/benchmark_20101212"
        if sample_dir.exists():
            manifest_path = sample_dir / "manifest/01_rd_manifest.csv"
            wcs_path = sample_dir / "manifest/target_wcs_header.txt"
            rd_path = sample_dir / "science_arrays/north_up_RD_DNs.npy"
            val_path = sample_dir / "science_arrays/north_up_pair_valid_masks.npy"
            th_path = sample_dir / "tables/03_detector_thresholds.csv"
            lbls_path = sample_dir / "science_arrays/patch_component_labels.npy"
            grid4_path = sample_dir / "science_arrays/patch_activation_grid4.npy"
            states_path = sample_dir / "tables/03_patch_e1_track_states.csv"
        else:
            val_root = repo_root / "validation_20101212"
            exp01_dir = val_root / "experiments/01_d7_baseline_segmentation"
            exp11_dir = val_root / "experiments/11_d7_patch_first_tracking_feasibility"
            manifest_path = exp01_dir / "results/00_manifest/01_rd_manifest.csv"
            wcs_path = exp01_dir / "results/00_manifest/target_wcs_header.txt"
            rd_path = exp01_dir / "results/science_arrays/north_up_RD_DNs.npy"
            val_path = exp01_dir / "results/science_arrays/north_up_pair_valid_masks.npy"
            th_path = exp01_dir / "results/tables/03_detector_thresholds.csv"
            lbls_path = exp11_dir / "results/science_arrays/patch_component_labels.npy"
            grid4_path = exp11_dir / "results/science_arrays/patch_activation_grid4.npy"
            states_path = exp11_dir / "results/tables/03_patch_e1_track_states.csv"

        # 1. Manifest and WCS
        df_man_all = pd.read_csv(manifest_path)
        dec12_man = df_man_all[df_man_all["date_str"] == config.date_str].reset_index(drop=True)
        n_pairs = len(dec12_man)

        hdr_txt = wcs_path.read_text(encoding="utf-8")
        wcs = WCS(fits.Header.fromstring(hdr_txt, sep="\n"))

        # Canonical geometry parameters
        sx, sy = 126.618, 127.503
        rsun_px = 8.163
        occulter_inner_radius_px = 2.0 * rsun_px
        occulter_r_min_px = config.segmentation.occulter_r_min_px
        fov_r_max_px = config.segmentation.fov_r_max_px

        rasters = compute_wcs_rasters(wcs=wcs, center_xy=(sx, sy), patch_size=config.tracking.patch_size)

        geom = GeometryContext(
            sx=sx,
            sy=sy,
            rsun_px=rsun_px,
            occulter_inner_radius_px=occulter_inner_radius_px,
            occulter_r_min_px=occulter_r_min_px,
            fov_r_max_px=fov_r_max_px,
            pa_raster=rasters["pa_raster"],
            radius_raster=rasters["radius_raster"],
            patch_pa=rasters["patch_pa"],
            patch_rad=rasters["patch_rad"],
            patch_x=rasters["patch_x"],
            patch_y=rasters["patch_y"],
            wcs_header_str=hdr_txt,
        )

        # 2. Raw RD DNs and Valid Masks
        rd_all = np.load(rd_path, mmap_mode="r")
        val_all = np.load(val_path, mmap_mode="r")
        df_th = pd.read_csv(th_path)

        # 3. Build Preprocessed Frame Cache
        frame_cache = []
        manifest_rows = []
        prev_end_time = None

        for f_idx in range(n_pairs):
            p_row = dec12_man.iloc[f_idx]
            p_num = int(p_row["pair_index"])
            p_raw = p_num - 1
            st = p_row["start_time"]
            et = p_row["end_time"]
            dt_pair = float(p_row["delta_t_min"])
            r_th = df_th.iloc[p_raw]

            t_curr_dt = datetime.fromisoformat(et)
            gap_sec = 0.0
            gap_from_prev_min = 0.0
            is_large_gap = False

            if prev_end_time is not None:
                t_prev_dt = datetime.fromisoformat(prev_end_time)
                gap_sec = (t_curr_dt - t_prev_dt).total_seconds()
                gap_from_prev_min = gap_sec / 60.0
                is_large_gap = gap_sec > config.tracking.max_cadence_gap_sec

            prev_end_time = et

            rd = np.asarray(rd_all[p_raw], dtype=np.float32)
            val = np.asarray(val_all[p_raw], dtype=bool)

            num = gaussian_filter(np.where(val, rd, 0.0), config.recovery.gaussian_sigma, mode="constant", cval=0.0)
            den = gaussian_filter(val.astype(float), config.recovery.gaussian_sigma, mode="constant", cval=0.0)
            usable = val & (den > 0)
            smoothed = np.full(rd.shape, np.nan, dtype=np.float32)
            smoothed[usable] = num[usable] / den[usable]

            blocks_val = val.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3)
            frac_val = np.sum(blocks_val, axis=(2, 3)) / 16.0

            vals = smoothed[usable & np.isfinite(smoothed)]
            m_val, s_val = float(np.mean(vals)), float(np.std(vals))
            z_global = np.zeros_like(rd, dtype=np.float32)
            z_global[usable] = (smoothed[usable] - m_val) / s_val

            frame_cache.append({
                "pair_idx": p_num,
                "f_idx": f_idx,
                "timestamp": t_curr_dt,
                "rd": rd,
                "val": val,
                "usable": usable,
                "smoothed": smoothed,
                "frac_val": frac_val,
                "z_global": z_global,
                "r_th": r_th,
                "is_large_gap": is_large_gap,
                "gap_from_prev_min": gap_from_prev_min,
            })

            manifest_rows.append({
                "f_idx": f_idx,
                "pair_index": p_num,
                "rd_pair_id": p_row["rd_pair_id"],
                "start_time": st,
                "end_time": et,
                "delta_t_min": dt_pair,
                "gap_from_prev_min": gap_from_prev_min if f_idx > 0 else 0.0,
                "is_large_gap": is_large_gap,
                "warmup_or_eval": "EVALUATION",
            })

        df_manifest = pd.DataFrame(manifest_rows)

        # 4. Load E1 Tracking Baseline Outputs
        lbls_all = np.load(lbls_path)
        grid4_all = np.load(grid4_path)
        df_states_e1 = pd.read_csv(states_path)

        obs_all = df_states_e1[df_states_e1["state"] == "OBSERVED"].copy()
        birth_df = obs_all.sort_values(["track_id", "pair_index"]).groupby("track_id").first().reset_index()
        track_birth_rmin = {int(r["track_id"]): float(r["birth_r_min_px"]) for _, r in birth_df.iterrows()}
        anchor_tracks = sorted([tid for tid, rmin in track_birth_rmin.items() if rmin <= config.birth_gate.max_birth_radius_px])

        context.info(f"Loaded {n_pairs} frames, {len(birth_df)} E1 tracks (Gate OFF), {len(anchor_tracks)} solar root anchors.")

        # Save stage tables
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_manifest.to_csv(stage_dir / "frame_manifest.csv", index=False)
        df_manifest.to_csv(context.tables_dir / "01_full_day_frame_manifest.csv", index=False)

        return TrackingOutput(
            manifest_df=df_manifest,
            track_states_df=df_states_e1,
            patch_component_labels=lbls_all,
            patch_activation_grid4=grid4_all,
            frame_cache=frame_cache,
            geometry=geom,
            total_tracks=len(birth_df),
            anchor_tracks=anchor_tracks,
        )

