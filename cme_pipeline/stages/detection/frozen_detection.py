"""Frozen Preprocessing & Hierarchical Detection Input Provider (2010-12-12 Benchmark).

Isolates shared preprocessing, WCS geometry, Gaussian background estimation,
thresholds, and raw 8-connected hierarchical detection inputs so that E1, A3, A4,
A5b, and future tracking methods all receive the exact same frozen detection state
without redundant FITS/NPY reloads or E1 track table parsing.

Note: Does not touch 1M Preprocessing/Detection Survey or Global Activation QC code.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, label

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import DetectionOutput, GeometryContext
from cme_pipeline.utils.geometry import compute_wcs_rasters

_DETECTION_CACHE: Dict[Tuple[Any, ...], DetectionOutput] = {}
STRUCT_8 = np.ones((3, 3), bool)


def clear_frozen_detection_cache() -> None:
    """Clears in-memory frozen detection cache."""
    _DETECTION_CACHE.clear()


def load_frozen_detection_inputs(
    config: PipelineConfig,
    context: PipelineContext,
    use_memory_cache: bool = True,
) -> DetectionOutput:
    """Loads frozen hierarchical detection arrays and frame preprocessing products.

    Caches result in memory keyed by frozen preprocessing parameters so sequential
    comparisons across E1, A3, A4, and A5b share the exact same detection state.
    """
    repo_root = context.repo_root
    cache_key = (
        str(repo_root),
        int(config.date_str),
        int(config.tracking.patch_size),
        float(config.segmentation.occulter_r_min_px),
        float(config.segmentation.fov_r_max_px),
        float(config.recovery.gaussian_sigma),
        float(config.tracking.max_cadence_gap_sec),
    )

    if use_memory_cache and cache_key in _DETECTION_CACHE:
        return _DETECTION_CACHE[cache_key]

    val_root = repo_root / "validation/frozen/20101212"
    exp01_dir = val_root / "experiments/01_d7_baseline_segmentation"
    exp11_dir = val_root / "experiments/11_d7_patch_first_tracking_feasibility"

    manifest_path = exp01_dir / "results/00_manifest/01_rd_manifest.csv"
    wcs_path = exp01_dir / "results/00_manifest/target_wcs_header.txt"
    rd_path = exp01_dir / "results/science_arrays/north_up_RD_DNs.npy"
    val_path = exp01_dir / "results/science_arrays/north_up_pair_valid_masks.npy"
    th_path = exp01_dir / "results/tables/03_detector_thresholds.csv"
    grid4_path = exp11_dir / "results/science_arrays/patch_activation_grid4.npy"
    comp_lbl_path = exp11_dir / "results/science_arrays/patch_component_labels.npy"

    # 1. Manifest & WCS
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

    # 4. Hierarchical Detection Outputs (grid4 and pure 8-connected components)
    grid4_all = np.load(grid4_path)
    if comp_lbl_path.is_file():
        raw_comp_labels = np.load(comp_lbl_path)
    else:
        raw_comp_labels = np.zeros(grid4_all.shape, dtype=np.int32)
        for i in range(n_pairs):
            lbl_i, _ = label(grid4_all[i], structure=STRUCT_8)
            raw_comp_labels[i] = lbl_i

    dataset_metadata = {
        "dataset_id": f"{config.date_str}_STEREO_A_COR2_d7",
        "n_pairs": n_pairs,
        "pair_range": [int(df_manifest["pair_index"].min()), int(df_manifest["pair_index"].max())] if n_pairs > 0 else [],
        "total_active_patches": int(grid4_all.sum()),
        "sources": {
            "manifest": str(manifest_path),
            "rd_array": str(rd_path),
            "valid_masks": str(val_path),
            "thresholds": str(th_path),
            "detection_grid4": str(grid4_path),
            "detection_component_labels": str(comp_lbl_path),
        },
    }

    det_out = DetectionOutput(
        manifest_df=df_manifest,
        patch_activation_grid4=grid4_all,
        raw_component_labels=raw_comp_labels,
        frame_cache=frame_cache,
        geometry=geom,
        dataset_metadata=dataset_metadata,
    )

    if use_memory_cache:
        _DETECTION_CACHE[cache_key] = det_out

    return det_out

