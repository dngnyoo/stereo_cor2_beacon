"""Availability-aware adapter from C0/C3 RD products to Tracking DetectionOutput.

The loader preserves unusable observations separately. It NEVER turns C3 -1
into valid empty observations, and it rejects malformed status/mask pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from cme_pipeline.core.stage import DetectionOutput, GeometryContext
from cme_pipeline.utils.geometry import compute_wcs_rasters
from cme_pipeline.cadence_selection import parse_timestamp


def observation_usable(det_out: DetectionOutput, idx: int) -> bool:
    """A frozen DetectionOutput is fully observed; live output has status."""
    df = det_out.manifest_df
    if "usable_detection" in df.columns:
        value = df.iloc[idx]["usable_detection"]
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)
    return True


def load_live_tracking_input(
    detection_dir: str | Path,
    *, max_cadence_gap_sec: float = 4500.,
    gaussian_sigma: float = 1.,
    occulter_r_min_px: float = 22.,
    fov_r_max_px: float = 122.5,
) -> DetectionOutput:
    folder = Path(detection_dir).resolve()
    meta = json.loads((folder / "detection_metadata.json").read_text())
    science = Path(meta["input_dir"]).resolve()
    raw = np.load(folder / "patch_state_p04.npy", mmap_mode="r")
    components = np.load(folder / "patch_component_labels.npy", mmap_mode="r")
    rd = np.load(science / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    masks = np.load(science / "north_up_pair_valid_masks.npy", mmap_mode="r")
    detection_rows = pd.read_csv(folder / "frame_detection_summary.csv")
    manifest = pd.read_csv(science / "01_rd_manifest.csv")
    n = len(manifest)
    if not (raw.shape == components.shape == (n, 64, 64)
            and rd.shape == masks.shape == (n, 256, 256)
            and len(detection_rows) == n):
        raise ValueError("Detection, RD and manifest frames do not align")
    if not np.array_equal(manifest["pair_index"], detection_rows["pair_index"]):
        raise ValueError("Detection/manifest pair-index alignment mismatch")
    if not np.array_equal(manifest["end_time"].astype(str), detection_rows["end_time"].astype(str)):
        raise ValueError("Detection/manifest timestamps differ")

    valid_status = detection_rows["status"].eq("VALID").to_numpy()
    usable = detection_rows["usable_detection"].astype(str).str.lower().eq("true").to_numpy()
    if not np.array_equal(valid_status, usable):
        raise ValueError("Status/usable_detection mismatch")
    raw_array = np.asarray(raw)
    comp_array = np.asarray(components)
    if not np.all(np.isin(raw_array, [-1, 0, 1])):
        raise ValueError("Patch state must contain only -1, 0, 1")
    if np.any(raw_array[~usable] != -1) or np.any(comp_array[~usable] != -1):
        raise ValueError("Unavailable observations require -1 state and -1 components")
    if np.any(raw_array[usable] < 0) or np.any(comp_array[usable] < 0):
        raise ValueError("Valid observations may not contain unavailable sentinels")
    if np.any((comp_array[usable] > 0) != (raw_array[usable] == 1)):
        raise ValueError("Component labels and active patch grid disagree")

    hdr_str = (science / "target_wcs_header.txt").read_text(encoding="utf-8")
    wcs = WCS(fits.Header.fromstring(hdr_str, sep="\n"))
    sx, sy = wcs.world_to_pixel_values(0.0, 0.0)
    if not np.isfinite([sx, sy]).all():
        raise ValueError("Invalid north-up solar center")
    rasters = compute_wcs_rasters(wcs=wcs, center_xy=(sx, sy), patch_size=4)
    geom = GeometryContext(
        sx=float(sx), sy=float(sy),
        rsun_px=float("nan"),
        occulter_inner_radius_px=float("nan"),
        occulter_r_min_px=float(occulter_r_min_px),
        fov_r_max_px=float(fov_r_max_px),
        pa_raster=rasters["pa_raster"], radius_raster=rasters["radius_raster"],
        patch_pa=rasters["patch_pa"], patch_rad=rasters["patch_rad"],
        patch_x=rasters["patch_x"], patch_y=rasters["patch_y"],
        wcs_header_str=hdr_str,
    )

    out_manifest = manifest.copy()
    out_manifest["f_idx"] = np.arange(n)
    out_manifest["observation_status"] = detection_rows["status"].to_numpy()
    out_manifest["usable_detection"] = usable
    out_manifest["warmup_or_eval"] = np.where(
        detection_rows["in_requested_date"].astype(str).str.lower().eq("true"),
        "EVALUATION", "WARMUP"
    )
    out_manifest["gap_from_prev_min"] = 0.
    out_manifest["is_large_gap"] = False
    last_valid = None
    cache = []
    for idx in range(n):
        timestamp = parse_timestamp(manifest.iloc[idx]["end_time"])
        if last_valid is not None:
            gap_sec = (timestamp - last_valid).total_seconds()
            out_manifest.at[idx, "gap_from_prev_min"] = gap_sec / 60.
            out_manifest.at[idx, "is_large_gap"] = bool(gap_sec > max_cadence_gap_sec)
        # Intentional: unavailable detection does not reset valid-observation gap.
        if usable[idx]:
            last_valid = timestamp
        image = np.asarray(rd[idx], dtype=np.float32)
        val = np.asarray(masks[idx], dtype=bool)
        num = gaussian_filter(np.where(val, image, 0.), gaussian_sigma,
                              mode="constant", cval=0.)
        den = gaussian_filter(val.astype(float), gaussian_sigma,
                              mode="constant", cval=0.)
        smoothed = np.full(image.shape, np.nan, dtype=np.float32)
        good = val & (den > 0)
        smoothed[good] = (num[good] / den[good]).astype(np.float32)
        finite_values = smoothed[good & np.isfinite(smoothed)]
        z = np.zeros(image.shape, dtype=np.float32)
        if finite_values.size and float(finite_values.std()) > 0:
            z[good] = ((smoothed[good] - float(finite_values.mean()))
                       / float(finite_values.std()))
        fractions = val.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3).mean(axis=(2, 3))
        cache.append({
            "pair_idx": int(manifest.iloc[idx]["pair_index"]),
            "f_idx": idx, "timestamp": timestamp, "rd": image,
            "val": val, "usable": good, "smoothed": smoothed,
            "frac_val": fractions, "z_global": z,
            "r_th": detection_rows.iloc[idx].to_dict(),
            "is_large_gap": bool(out_manifest.iloc[idx]["is_large_gap"]),
            "gap_from_prev_min": float(out_manifest.iloc[idx]["gap_from_prev_min"]),
            "observation_status": str(detection_rows.iloc[idx]["status"]),
            "usable_detection": bool(usable[idx]),
        })
    active = np.asarray(raw == 1, dtype=bool)
    dataset_metadata = {
        "dataset_id": str(meta["requested_date"]) + "_STEREO_A_COR2_d7_live",
        "detector": meta["detector"], "denoiser": meta["denoiser"],
        "detection_dir": str(folder), "science_dir": str(science),
        "n_pairs": n, "n_unavailable": int((~usable).sum()),
        "observation_statuses": detection_rows["status"].value_counts().to_dict(),
        "not_validated_against_frozen_baseline": True,
    }
    return DetectionOutput(
        manifest_df=out_manifest, patch_activation_grid4=active,
        raw_component_labels=comp_array.copy(), frame_cache=cache,
        geometry=geom, dataset_metadata=dataset_metadata,
    )
