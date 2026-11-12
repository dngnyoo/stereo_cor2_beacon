"""C0 baseline and C3 QC-first detectors for North-up No-MSB COR2 RD.

C0 lineage: segmentation_pipeline/05_run_dynamicT_detection.py.
C3 lineage: one_month_preprocessing_detection_survey/
  09_frame_median_floor_qc/c3_qc_first_p04_floor.
C3 is the intended default; historical frozen detection is unaffected.
"""
from __future__ import annotations

from collections import deque
from datetime import timedelta
from pathlib import Path
import csv
import json
import warnings

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, label, median_filter

from cme_pipeline.cadence_selection import parse_timestamp

SCALES = (16, 8, 4)
SHAPES = {16: (16, 16), 8: (32, 32), 4: (64, 64)}
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def _scores(z: np.ndarray, valid: np.ndarray, size: int):
    n = 256 // size
    if z.shape != (256, 256) or valid.shape != z.shape:
        raise ValueError("Detection input must have 256x256 RD and valid masks")
    blocks = np.where(valid, np.abs(z), np.nan).reshape(
        n, size, n, size).transpose(0, 2, 1, 3).reshape(n, n, size * size)
    fractions = valid.reshape(n, size, n, size).mean(axis=(1, 3))
    accepted = (fractions >= .5) & (np.isfinite(blocks).sum(axis=2) > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p95 = np.nanpercentile(blocks, 95., axis=2)
    return p95, accepted & np.isfinite(p95)


def _smooth_and_z(rd: np.ndarray, valid: np.ndarray, *, denoiser: str):
    if denoiser == "gaussian":
        numerator = gaussian_filter(np.where(valid, rd, 0.), 1., mode="constant", cval=0.)
        denominator = gaussian_filter(valid.astype(float), 1., mode="constant", cval=0.)
        smoothed = np.full(rd.shape, np.nan, dtype=np.float64)
        usable = valid & (denominator > 0)
        smoothed[usable] = numerator[usable] / denominator[usable]
    elif denoiser == "median":
        # Legacy median implementation used nanmedian 3x3 windows.
        from numpy.lib.stride_tricks import sliding_window_view
        source = np.where(valid, rd, np.nan)
        padded = np.pad(source, 1, mode="constant", constant_values=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            smoothed = np.nanmedian(sliding_window_view(padded, (3, 3)), axis=(-2, -1))
        smoothed[~valid] = np.nan
        usable = valid & np.isfinite(smoothed)
    else:
        raise ValueError("denoiser must be gaussian or median")
    values = smoothed[usable & np.isfinite(smoothed)]
    if not len(values) or not np.isfinite(values).all() or float(np.std(values)) <= 0:
        return None, smoothed, float("nan"), float("nan")
    mean, std = float(values.mean()), float(values.std())
    z = np.full(rd.shape, np.nan)
    z[usable] = (smoothed[usable] - mean) / std
    return z, smoothed, mean, std


def _threshold(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    m = float(np.median(values))
    mad = float(np.median(np.abs(values - m)))
    return m + 3.0 * 1.4826 * mad


def _expand_neighbors(grid: np.ndarray) -> np.ndarray:
    from scipy.ndimage import maximum_filter
    return maximum_filter(grid.astype(np.uint8), size=3, mode="constant").astype(bool)


def _c0_hierarchy(flat: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    # Historical Exp39 neighbor-margin=1 hierarchy (not C3 pure gating).
    p16 = flat[16].copy()
    allowed8 = np.repeat(np.repeat(_expand_neighbors(p16), 2, axis=0), 2, axis=1)
    p08 = flat[8] & allowed8
    allowed4 = np.repeat(np.repeat(_expand_neighbors(p08), 2, axis=0), 2, axis=1)
    return {16: p16, 8: p08, 4: flat[4] & allowed4}


def _pure_hierarchy(flat: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    p16 = flat[16].copy()
    p08 = flat[8] & np.repeat(np.repeat(p16, 2, axis=0), 2, axis=1)
    p04 = flat[4] & np.repeat(np.repeat(p08, 2, axis=0), 2, axis=1)
    return {16: p16, 8: p08, 4: p04}


def run_detection(preprocessed_dir: str | Path, *, method: str = "c3",
                  denoiser: str = "gaussian", requested_date: str | None = None,
                  qc_max_active_fraction: float = .70,
                  min_floor_patches: int = 3,\n                  output_dir: str | Path | None = None) -> Path:
    """Compute C0 or C3 on all available RD, save explicit detection statuses.

    Historical C0 and C3 differ in parent hierarchy. Both share P95/T24 and
    process frames chronologically. Only previous frames enter T24.
    """
    if method not in {"c0", "c3"}:
        raise ValueError("Detector must be 'c0' or 'c3'")
    if not 0 < qc_max_active_fraction <= 1:
        raise ValueError("QC active fraction must be within (0, 1]")
    if min_floor_patches < 1:
        raise ValueError("min_floor_patches must be positive")
    folder = Path(preprocessed_dir).resolve()
    metadata = json.loads((folder / "preprocessing_metadata.json").read_text())
    if metadata.get("unit") != "DN/s" or metadata.get("msb_applied") is not False:
        raise ValueError("Expected validated No-MSB DN/s preprocessing")
    with (folder / "01_rd_manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    rd = np.load(folder / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(folder / "north_up_pair_valid_masks.npy", mmap_mode="r")
    if rd.shape != valid.shape or rd.ndim != 3 or rd.shape[1:] != (256, 256):
        raise ValueError("Incorrect RD or validity shape")
    if len(manifest) != len(rd):
        raise ValueError("RD manifest and image count mismatch")
    if requested_date is None:
        requested_date = str(metadata.get("requested_date", manifest[-1]["date_str"]))
    out = Path(output_dir).resolve() if output_dir is not None else (\n        folder.parent / f"detection_{method}_{denoiser}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite detector results: {out}")

    histories = {size: deque() for size in SCALES}
    rows, states, labels_list = [], [], []
    first_time = parse_timestamp(manifest[0]["end_time"])
    last_time = None
    for idx, item in enumerate(manifest):
        current = parse_timestamp(item["end_time"])
        if last_time is not None and current <= last_time:
            raise ValueError("RD manifest must be strictly chronological")
        last_time = current
        usable = np.asarray(valid[idx], dtype=bool)
        image = np.asarray(rd[idx], dtype=np.float64)
        has_data = bool(np.any(usable) and np.isfinite(image[usable]).all())
        z = None
        if has_data:
            z, _, mean, std = _smooth_and_z(image, usable, denoiser=denoiser)
            has_data = z is not None
        else:
            mean = std = float("nan")
        scores = {}
        accepted = {}
        for size in SCALES:
            if has_data:
                scores[size], accepted[size] = _scores(z, usable, size)
            else:
                scores[size] = np.full(SHAPES[size], np.nan)
                accepted[size] = np.zeros(SHAPES[size], dtype=bool)

        thresholds = {}
        has_history = True
        lower = current - timedelta(hours=24)
        for size in SCALES:
            hist = histories[size]
            while hist and hist[0][0] < lower:
                hist.popleft()
            pooled = np.concatenate([values for _, values in hist]) if hist else np.array([])
            thresholds[size] = _threshold(pooled)
            has_history &= bool(len(pooled) > 0 and np.isfinite(thresholds[size]))

        # Status reflects observation validity; history is updated afterwards.
        if not has_data or not np.any(accepted[4]):
            status = "MISSING_DATA"
        elif (method == "c0" and metadata.get("frozen_reference_pairs")
              and (current.strftime("%Y%m%d") == "20101211"
                   or (current - first_time).total_seconds() / 3600. < 23.9)):
            status = "WARMUP"
        elif (not (method == "c0" and metadata.get("frozen_reference_pairs"))
              and (current - first_time).total_seconds() < 24 * 3600):
            status = "WARMUP"
        elif not has_history:
            status = "NO_HISTORY"
        else:
            status = "VALID"

        flat = {
            size: accepted[size] & (scores[size] > thresholds[size])
            if status == "VALID" else np.zeros(SHAPES[size], bool)
            for size in SCALES
        }
        pre = _pure_hierarchy(flat) if method == "c3" else _c0_hierarchy(flat)
        denom = int(accepted[4].sum())
        active_fraction_pre = float((pre[4] & accepted[4]).sum() / denom) if denom else float("nan")
        floor = effective = float("nan")
        floor_bound = False
        if method == "c3" and status == "VALID":
            if active_fraction_pre >= qc_max_active_fraction:
                status = "GLOBAL_ACTIVATION"
            else:
                valid_scores = scores[4][accepted[4]]
                effective = float(thresholds[4])
                if len(valid_scores) >= min_floor_patches:
                    floor = float(np.median(valid_scores))
                    effective = max(effective, floor)
                    floor_bound = bool(floor > thresholds[4])
                pre[4] = accepted[4] & (scores[4] > effective) & np.repeat(
                    np.repeat(pre[8], 2, axis=0), 2, axis=1
                )
        if status == "VALID":
            state = pre[4].astype(np.int8)
            components, _ = label(state.astype(bool), structure=CONNECTIVITY_8)
            components = components.astype(np.int32)
        else:
            state = np.full((64, 64), -1, dtype=np.int8)
            components = np.full((64, 64), -1, dtype=np.int32)
        states.append(state)
        labels_list.append(components)
        rows.append({
            "pair_index": item["pair_index"], "end_time": item["end_time"],
            "analysis_date": requested_date,
            "in_requested_date": current.strftime("%Y%m%d") == requested_date,
            "status": status, "usable_detection": status == "VALID",
            "denoiser": denoiser, "detector": method,
            "active_fraction_pre": active_fraction_pre,
            "T24_p16": thresholds[16], "T24_p08": thresholds[8],
            "T24_p04": thresholds[4], "floor_p04": floor,
            "effective_threshold_p04": effective, "floor_bound_p04": floor_bound,
            "frame_z_mean": mean, "frame_z_std": std,
            "n_active_p04": int(state.sum()) if status == "VALID" else "",
        })
        # Causal raw score history: never use adjusted floor/QC masks.
        if has_data:
            for size in SCALES:
                values = scores[size][accepted[size]]
                if len(values):
                    histories[size].append((current, values.copy()))
    out.mkdir(parents=False)
    np.save(out / "patch_state_p04.npy", np.stack(states))
    np.save(out / "patch_component_labels.npy", np.stack(labels_list))
    with (out / "frame_detection_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "detection_metadata.json").write_text(json.dumps({
        "detector": method, "denoiser": denoiser,
        "requested_date": requested_date,
        "input_dir": str(folder),
        "frames": len(rows),
        "valid_requested_frames": sum(r["status"] == "VALID" and r["in_requested_date"] for r in rows),
        "warmup_frames": sum(r["status"] == "WARMUP" for r in rows),
        "qc_max_active_fraction": qc_max_active_fraction if method == "c3" else None,
        "history_rule": "strictly earlier observations within trailing 24h",
        "status": "DETECTION_READY_TRACKING_NOT_RUN",
    }, indent=2), encoding="utf-8")
    return out
