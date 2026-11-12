#!/usr/bin/env python3
"""Compare a freshly computed C0 run against frozen 2010-12-12 products.

Always align by UTC end_time (never by row index). Do not equate DN/s and
frozen DN without an independently validated physical-units contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _iso(t):
    return pd.to_datetime(t, utc=True)


def _match(new: pd.DataFrame, ref: pd.DataFrame, *, tolerance_s=.05):
    new = new.copy()
    ref = ref.copy()
    new["_time"] = _iso(new["end_time"])
    ref["_time"] = _iso(ref["end_time"])
    if ref["_time"].duplicated().any() or new["_time"].duplicated().any():
        raise ValueError("Duplicate UTC timestamps")
    idx = pd.merge_asof(
        ref.sort_values("_time").reset_index().rename(columns={"index": "ref_idx"}),
        new.sort_values("_time").reset_index().rename(columns={"index": "new_idx"}),
        on="_time", direction="nearest", tolerance=pd.Timedelta(seconds=tolerance_s),
        suffixes=("_ref", "_new")
    )
    return idx


def run(repo: Path, science: Path, detection: Path, output: Path):
    val = repo / "validation_20101212/experiments"
    frozen_pre = val / "01_d7_baseline_segmentation/results"
    frozen_det = val / "11_d7_patch_first_tracking_feasibility/results"
    ref_manifest = pd.read_csv(frozen_pre / "00_manifest/01_rd_manifest.csv")
    ref = ref_manifest[ref_manifest["date_str"] == 20101212].copy()
    new_manifest = pd.read_csv(science / "01_rd_manifest.csv")
    matched = _match(new_manifest, ref)
    paired = matched.dropna(subset=["new_idx"])
    errors = []
    if len(paired) != len(ref):
        errors.append(f"pair match count: matched={len(paired)} frozen={len(ref)}")
    if len(paired):
        delta = (pd.to_datetime(paired["start_time_ref"], utc=True) -
                 pd.to_datetime(paired["start_time_new"], utc=True)).dt.total_seconds().abs()
        if not (delta <= .05).all():
            errors.append(f"RD start times diverge for {int((delta > .05).sum())} pair(s)")

    ref_rd = np.load(frozen_pre / "science_arrays/north_up_RD_DNs.npy", mmap_mode="r")
    ref_val = np.load(frozen_pre / "science_arrays/north_up_pair_valid_masks.npy", mmap_mode="r")
    new_rd = np.load(science / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    new_val = np.load(science / "north_up_pair_valid_masks.npy", mmap_mode="r")
    ref_grid = np.load(frozen_det / "science_arrays/patch_activation_grid4.npy", mmap_mode="r")
    new_grid = np.load(detection / "patch_state_p04.npy", mmap_mode="r")
    new_detect = pd.read_csv(detection / "frame_detection_summary.csv")
    ref_th = pd.read_csv(frozen_pre / "tables/03_detector_thresholds.csv")

    if len(new_detect) != len(new_manifest):
        raise ValueError("New detector summary does not align with new RD manifest")
    if not (len(ref_rd) == len(ref_val) == len(ref_th)):
        raise ValueError("Frozen manifest/arrays are inconsistent")
    summary = []
    for row in paired.itertuples(index=False):
        i, j = int(row.ref_idx), int(row.new_idx)
        # Reference arrays cover both Dec 11 and Dec 12; reference row position
        # is global file position. Never use Dec12-only local index here.
        active = np.asarray(ref_val[i], bool) & np.asarray(new_val[j], bool)
        x, y = np.asarray(new_rd[j], float), np.asarray(ref_rd[i], float)
        finite = active & np.isfinite(x) & np.isfinite(y)
        xx, yy = x[finite], y[finite]
        slope = float(np.dot(xx, yy) / np.dot(xx, xx)) if xx.size and np.dot(xx, xx) else np.nan
        residual = yy - slope * xx if np.isfinite(slope) else np.array([])
        status = str(new_detect.iloc[j]["status"])
        if status == "VALID":
            compare = np.asarray(new_grid[j] == 1)
            frozen = np.asarray(ref_grid[i], bool)
            intersection = np.logical_and(compare, frozen).sum()
            union = np.logical_or(compare, frozen).sum()
            iou = float(intersection / union) if union else 1.0
            mismatch = int(np.logical_xor(compare, frozen).sum())
        else:
            iou, mismatch = np.nan, None
        summary.append({
            "utc_end": row._time.isoformat(), "frozen_pair_index": i + 1,
            "new_pair_index": j + 1, "status": status,
            "mask_valid_iou": (float((np.asarray(ref_val[i], bool) & np.asarray(new_val[j], bool)).sum() /
                                  (np.asarray(ref_val[i], bool) | np.asarray(new_val[j], bool)).sum())
                               if np.any(np.asarray(ref_val[i], bool) | np.asarray(new_val[j], bool)) else 1.),
            "rd_best_scalar_frozen_over_new": slope,
            "rd_scalar_residual_rmse": float(np.sqrt(np.mean(residual ** 2))) if residual.size else np.nan,
            "frozen_t24_p04": float(ref_th.iloc[i]["T24_p04"]) if pd.notna(ref_th.iloc[i]["T24_p04"]) else np.nan,
            "new_t24_p04": float(new_detect.iloc[j]["T24_p04"]) if pd.notna(new_detect.iloc[j]["T24_p04"]) else np.nan,
            "detection_patch_iou": iou, "detection_patch_mismatches": mismatch,
        })
    output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary)
    df.to_csv(output / "per_pair_comparison.csv", index=False)
    compared = df[df["status"] == "VALID"] if not df.empty else df
    report = {
        "frozen_dec12_pairs": len(ref),
        "matched_pairs": len(paired),
        "pair_matching_pass": len(paired) == len(ref) and not errors,
        "status": "COMPARED_NOT_CERTIFIED",
        "scientific_caveat": ("The new RD is DN/s and frozen RD is named DN. "
          "Best-fit scalar is a diagnostic, not proof of calibrated equivalence; "
          "mask mismatches do not isolate detector error unless preprocessing matches."),
        "valid_detection_pairs_compared": len(compared),
        "mean_detection_iou": float(compared["detection_patch_iou"].mean()) if len(compared) else None,
        "total_patch_mismatches": int(compared["detection_patch_mismatches"].sum()) if len(compared) else None,
        "errors": errors,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["pair_matching_pass"] else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--science-dir", type=Path, required=True)
    p.add_argument("--detection-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/c0_20101212_validation"))
    args = p.parse_args()
    raise SystemExit(run(args.repo.resolve(), args.science_dir.resolve(),
                         args.detection_dir.resolve(), args.output_dir.resolve()))
