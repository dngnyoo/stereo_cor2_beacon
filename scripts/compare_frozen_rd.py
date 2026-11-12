#!/usr/bin/env python3
"""Diagnostic comparison of all common RD pairs with the 2010-12 frozen arrays.

Pair identity is defined by both start and end d7 FITS filenames; array rows
are never assumed to align. Frozen 'DN' and new 'DN/s' are deliberately NOT
claimed to share a physical unit. The fitted scalar is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pair_key(start, end):
    return (Path(str(start)).name, Path(str(end)).name)


def run(new_root: Path, repo: Path, output: Path):
    new_root, repo, output = new_root.resolve(), repo.resolve(), output.resolve()
    frozen = repo / "validation/frozen/20101212/experiments/01_d7_baseline_segmentation/results"
    old_manifest = pd.read_csv(frozen / "00_manifest/01_rd_manifest.csv")
    fresh = pd.read_csv(new_root / "preprocessed/01_rd_manifest.csv")
    old_keys = [pair_key(a, b) for a, b in zip(old_manifest["prev_file"], old_manifest["curr_file"])]
    new_keys = [pair_key(a, b) for a, b in zip(fresh["start_file"], fresh["end_file"])]
    if len(set(old_keys)) != len(old_keys) or len(set(new_keys)) != len(new_keys):
        raise ValueError("Duplicate pair identity in manifest")
    old_index = {k: i for i, k in enumerate(old_keys)}
    new_index = {k: i for i, k in enumerate(new_keys)}
    common = [k for k in old_keys if k in new_index]
    if not common:
        raise ValueError("No common pairs; verify frozen and new manifests")
    old_rd = np.load(frozen / "science_arrays/north_up_RD_DNs.npy", mmap_mode="r")
    old_mask = np.load(frozen / "science_arrays/north_up_pair_valid_masks.npy", mmap_mode="r")
    new_rd = np.load(new_root / "preprocessed/north_up_RD_DNs_per_second.npy", mmap_mode="r")
    new_mask = np.load(new_root / "preprocessed/north_up_pair_valid_masks.npy", mmap_mode="r")
    if len(old_rd) != len(old_keys) or len(new_rd) != len(new_keys):
        raise ValueError("Manifest row count does not equal RD array row count")
    if old_rd.shape[1:] != new_rd.shape[1:] or old_mask.shape != old_rd.shape or new_mask.shape != new_rd.shape:
        raise ValueError("RD array or mask shapes disagree")
    records = []
    for start, end in common:
        key = start, end
        i, j = old_index[key], new_index[key]
        a = np.asarray(old_mask[i], dtype=bool)
        b = np.asarray(new_mask[j], dtype=bool)
        intersection = a & b
        union = a | b
        x = np.asarray(new_rd[j], dtype=np.float64)[intersection]
        y = np.asarray(old_rd[i], dtype=np.float64)[intersection]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        xx = np.dot(x, x)
        slope = float(np.dot(x, y) / xx) if xx > 0 else float("nan")
        residual = y - slope * x if np.isfinite(slope) else np.array([])
        corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and np.std(x) and np.std(y) else float("nan")
        records.append({
            "start_file": start, "end_file": end, "end_date": end[:8],
            "frozen_row": i, "new_row": j,
            "mask_iou": float(intersection.sum() / union.sum()) if union.any() else 1.0,
            "mask_disagree_pixels": int(np.logical_xor(a, b).sum()),
            "common_valid_pixels": int(x.size),
            "rd_corr_on_common_valid": corr,
            "best_scalar_frozen_over_new_diagnostic_only": slope,
            "scalar_residual_rmse_diagnostic_only": float(np.sqrt(np.mean(residual ** 2))) if residual.size else float("nan"),
            "frozen_max_abs_common_valid": float(np.max(np.abs(y))) if y.size else float("nan"),
            "new_max_abs_common_valid": float(np.max(np.abs(x))) if x.size else float("nan"),
        })
    df = pd.DataFrame(records)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "per_pair_rd_comparison.csv", index=False)
    report = {
        "frozen_total_pairs": len(old_keys),
        "new_total_pairs": len(new_keys),
        "common_pairs": len(common),
        "frozen_only": len(set(old_keys) - set(new_keys)),
        "new_only": len(set(new_keys) - set(old_keys)),
        "mean_mask_iou": float(df["mask_iou"].mean()),
        "median_rd_correlation": float(df["rd_corr_on_common_valid"].median()),
        "median_best_scalar_diagnostic_only": float(df["best_scalar_frozen_over_new_diagnostic_only"].median()),
        "median_scalar_residual_rmse_diagnostic_only": float(df["scalar_residual_rmse_diagnostic_only"].median()),
        "worst_five_mask_pairs": df.nsmallest(5, "mask_iou")[["end_file", "mask_iou", "mask_disagree_pixels"]].to_dict("records"),
        "worst_five_rd_correlation_pairs": df.nsmallest(5, "rd_corr_on_common_valid")[["end_file", "rd_corr_on_common_valid", "scalar_residual_rmse_diagnostic_only"]].to_dict("records"),
        "warning": "Frozen array is named DN and new array DN/s. Best-fit slope and residual are diagnostic; do not claim physical-unit equivalence. New north-up WCS includes additional selected frames.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["frozen_only"] == 0 else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, default=Path("runs/dual_archive_20101212"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dual_archive_20101212/rd_comparison"))
    args = parser.parse_args()
    raise SystemExit(run(args.new_root, args.repo, args.output_dir))
