"""Run comparison and regression audit utility for CME Pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from cme_pipeline.utils.io import load_json


def compare_runs(
    run_dir_a: Union[str, Path],
    run_dir_b: Union[str, Path],
) -> Dict[str, Any]:
    """Compares outputs, metrics, and arrays between two pipeline runs."""
    p_a = Path(run_dir_a)
    p_b = Path(run_dir_b)

    results: Dict[str, Any] = {
        "run_a": p_a.name,
        "run_b": p_b.name,
        "manifest_a": load_json(p_a / "manifest.json") if (p_a / "manifest.json").is_file() else {},
        "manifest_b": load_json(p_b / "manifest.json") if (p_b / "manifest.json").is_file() else {},
        "array_comparisons": {},
        "metric_comparisons": {},
    }

    # Compare recovery masks
    rec_a_file = p_a / "arrays/recovered_patch_masks_pipeline_b_full_day.npy"
    rec_b_file = p_b / "arrays/recovered_patch_masks_pipeline_b_full_day.npy"
    if rec_a_file.is_file() and rec_b_file.is_file():
        arr_a = np.load(rec_a_file)
        arr_b = np.load(rec_b_file)
        results["array_comparisons"]["recovered_patch_masks"] = {
            "sum_a": int(arr_a.sum()),
            "sum_b": int(arr_b.sum()),
            "is_equal": bool(np.array_equal(arr_a, arr_b)),
            "diff_count": int(np.sum(arr_a ^ arr_b)),
        }

    # Compare segmentation masks
    seg_a_file = p_a / "arrays/final_pixel_segmentation_union.npy"
    seg_b_file = p_b / "arrays/final_pixel_segmentation_union.npy"
    if seg_a_file.is_file() and seg_b_file.is_file():
        arr_a = np.load(seg_a_file)
        arr_b = np.load(seg_b_file)
        results["array_comparisons"]["final_pixel_segmentation_union"] = {
            "sum_a": int(arr_a.sum()),
            "sum_b": int(arr_b.sum()),
            "is_equal": bool(np.array_equal(arr_a, arr_b)),
            "diff_count": int(np.sum(arr_a ^ arr_b)),
        }

    return results

