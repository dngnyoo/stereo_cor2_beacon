"""Unit tests for Science Array Bit-Exact Reproducibility."""

import unittest
import numpy as np
from pathlib import Path


class TestSegmentationReproducibility(unittest.TestCase):
    def test_recovered_patch_masks_reproducibility(self):
        hist_file = Path("validation_20101212/full_day_pipeline_validation/01_full_day_pipeline/results/science_arrays/recovered_patch_masks_pipeline_b_full_day.npy")
        run_file = Path("runs/baseline_rule_b/arrays/recovered_patch_masks_pipeline_b_full_day.npy")

        self.assertTrue(hist_file.is_file(), f"Historical file missing: {hist_file}")
        self.assertTrue(run_file.is_file(), f"Run file missing: {run_file}")

        arr_hist = np.load(hist_file)
        arr_run = np.load(run_file)

        self.assertEqual(int(arr_hist.sum()), 5174)
        self.assertEqual(int(arr_run.sum()), 5174)
        self.assertTrue(
            np.array_equal(arr_hist, arr_run),
            f"Bitwise mismatch in recovery arrays: {np.sum(arr_hist ^ arr_run)} differing patches"
        )

    def test_final_pixel_segmentation_reproducibility(self):
        hist_file = Path("validation_20101212/full_day_pipeline_validation/01_full_day_pipeline/results/science_arrays/final_pixel_segmentation_union.npy")
        run_file = Path("runs/baseline_rule_b/arrays/final_pixel_segmentation_union.npy")

        self.assertTrue(hist_file.is_file(), f"Historical file missing: {hist_file}")
        self.assertTrue(run_file.is_file(), f"Run file missing: {run_file}")

        arr_hist = np.load(hist_file)
        arr_run = np.load(run_file)

        self.assertEqual(int(arr_hist.sum()), 50665)
        self.assertEqual(int(arr_run.sum()), 50665)
        self.assertTrue(
            np.array_equal(arr_hist, arr_run),
            f"Bitwise mismatch in pixel segmentation arrays: {np.sum(arr_hist ^ arr_run)} differing pixels"
        )


if __name__ == "__main__":
    unittest.main()

