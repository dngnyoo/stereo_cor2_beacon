"""Isolated Stage 07: Recovery mask and immutable source preflight tests."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cme_pipeline.core.run_layout import initialize_run
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.standard_segmentation_parent import (
    validate_saved_masks, execute_from_recovery
)


class SegmentationParentTests(unittest.TestCase):
    def make_config(self, folder):
        file = Path(folder) / "cfg.yaml"
        file.write_text('experiment:\n  date: "20240503"\n'
                        'segmentation:\n  method: island_first\n')
        return StandardConfig.load(file)

    def test_preserves_two_family_overlap(self):
        first = np.zeros((2, 64, 64), dtype=bool)
        second = np.zeros_like(first)
        first[0, 3, 3] = True
        second[0, 3, 3] = True
        second[1, 4, 4] = True
        union = first | second
        self.assertEqual(
            validate_saved_masks(union, {
                "rec_FAM_T01": first, "rec_FAM_T02": second
            }, ["FAM_T01", "FAM_T02"]), 2
        )

    def test_corrupt_union_rejected(self):
        first = np.zeros((1, 64, 64), dtype=bool)
        first[0, 2, 3] = True
        with self.assertRaisesRegex(ValueError, "union"):
            validate_saved_masks(np.zeros_like(first), {"rec_FAM_T01": first},
                                 ["FAM_T01"])

    def test_missing_family_rejected(self):
        union = np.zeros((1, 64, 64), dtype=bool)
        with self.assertRaisesRegex(ValueError, "keys"):
            validate_saved_masks(union, {}, ["FAM_T01"])

    def test_wrong_shape_rejected(self):
        union = np.zeros((1, 64, 64), dtype=bool)
        with self.assertRaisesRegex(ValueError, "mask"):
            validate_saved_masks(union, {"rec_FAM_T01": np.zeros((1, 32, 32), dtype=bool)},
                                 ["FAM_T01"])

    def test_incomplete_parent_creates_no_child(self):
        with TemporaryDirectory() as temp:
            cfg = self.make_config(temp)
            parent = Path(temp) / "parent"
            initialize_run(parent, cfg.data, "synthetic", {})
            child = Path(temp) / "child"
            with self.assertRaises(ValueError):
                execute_from_recovery(cfg, input_run=parent, until="segmentation", run_dir=child)
            self.assertFalse(child.exists())

    def test_reject_other_until(self):
        with TemporaryDirectory() as temp:
            cfg = self.make_config(temp)
            with self.assertRaises(ValueError):
                execute_from_recovery(cfg, input_run=Path(temp) / "missing", until="recovery",
                                      run_dir=Path(temp) / "child")


if __name__ == "__main__":
    unittest.main()
