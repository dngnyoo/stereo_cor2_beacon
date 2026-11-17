"""Offline checks for optional, non-destructive pre-tracking size filtering."""
import unittest
import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.stage import DetectionOutput
from cme_pipeline.stages.gates.pre_tracking_component_size import filter_small_components


class TestPreTrackingComponentGate(unittest.TestCase):
    def test_keeps_two_patch_component_and_unavailable_sentinel(self):
        active = np.zeros((2, 64, 64), dtype=bool)
        active[0, 10, 10] = True
        active[0, 30, 30] = True
        active[0, 31, 31] = True  # 8-connected diagonal
        labels = np.zeros((2, 64, 64), dtype=np.int32)
        labels[0, 10, 10] = 1
        labels[0, 30, 30] = 2
        labels[0, 31, 31] = 2
        labels[1] = -1
        observed = DetectionOutput(
            manifest_df=pd.DataFrame({"usable_detection": [True, False]}),
            patch_activation_grid4=active,
            raw_component_labels=labels,
            frame_cache=[], geometry=None,
            dataset_metadata={"origin": "untouched"},
        )
        cleaned = filter_small_components(observed, min_component_patches=2)
        self.assertFalse(cleaned.patch_activation_grid4[0, 10, 10])
        self.assertTrue(cleaned.patch_activation_grid4[0, 30, 30])
        self.assertTrue(cleaned.patch_activation_grid4[0, 31, 31])
        self.assertTrue(np.all(cleaned.raw_component_labels[1] == -1))
        self.assertTrue(active[0, 10, 10])  # source immutability
        self.assertEqual(labels[0, 10, 10], 1)
        self.assertEqual(cleaned.dataset_metadata["pre_tracking_component_gate"]["removed_patches"], 1)

    def test_yaml_config_parsing(self):
        cfg = PipelineConfig.from_dict({
            "pre_size_gate": {
                "enabled": True, "connectivity": 8, "min_component_patches": 2
            },
            "size_gate": {"min_detected_patches": 10, "min_observed_pairs": 3},
        })
        self.assertTrue(cfg.pre_size_gate.enabled)
        self.assertEqual(cfg.size_gate.min_observed_pairs, 3)

    def test_reject_nonpositive_threshold(self):
        observed = DetectionOutput(
            manifest_df=pd.DataFrame({"usable_detection": [True]}),
            patch_activation_grid4=np.zeros((1, 64, 64), bool),
            raw_component_labels=np.zeros((1, 64, 64), np.int32),
            frame_cache=[], geometry=None,
        )
        with self.assertRaises(ValueError):
            filter_small_components(observed, min_component_patches=0)


if __name__ == "__main__":
    unittest.main()
