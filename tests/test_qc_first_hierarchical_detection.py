"""Unit tests for the isolated QC-first hierarchical detector kernel (no benchmark datasets needed)."""

import unittest

import numpy as np

from cme_pipeline.stages.detection.qc_first_hierarchical_detection import (
    apply_hierarchical_gating,
    detect_qc_first_frame,
)


def frame_inputs():
    scores = {16: np.zeros((16, 16)), 8: np.zeros((32, 32)), 4: np.zeros((64, 64))}
    accepted = {scale: np.ones_like(grid, dtype=bool) for scale, grid in scores.items()}
    thresholds = {16: 1.0, 8: 1.0, 4: 1.0}
    return scores, accepted, thresholds


class TestQCFirstHierarchicalDetection(unittest.TestCase):
    def test_hierarchical_parent_gate(self):
        coarse = np.zeros((16, 16), dtype=bool)
        mid = np.ones((32, 32), dtype=bool)
        fine = np.ones((64, 64), dtype=bool)
        coarse[1, 1] = True
        _, h08, h04 = apply_hierarchical_gating(coarse, mid, fine)
        self.assertEqual(int(h08.sum()), 4)
        self.assertEqual(int(h04.sum()), 16)

    def test_p04_floor_only_and_strict_greater(self):
        scores, accepted, thresholds = frame_inputs()
        scores[16][0, 0] = 2.0
        scores[8][0:2, 0:2] = 2.0
        scores[4][:] = 2.0
        scores[4][0, 0] = 3.0
        result = detect_qc_first_frame(scores, accepted, thresholds)
        self.assertTrue(result.usable_detection)
        self.assertEqual(result.observation_status, "VALID")
        self.assertTrue(result.floor_bound_p04)
        self.assertEqual(result.effective_threshold_p04, 2.0)
        self.assertEqual(int(result.final_states[16].sum()), 1)
        self.assertEqual(int(result.final_states[8].sum()), 4)
        self.assertEqual(int(result.pre_hierarchical[4].sum()), 16)
        self.assertEqual(int(result.final_states[4].sum()), 1)
        self.assertEqual(int(result.component_labels_p04.max()), 1)

    def test_global_activation_unavailable_not_empty(self):
        scores, accepted, thresholds = frame_inputs()
        for scale in (16, 8, 4):
            scores[scale][:] = 3.0
        result = detect_qc_first_frame(scores, accepted, thresholds)
        self.assertEqual(result.observation_status, "GLOBAL_ACTIVATION")
        self.assertFalse(result.usable_detection)
        self.assertEqual(result.active_fraction_pre, 1.0)
        self.assertTrue(np.all(result.final_states[4] == -1))
        self.assertTrue(np.all(result.component_labels_p04 == -1))

    def test_warmup_and_history_not_valid_empty(self):
        scores, accepted, thresholds = frame_inputs()
        warmup = detect_qc_first_frame(scores, accepted, thresholds, input_status="WARMUP")
        self.assertEqual(warmup.observation_status, "WARMUP")
        self.assertTrue(np.all(warmup.final_states[4] == -1))
        thresholds[4] = float("nan")
        no_history = detect_qc_first_frame(scores, accepted, thresholds)
        self.assertEqual(no_history.observation_status, "NO_HISTORY")
        self.assertFalse(no_history.usable_detection)

    def test_no_valid_patches_is_missing(self):
        scores, accepted, thresholds = frame_inputs()
        accepted[4][:] = False
        result = detect_qc_first_frame(scores, accepted, thresholds)
        self.assertEqual(result.observation_status, "MISSING_DATA")
        self.assertTrue(np.all(result.final_states[4] == -1))

    def test_floor_never_adds_patches(self):
        scores, accepted, thresholds = frame_inputs()
        scores[16][:] = 2.0
        scores[8][:] = 2.0
        scores[4][:] = 2.0
        scores[4][0, 0] = 3.0
        scores[4][0, 1] = 0.0  # below threshold so QC fraction < 1.0
        result = detect_qc_first_frame(
            scores, accepted, thresholds, max_active_fraction_pre=1.0
        )
        self.assertTrue(result.usable_detection)
        self.assertTrue(np.all(
            result.final_states[4].astype(bool) <= result.pre_hierarchical[4]
        ))


if __name__ == "__main__":
    unittest.main()
