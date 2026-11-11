"""Unit tests for Experiment 07: A5b End-to-End CME Segmentation Pipeline & Compatibility Adapter."""

import unittest
import warnings
from pathlib import Path
import numpy as np

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v01_baseline_tracking import E1TrackingStage
from cme_pipeline.stages.tracking.v07_pa_width_weighted_cpa import PAWidthWeightedCPAAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output
from cme_pipeline.stages.tracking.a5b_pipeline_adapter import adapt_a5b_tracking_output


class TestA5bEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.cfg = PipelineConfig()
        cls.ctx = PipelineContext(cls.cfg)

    def test_a5b_adapter_contract_integrity(self):
        """Verify canonical adapter preserves component/track separation and populates required downstream fields."""
        stage = PAWidthWeightedCPAAdaptiveROITrackingStage()
        raw_out = stage.run(None, self.cfg, self.ctx)
        adapted_out = adapt_tracking_output(raw_out)

        df = adapted_out.track_states_df
        required_cols = ["birth_r_min_px", "radial_front", "pa_min", "pa_max", "leading_front_state"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Verify Track IDs and Component IDs are NOT conflated
        self.assertIsNotNone(adapted_out.track_ownership_masks)
        self.assertTrue(np.array_equal(raw_out.patch_component_labels, adapted_out.patch_component_labels))
        self.assertTrue(np.array_equal(raw_out.track_ownership_masks, adapted_out.track_ownership_masks))
        self.assertFalse(
            np.array_equal(adapted_out.patch_component_labels, adapted_out.track_ownership_masks),
            "patch_component_labels must store spatial components, distinct from track_ownership_masks"
        )

    def test_legacy_a5b_wrapper_backwards_compatibility(self):
        """Verify legacy adapt_a5b_tracking_output preserves Exp 07 runner backwards compatibility cleanly."""
        stage = PAWidthWeightedCPAAdaptiveROITrackingStage()
        raw_out = stage.run(None, self.cfg, self.ctx)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            legacy_out = adapt_a5b_tracking_output(raw_out)
        df = legacy_out.track_states_df
        self.assertTrue((df["component_id"] == df["track_id"]).all())
        self.assertTrue(np.array_equal(legacy_out.patch_component_labels, raw_out.track_ownership_masks))

    def test_e1_track_count_is_219(self):
        """Verify E1 track count across full 32 frames is canonical 219."""
        t_stage = E1TrackingStage()
        out = t_stage.run(None, self.cfg, self.ctx)
        self.assertEqual(out.total_tracks, 219, "E1 baseline must have 219 tracks across full 32 frames")

    def test_a5b_track_count_is_21_under_default_config(self):
        """Verify A5b track count across full 32 frames is 21 under default outward_gate=False (Exp 06 parity)."""
        stage = PAWidthWeightedCPAAdaptiveROITrackingStage()
        raw_out = stage.run(None, self.cfg, self.ctx)
        self.assertEqual(raw_out.total_tracks, 21, "A5b default outward_gate=False must converge to 21 tracks")

    def test_predicted_only_tracks_have_zero_patches(self):
        """Verify that tracks in PREDICTED_ONLY state contribute 0 observed ownership patches."""
        stage = PAWidthWeightedCPAAdaptiveROITrackingStage()
        raw_out = stage.run(None, self.cfg, self.ctx)
        own = raw_out.track_ownership_masks
        man_df = raw_out.manifest_df
        df = raw_out.track_states_df

        pred_rows = df[df["state"] == "PREDICTED_ONLY"]
        self.assertGreater(len(pred_rows), 0, "There must be PREDICTED_ONLY states in A5b")
        for _, r in pred_rows.iterrows():
            p_idx = int(r["pair_index"])
            tid = int(r["track_id"])
            f_idx = man_df.index[man_df["pair_index"] == p_idx][0]
            patches_count = int((own[f_idx] == tid).sum())
            self.assertEqual(patches_count, 0, f"Track {tid} in pair {p_idx} has {patches_count} owned patches while PREDICTED_ONLY")


if __name__ == "__main__":
    unittest.main()
