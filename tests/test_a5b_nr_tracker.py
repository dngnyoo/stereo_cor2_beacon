"""Integration and stage-level unit test for A5b-NR1 and A5b-NR2 Tracking Stages."""

from __future__ import annotations

import unittest
import numpy as np

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v11_redesigned_normalized_a5b import PAWidthWeightedCPARedesignedTrackingStage
from cme_pipeline.stages.tracking.v12_score_aware_redesigned_a5b import PAWidthWeightedCPAScoreAwareTrackingStage
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output


class TestA5bNR1TrackingStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PipelineConfig()
        cls.ctx = PipelineContext(cls.cfg)
        stage = PAWidthWeightedCPARedesignedTrackingStage()
        cls.raw_out = stage.run(None, cls.cfg, cls.ctx)
        cls.adapted_out = adapt_tracking_output(cls.raw_out)

    def test_stage_output_contract(self):
        self.assertGreater(self.raw_out.total_tracks, 0)
        self.assertEqual(self.raw_out.tracking_method, "a5b_nr1")
        self.assertIsNotNone(self.raw_out.track_ownership_masks)
        masks = self.raw_out.track_ownership_masks
        self.assertTrue(np.all(masks >= 0))
        # Ensure Component IDs and Track IDs are separate representations
        self.assertFalse(np.array_equal(self.raw_out.patch_component_labels, self.raw_out.track_ownership_masks))

    def test_diagnostics_df_contract_and_no_zero_collapse(self):
        df = getattr(self.raw_out, "normalized_candidate_scores_df", None)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

        expected_cols = [
            "frame_index", "pair_index", "timestamp", "track_id", "component_id",
            "direct_overlap_patches", "adjacent_overlap_patches",
            "raw_overlap", "raw_angular", "raw_kinematic", "raw_front_bonus",
            "C", "A", "K", "D", "has_spatial_contact",
            "candidate_score", "normalized_geometric_score", "raw_a5b_score",
            "candidate_eligibility_category", "candidate_rank",
            "selected_rejected_outcome", "final_ownership_outcome", "rejection_reason",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # CRITICAL TEST: Redesigned evidence terms must NEVER collapse to zero!
        self.assertTrue((df["C"] > 0.0).all(), "C contains zero or negative values")
        self.assertTrue((df["A"] > 0.0).all(), "A contains zero or negative values")
        self.assertTrue((df["K"] > 0.0).all(), "K contains zero or negative values")
        self.assertTrue((df["D"] > 0.0).all(), "D contains zero or negative values")

        self.assertTrue((df["C"] <= 1.0).all(), "C exceeds 1.0")
        self.assertTrue((df["A"] <= 1.0).all(), "A exceeds 1.0")
        self.assertTrue((df["K"] <= 1.0).all(), "K exceeds 1.0")
        self.assertTrue((df["D"] <= 1.0).all(), "D exceeds 1.0")

        # Multiplicative score must be strictly in (0, 1] - ZERO COLLAPSE ELIMINATED
        self.assertTrue((df["candidate_score"] > 0.0).all(), "candidate_score collapsed to zero!")
        self.assertTrue((df["candidate_score"] <= 1.0).all(), "candidate_score exceeds 1.0")

        self.assertTrue((df["candidate_rank"] >= 1).all())
        self.assertTrue(set(df["selected_rejected_outcome"].unique()).issubset({"ACCEPTED", "REJECTED"}))
        self.assertTrue(set(df["final_ownership_outcome"].unique()).issubset({
            "EXCLUSIVE", "PARTITIONED", "LOST_CONTEST", "UNCLAIMED_COLLISION", "REJECTED"
        }))


class TestA5bNR2TrackingStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PipelineConfig()
        cls.cfg.tracking.method = "a5b_nr2"
        cls.ctx = PipelineContext(cls.cfg)
        stage = PAWidthWeightedCPAScoreAwareTrackingStage()
        cls.raw_out = stage.run(None, cls.cfg, cls.ctx)
        cls.adapted_out = adapt_tracking_output(cls.raw_out)

    def test_stage_output_contract(self):
        self.assertGreater(self.raw_out.total_tracks, 0)
        self.assertEqual(self.raw_out.tracking_method, "a5b_nr2")
        self.assertIsNotNone(self.raw_out.track_ownership_masks)
        masks = self.raw_out.track_ownership_masks
        self.assertTrue(np.all(masks >= 0))
        self.assertFalse(np.array_equal(self.raw_out.patch_component_labels, self.raw_out.track_ownership_masks))

    def test_diagnostics_df_contract_and_no_zero_collapse(self):
        df = getattr(self.raw_out, "normalized_candidate_scores_df", None)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

        self.assertTrue((df["C"] > 0.0).all())
        self.assertTrue((df["A"] > 0.0).all())
        self.assertTrue((df["K"] > 0.0).all())
        self.assertTrue((df["D"] > 0.0).all())
        self.assertTrue((df["candidate_score"] > 0.0).all())
        self.assertTrue((df["candidate_score"] <= 1.0).all())

    def test_score_aware_contested_allocations_recorded(self):
        df_contested = getattr(self.raw_out, "contested_allocation_df", None)
        self.assertIsNotNone(df_contested)
        if len(df_contested) > 0:
            self.assertIn("candidate_score", df_contested.columns)
            self.assertIn("score_beta", df_contested.columns)
            self.assertTrue((df_contested["method"] == "A5b_NR2_Score_Aware_PA_Width_CPA").all())


if __name__ == "__main__":
    unittest.main()
