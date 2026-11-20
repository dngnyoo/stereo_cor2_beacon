"""Controlled component-scoring checks for experimental A5b-N."""
import math
import unittest

from cme_pipeline.stages.tracking.a5b_normalized_scoring import normalized_candidate_score


class TestA5bNormalizedScore(unittest.TestCase):
    def test_all_components_equal_one(self):
        score, terms = normalized_candidate_score(1, 1, 1, 122.5, 122.5,
                                                  has_spatial_contact=True)
        self.assertEqual(score, 1)
        self.assertTrue(all(x == 1 for x in terms.values()))

    def test_contact_product_and_clipping(self):
        score, terms = normalized_candidate_score(1.5, .8, .5, 61.25, 122.5,
                                                  has_spatial_contact=True)
        self.assertEqual(terms, {"O": 1., "A": .8, "K": .5, "F": .5})
        self.assertAlmostEqual(score, (.8 * .5 * .5) ** .25)

    def test_noncontact_does_not_collapse_on_zero_overlap(self):
        score, terms = normalized_candidate_score(0, 1, 1, 61.25, 122.5,
                                                  has_spatial_contact=False)
        self.assertEqual(terms["O"], 0.)
        self.assertAlmostEqual(score, .5 ** (1/3))

    def test_negative_and_excessive_terms_bounded(self):
        score, terms = normalized_candidate_score(-1, 5, 3, 400, 122.5,
                                                  has_spatial_contact=True)
        self.assertEqual(score, 0)
        self.assertTrue(all(0 <= t <= 1 for t in terms.values()))

    def test_invalid_radius_fails(self):
        with self.assertRaises(ValueError):
            normalized_candidate_score(1, 1, 1, 1, 0, has_spatial_contact=True)
        with self.assertRaises(ValueError):
            normalized_candidate_score(1, 1, 1, 1, -10.0, has_spatial_contact=True)
        with self.assertRaises(ValueError):
            normalized_candidate_score(1, 1, 1, 1, float("nan"), has_spatial_contact=True)

    def test_nan_and_inf_scores_raise(self):
        with self.assertRaises(ValueError):
            normalized_candidate_score(float("nan"), 1, 1, 50, 122.5, has_spatial_contact=True)
        with self.assertRaises(ValueError):
            normalized_candidate_score(1, float("inf"), 1, 50, 122.5, has_spatial_contact=True)
        with self.assertRaises(ValueError):
            normalized_candidate_score(1, 1, float("-inf"), 50, 122.5, has_spatial_contact=True)
        with self.assertRaises(ValueError):
            normalized_candidate_score(1, 1, 1, float("nan"), 122.5, has_spatial_contact=True)

    def test_equal_factor_interpretation(self):
        score, _ = normalized_candidate_score(.8, .8, .8, 98, 122.5,
                                               has_spatial_contact=True)
        self.assertAlmostEqual(score, .8)

    def test_zero_factor_vanishes_score(self):
        score, terms = normalized_candidate_score(0.0, 0.8, 0.9, 100.0, 122.5,
                                                  has_spatial_contact=True)
        self.assertEqual(score, 0.0)
        score_nc, terms_nc = normalized_candidate_score(0.5, 0.0, 0.9, 100.0, 122.5,
                                                        has_spatial_contact=False)
        self.assertEqual(score_nc, 0.0)

    def test_front_radius_bounds(self):
        # r_front at 0
        score_0, terms_0 = normalized_candidate_score(1.0, 1.0, 1.0, 0.0, 122.5,
                                                      has_spatial_contact=True)
        self.assertEqual(terms_0["F"], 0.0)
        self.assertEqual(score_0, 0.0)

        # r_front at max FOV
        score_max, terms_max = normalized_candidate_score(1.0, 1.0, 1.0, 122.5, 122.5,
                                                          has_spatial_contact=True)
        self.assertEqual(terms_max["F"], 1.0)
        self.assertEqual(score_max, 1.0)

        # r_front beyond max FOV clipped to 1.0
        score_beyond, terms_beyond = normalized_candidate_score(1.0, 1.0, 1.0, 150.0, 122.5,
                                                                has_spatial_contact=True)
        self.assertEqual(terms_beyond["F"], 1.0)
        self.assertEqual(score_beyond, 1.0)

    def test_score_strictly_in_unit_interval(self):
        import itertools
        values = [0.0, 0.1, 0.5, 0.9, 1.0, 1.5, -0.5]
        for o, a, k, r in itertools.product(values, values, values, [0, 50, 122.5, 200]):
            score_c, terms_c = normalized_candidate_score(o, a, k, r, 122.5, has_spatial_contact=True)
            self.assertTrue(0.0 <= score_c <= 1.0)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in terms_c.values()))

            score_nc, terms_nc = normalized_candidate_score(o, a, k, r, 122.5, has_spatial_contact=False)
            self.assertTrue(0.0 <= score_nc <= 1.0)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in terms_nc.values()))


class TestA5bNormalizedTrackingStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from cme_pipeline.core.config import PipelineConfig
        from cme_pipeline.core.context import PipelineContext
        from cme_pipeline.stages.tracking.v08_normalized_a5b import PAWidthWeightedCPANormalizedTrackingStage
        from cme_pipeline.stages.tracking.adapter import adapt_tracking_output

        cls.cfg = PipelineConfig()
        cls.ctx = PipelineContext(cls.cfg)
        stage = PAWidthWeightedCPANormalizedTrackingStage()
        cls.raw_out = stage.run(None, cls.cfg, cls.ctx)
        cls.adapted_out = adapt_tracking_output(cls.raw_out)

    def test_stage_output_contract(self):
        import numpy as np
        self.assertGreater(self.raw_out.total_tracks, 0)
        self.assertEqual(self.raw_out.tracking_method, "a5b_n")
        self.assertIsNotNone(self.raw_out.track_ownership_masks)
        masks = self.raw_out.track_ownership_masks
        self.assertTrue(np.all(masks >= 0))
        # Ensure Component IDs and Track IDs are separate
        self.assertFalse(np.array_equal(self.raw_out.patch_component_labels, self.raw_out.track_ownership_masks))

    def test_diagnostics_df_contract(self):
        df = getattr(self.raw_out, "normalized_candidate_scores_df", None)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        expected_cols = [
            "frame_index", "pair_index", "timestamp", "track_id", "component_id",
            "direct_overlap_patches", "adjacent_overlap_patches",
            "raw_overlap", "raw_angular", "raw_kinematic", "raw_front_bonus",
            "O", "A", "K", "F", "has_spatial_contact",
            "candidate_score", "normalized_geometric_score", "raw_a5b_score",
            "candidate_eligibility_category", "candidate_rank",
            "selected_rejected_outcome", "final_ownership_outcome", "rejection_reason",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        self.assertTrue((df["candidate_score"] >= 0.0).all() and (df["candidate_score"] <= 1.0).all())
        self.assertTrue((df["O"] >= 0.0).all() and (df["O"] <= 1.0).all())
        self.assertTrue((df["A"] >= 0.0).all() and (df["A"] <= 1.0).all())
        self.assertTrue((df["K"] >= 0.0).all() and (df["K"] <= 1.0).all())
        self.assertTrue((df["F"] >= 0.0).all() and (df["F"] <= 1.0).all())
        self.assertTrue((df["candidate_rank"] >= 1).all())
        self.assertTrue(set(df["selected_rejected_outcome"].unique()).issubset({"SELECTED", "REJECTED"}))
        self.assertTrue(set(df["final_ownership_outcome"].unique()).issubset({
            "EXCLUSIVE", "PARTITIONED", "LOST_CONTEST", "UNCLAIMED_COLLISION", "REJECTED"
        }))


if __name__ == "__main__":
    unittest.main()


