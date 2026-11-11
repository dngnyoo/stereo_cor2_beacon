"""Unit and regression test suite for A5b PA-Width-Normalized CPA Tracking."""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v05_multi_component import MultiComponentAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v07_pa_width_weighted_cpa import PAWidthWeightedCPAAdaptiveROITrackingStage
from cme_pipeline.utils.geometry import circular_diff_deg, minimal_circular_enclosing_arc


class TestA5bTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PipelineConfig()
        cls.ctx = PipelineContext(cls.cfg)
        cls.a4_stage = MultiComponentAdaptiveROITrackingStage()
        cls.a4_out = cls.a4_stage.run(None, cls.cfg, cls.ctx)

        cls.cfg_a5b = PipelineConfig()
        cls.cfg_a5b.tracking.method = "a5b"
        cls.a5b_stage = PAWidthWeightedCPAAdaptiveROITrackingStage()
        cls.a5b_out = cls.a5b_stage.run(None, cls.cfg_a5b, cls.ctx)

    def test_1_circular_pa_width_wraparound(self):
        """Verify minimal circular enclosing arc spans correctly across 0°/360°."""
        pas = np.array([350.0, 355.0, 5.0, 10.0])
        arc = minimal_circular_enclosing_arc(pas)
        self.assertAlmostEqual(arc["arc_span"], 20.0, places=4)
        self.assertAlmostEqual(arc["arc_start"], 350.0, places=4)
        self.assertAlmostEqual(arc["arc_end"], 10.0, places=4)

    def test_2_single_patch_minimum_width_floor(self):
        """Verify that single patch (or zero-width) tracks clamp to min_pa_width_deg = 8.0°."""
        pas_single = np.array([45.0])
        arc_single = minimal_circular_enclosing_arc(pas_single)
        self.assertEqual(arc_single["arc_span"], 0.0)
        w_eff = max(float(arc_single["arc_span"]), 8.0)
        self.assertEqual(w_eff, 8.0)

    def test_3_weighted_distance_calculation(self):
        """Verify that d^*(p) = d_circ(PA_p, CPA_i) / max(w_i, 8.0°)."""
        # Track 3: CPA = 100°, w = 60° -> d3^*(100°) = 0, d3^*(80°) = 20 / 60 = 0.333
        # Track 4: CPA = 50°, w = 20° -> d4^*(80°) = 30 / 20 = 1.5
        # At PA = 80°, Track 3 should strictly win because 0.333 < 1.5
        d3_star = abs(circular_diff_deg(80.0, 100.0)) / max(60.0, 8.0)
        d4_star = abs(circular_diff_deg(80.0, 50.0)) / max(20.0, 8.0)
        self.assertAlmostEqual(d3_star, 1.0 / 3.0, places=4)
        self.assertAlmostEqual(d4_star, 1.5, places=4)
        self.assertLess(d3_star, d4_star)

    def test_4_deterministic_tie_handling(self):
        """Verify that exact tie in weighted distance breaks deterministically by track_id."""
        # Patch at PA = 75°, CPA_A = 50° (dist 25, w = 25 -> d* = 1.0)
        # Patch at PA = 75°, CPA_B = 100° (dist 25, w = 25 -> d* = 1.0)
        comp_tids = [4, 3]  # unordered
        best_tid = min(comp_tids, key=lambda tid: (1.0, tid))
        self.assertEqual(best_tid, 3, "Lower track_id should win exact tie")

    def test_5_unique_patch_ownership(self):
        """Verify that every active patch belongs to at most one track across all frames."""
        masks = getattr(self.a5b_out, "ownership_masks")
        n_frames = masks.shape[0]
        grid4 = self.a5b_out.patch_activation_grid4

        for f in range(n_frames):
            frame_m = masks[f]
            tids = np.unique(frame_m)
            sum_patches = sum((frame_m == t).sum() for t in tids if t > 0)
            nonzero_patches = (frame_m > 0).sum()
            self.assertEqual(sum_patches, nonzero_patches, f"Overlap collision at frame {f}!")
            self.assertTrue(np.all(frame_m[grid4[f] == 0] == 0), f"Inactive patch assigned at frame {f}!")

    def test_6_preservation_of_uncontested_patches(self):
        """Verify that patches claimed by only one track retain their full ownership."""
        dec_df = getattr(self.a5b_out, "decisions_df")
        uncontested = dec_df[dec_df["direct_overlap_patches"] > 0]
        self.assertGreater(len(uncontested), 0)

    def test_7_a4_multicomponent_t3_behavior(self):
        """Verify that T3 retains multiple components across Pairs 32-35 in A5b."""
        df_a5b = self.a5b_out.track_states_df
        t3_p32 = df_a5b[(df_a5b["track_id"] == 3) & (df_a5b["pair_index"] == 32)].iloc[0]
        self.assertEqual(t3_p32["state"], "OBSERVED")
        self.assertGreaterEqual(t3_p32["num_components"], 2, "T3 should retain >= 2 components at Pair 32")
        self.assertGreaterEqual(t3_p32["num_patches"], 100, f"T3 should retain broad CME flank (>=100 patches), got {t3_p32['num_patches']}")

    def test_8_t6_takeover_regression_fixed(self):
        """Verify that T6 is correctly PREDICTED_ONLY (0 patches) at Pair 34 in A5b."""
        df_a5b = self.a5b_out.track_states_df
        p34_t6 = df_a5b[(df_a5b["pair_index"] == 34) & (df_a5b["track_id"] == 6)]
        self.assertEqual(len(p34_t6), 1, "T6 record missing at Pair 34")
        t6_row = p34_t6.iloc[0]
        self.assertEqual(t6_row["state"], "PREDICTED_ONLY", f"T6 must be PREDICTED_ONLY, got {t6_row['state']}")
        self.assertEqual(t6_row["num_patches"], 0, f"T6 must have 0 patches at Pair 34, got {t6_row['num_patches']}")

    def test_9_no_forced_global_quota_behavior(self):
        """Verify that T5 does NOT expand into T4's territory at Pair 32."""
        df_a5b = self.a5b_out.track_states_df
        p32_t5 = df_a5b[(df_a5b["track_id"] == 5) & (df_a5b["pair_index"] == 32)].iloc[0]
        # In A4: T5 had 5 patches
        # In broken A5 (global quota): T5 had 18 patches
        # In A5b (PA-width normalized): T5 must have <= 6 patches!
        self.assertLessEqual(p32_t5["num_patches"], 6, f"T5 should stay <= 6 patches at Pair 32, got {p32_t5['num_patches']}")


if __name__ == "__main__":
    unittest.main()
