"""Unit and regression tests for A4 Multi-Component Adaptive PA-ROI Tracking and A3 Bug Fix."""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v04_component_first import ComponentFirstAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v05_multi_component import MultiComponentAdaptiveROITrackingStage


class TestA4Tracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PipelineConfig()
        cls.ctx = PipelineContext(cls.cfg)
        cls.a3_stage = ComponentFirstAdaptiveROITrackingStage()
        cls.a3_out = cls.a3_stage.run(None, cls.cfg, cls.ctx)
        cls.a4_stage = MultiComponentAdaptiveROITrackingStage()
        cls.a4_out = cls.a4_stage.run(None, cls.cfg, cls.ctx)

    def test_a3_ownership_bug_fixed_regression(self):
        """Confirm that the T6 75-patch takeover bug at Pair 34 is completely eliminated in A3-fixed."""
        df = self.a3_out.track_states_df
        p34_t6 = df[(df["pair_index"] == 34) & (df["track_id"] == 6)]
        self.assertEqual(len(p34_t6), 1, "T6 state row missing at Pair 34")
        t6_row = p34_t6.iloc[0]

        # In buggy A3, T6 incorrectly had num_patches == 75 and state == OBSERVED
        # In fixed A3, T6 has num_patches == 0 and state == PREDICTED_ONLY
        self.assertEqual(t6_row["state"], "PREDICTED_ONLY", f"Expected PREDICTED_ONLY for T6 at Pair 34, got {t6_row['state']}")
        self.assertEqual(t6_row["num_patches"], 0, f"Expected 0 patches for T6 at Pair 34, got {t6_row['num_patches']}")
        self.assertLessEqual(t6_row["front_radius"], 30.0, f"Expected front_radius <= 30 px for T6, got {t6_row['front_radius']}")

        # Confirm T4 received its rightful streamer component
        p34_t4 = df[(df["pair_index"] == 34) & (df["track_id"] == 4)].iloc[0]
        self.assertEqual(p34_t4["state"], "OBSERVED")
        self.assertGreaterEqual(p34_t4["num_patches"], 40, f"T4 should have >= 40 patches, got {p34_t4['num_patches']}")

    def test_a3_unique_patch_ownership(self):
        """Verify that every active patch belongs to at most one track across all frames in A3-fixed."""
        masks = getattr(self.a3_out, "ownership_masks")
        n_frames = masks.shape[0]
        for f in range(n_frames):
            frame_mask = masks[f]
            grid = self.a3_out.patch_activation_grid4[f]
            self.assertTrue(np.all(frame_mask[~grid] == 0), f"Frame {f}: inactive patches assigned to tracks in A3-fixed!")

    def test_a4_unique_patch_ownership(self):
        """Verify that every active patch belongs to at most one track across all frames in A4."""
        masks = getattr(self.a4_out, "ownership_masks")
        n_frames = masks.shape[0]
        for f in range(n_frames):
            frame_mask = masks[f]
            grid = self.a4_out.patch_activation_grid4[f]
            self.assertTrue(np.all(frame_mask[~grid] == 0), f"Frame {f}: inactive patches assigned to tracks in A4!")

    def test_a4_multiple_disconnected_components_under_single_id(self):
        """Verify that A4 successfully retains multiple disconnected components under Track 3 at Pair 34."""
        df_track = self.a4_out.track_states_df
        df_comp = getattr(self.a4_out, "component_states_df")

        # Check T3 at Pair 34
        p34_t3 = df_track[(df_track["pair_index"] == 34) & (df_track["track_id"] == 3)]
        self.assertEqual(len(p34_t3), 1)
        t3_row = p34_t3.iloc[0]
        self.assertEqual(t3_row["state"], "OBSERVED")
        self.assertGreaterEqual(t3_row["num_components"], 2, f"Expected multiple components for T3 at P34, got {t3_row['num_components']}")
        self.assertGreaterEqual(t3_row["num_patches"], 90, f"Expected >= 90 patches for T3 at P34, got {t3_row['num_patches']}")

        # Verify component-level breakdown
        p34_t3_comps = df_comp[(df_comp["pair_index"] == 34) & (df_comp["track_id"] == 3)]
        self.assertGreaterEqual(len(p34_t3_comps), 2, "Expected multiple component records for T3 at Pair 34")

        # Verify that outer front and inner root both exist
        c_front = p34_t3_comps[p34_t3_comps["is_leading_front"] == True]
        self.assertEqual(len(c_front), 1, "Expected exactly 1 component flagged as leading front")
        self.assertGreater(c_front.iloc[0]["r_front"], 80.0, f"Expected outer front > 80 px, got {c_front.iloc[0]['r_front']}")

    def test_a4_leading_front_decoupling(self):
        """Verify leading front state separation in A4."""
        df_lf = getattr(self.a4_out, "leading_front_states_df")
        self.assertIn("leading_front_state", df_lf.columns)
        self.assertIn("responsible_comp_id", df_lf.columns)
        self.assertTrue(set(df_lf["leading_front_state"].unique()).issubset({"OBSERVED", "PREDICTED"}))

    def test_a4_determinism(self):
        """Verify deterministic output from identical inputs."""
        stage2 = MultiComponentAdaptiveROITrackingStage()
        out2 = stage2.run(None, self.cfg, self.ctx)

        pd.testing.assert_frame_equal(self.a4_out.track_states_df, out2.track_states_df)
        np.testing.assert_array_equal(getattr(self.a4_out, "ownership_masks"), getattr(out2, "ownership_masks"))


if __name__ == "__main__":
    unittest.main()
