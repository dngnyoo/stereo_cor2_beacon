"""Unit and regression tests for A5: PA-Width-Proportional Ownership Tracking."""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.multi_component_tracking import MultiComponentAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.pa_width_proportional_tracking import PAWidthProportionalAdaptiveROITrackingStage


class TestA5Tracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PipelineConfig()
        cls.ctx = PipelineContext(cls.cfg)
        cls.a4_stage = MultiComponentAdaptiveROITrackingStage()
        cls.a4_out = cls.a4_stage.run(None, cls.cfg, cls.ctx)

        cls.cfg_a5 = PipelineConfig()
        cls.cfg_a5.tracking.method = "a5"
        cls.a5_stage = PAWidthProportionalAdaptiveROITrackingStage()
        cls.a5_out = cls.a5_stage.run(None, cls.cfg_a5, cls.ctx)

    def test_a5_stage_metadata(self):
        """Verify A5 stage naming, versioning, and fingerprinting."""
        self.assertEqual(self.a5_stage.name, "tracking")
        self.assertEqual(self.a5_stage.version, "5.0.0")
        fp = self.a5_stage.compute_fingerprint(None, self.cfg_a5, self.ctx)
        self.assertIsInstance(fp, str)
        self.assertEqual(len(fp), 64)

    def test_a5_unique_patch_ownership(self):
        """Verify that every active patch belongs to at most one track across all frames."""
        masks = getattr(self.a5_out, "ownership_masks")
        n_frames = masks.shape[0]
        grid4 = self.a5_out.patch_activation_grid4

        for f in range(n_frames):
            frame_m = masks[f]
            tids = np.unique(frame_m)
            sum_patches = sum((frame_m == t).sum() for t in tids if t > 0)
            nonzero_patches = (frame_m > 0).sum()
            self.assertEqual(sum_patches, nonzero_patches, f"Overlap collision at frame {f}!")
            # Nonzero patches must be subset of active grid
            self.assertTrue(np.all(frame_m[grid4[f] == 0] == 0), f"Inactive patch assigned at frame {f}!")

    def test_contested_allocation_conservation(self):
        """Verify that every contested allocation conserves total component patches exactly."""
        df_alloc = getattr(self.a5_out, "contested_allocation_df")
        self.assertGreater(len(df_alloc), 0, "No contested allocations logged!")

        grouped = df_alloc.groupby(["pair_index", "comp_id"])
        for (p, cid), group in grouped:
            tot = group["total_patches"].iloc[0]
            allocated_sum = group["allocated_patches"].sum()
            target_sum = group["target_patches"].sum()
            self.assertEqual(allocated_sum, tot, f"Allocated sum mismatch at Pair {p}, Comp {cid}: {allocated_sum} != {tot}")
            self.assertEqual(target_sum, tot, f"Target sum mismatch at Pair {p}, Comp {cid}: {target_sum} != {tot}")

    def test_a4_vs_a5_t3_t4_t5_ownership_shift(self):
        """Verify and document quantitative shift between A4 and A5 at Pair 31."""
        df_a4 = self.a4_out.track_states_df
        df_a5 = self.a5_out.track_states_df

        p31_a4 = df_a4[df_a4["pair_index"] == 31].set_index("track_id")
        p31_a5 = df_a5[df_a5["pair_index"] == 31].set_index("track_id")

        # In Pair 31:
        # A4: T3=67, T4=22, T5=6
        # A5: T3=64, T4=24, T5=7
        self.assertEqual(p31_a4.loc[3, "num_patches"], 67)
        self.assertEqual(p31_a5.loc[3, "num_patches"], 64)
        self.assertEqual(p31_a4.loc[4, "num_patches"], 22)
        self.assertEqual(p31_a5.loc[4, "num_patches"], 24)
        self.assertEqual(p31_a4.loc[5, "num_patches"], 6)
        self.assertEqual(p31_a5.loc[5, "num_patches"], 7)


if __name__ == "__main__":
    unittest.main()
