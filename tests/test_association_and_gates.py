"""Unit tests for Family Association and Gate rules."""

import unittest
import pandas as pd
from pathlib import Path

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.pipeline import Pipeline


class TestAssociationAndGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = PipelineConfig.from_yaml("configs/baseline_rule_b.yaml")
        cls.pipeline = Pipeline(cls.config)
        # Load from existing cached run
        cls.track_out = cls.pipeline.context.cache.load_output("tracking")
        cls.assoc_out = cls.pipeline.context.cache.load_output("association")
        cls.birth_out = cls.pipeline.context.cache.load_output("birth_gate")
        cls.size_out = cls.pipeline.context.cache.load_output("size_gate")

    def test_no_premature_anchor_filtering(self):
        # All 219 tracks must be present in association membership
        total_tracks = self.track_out.total_tracks
        self.assertEqual(total_tracks, 219)

        all_assoc_tracks = set()
        for members in self.assoc_out.family_members.values():
            all_assoc_tracks.update(members)
        self.assertEqual(len(all_assoc_tracks), 219, "All 219 tracks must be evaluated in Family Association")

    def test_no_unintended_family_merger(self):
        # In Rule B, when competing parents exist, auto-merger between parent families is strictly prevented.
        # Anchor tracks with competing parents are preserved as independent seeds.
        competing_rows = self.assoc_out.track_membership_df[
            self.assoc_out.track_membership_df["membership_status"] == "COMPETING_PARENT_BRANCH"
        ]
        self.assertGreater(len(competing_rows), 0, "Competing parent transitions must be tracked")
        for _, row in competing_rows.iterrows():
            decision = row["decision"]
            self.assertIn(
                decision,
                ["CONFIRMED_INDEPENDENT_SEED", "UNRESOLVED_COMPETING"],
                "Competing parents must never be auto-merged into a single family"
            )

    def test_family_level_birth_gate(self):
        # Birth gate must be applied at the family level
        # A family is admitted if anchor_count > 0
        df_summary = self.size_out.family_summary_df
        for _, row in df_summary.iterrows():
            fid = row["family_id"]
            n_anchors = row["anchor_count"]
            status = row["status"]
            reason = row["rejection_reason"]
            if n_anchors == 0:
                self.assertEqual(status, "REJECTED")
                self.assertEqual(reason, "NO_SOLAR_ROOT_ANCHOR")

    def test_family_level_size_gate(self):
        # Size gate must require max_detected_patches >= 5
        df_summary = self.size_out.family_summary_df
        admitted = df_summary[df_summary["status"] == "ADMITTED"]
        self.assertEqual(len(admitted), 15, "Baseline Rule B must admit exactly 15 families")
        for _, row in admitted.iterrows():
            self.assertGreaterEqual(row["max_detected_patches"], 5)
            self.assertGreater(row["anchor_count"], 0)


if __name__ == "__main__":
    unittest.main()
