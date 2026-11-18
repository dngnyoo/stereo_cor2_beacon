"""Contracts for restoring saved Identity Association without recomputation."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.core.run_layout import initialize_run
from cme_pipeline.standard_association_parent import restore_association, execute_from_association


class AssociationParentTests(unittest.TestCase):
    def setUp(self):
        self.states = pd.DataFrame([
            {"pair_index": 49, "track_id": 1, "state": "OBSERVED", "birth_r_min_px": 25.0},
            {"pair_index": 50, "track_id": 1, "state": "OBSERVED", "birth_r_min_px": 25.0},
            {"pair_index": 49, "track_id": 2, "state": "OBSERVED", "birth_r_min_px": 35.0},
        ])
        self.tracking = SimpleNamespace(track_states_df=self.states)
        self.membership = pd.DataFrame([
            {"pair_index": 49, "track_id": 1, "family_id": "FAM_T01",
             "is_solar_root_anchor": True, "membership_status": "IDENTITY_SEED",
             "link_type": "DIRECT_IDENTITY_MAPPING"},
            {"pair_index": 50, "track_id": 1, "family_id": "FAM_T01",
             "is_solar_root_anchor": True, "membership_status": "CONTINUING_MEMBER",
             "link_type": "DIRECT_IDENTITY_MAPPING"},
            {"pair_index": 49, "track_id": 2, "family_id": "FAM_T02",
             "is_solar_root_anchor": False, "membership_status": "IDENTITY_SEED",
             "link_type": "DIRECT_IDENTITY_MAPPING"},
        ])
        self.edges = pd.DataFrame(columns=["parent_track_id", "child_track_id"])

    def test_restore_identity_and_new_birth_threshold(self):
        assoc = restore_association(self.membership, self.edges, self.tracking,
                                    birth_radius_px=30.0)
        self.assertEqual(assoc.candidate_family_count, 2)
        self.assertEqual(assoc.family_anchors, {"FAM_T01": [1]})
        changed = restore_association(self.membership, self.edges, self.tracking,
                                      birth_radius_px=40.0)
        self.assertEqual(changed.family_anchors, {"FAM_T01": [1], "FAM_T02": [2]})
        self.assertEqual(changed.family_members, assoc.family_members)

    def test_missing_member_rejected(self):
        with self.assertRaises(ValueError):
            restore_association(self.membership.iloc[:-1], self.edges, self.tracking,
                                birth_radius_px=30.0)

    def test_inconsistent_family_rejected(self):
        corrupt = self.membership.copy()
        corrupt.loc[1, "family_id"] = "FAM_T99"
        with self.assertRaises(ValueError):
            restore_association(corrupt, self.edges, self.tracking, birth_radius_px=30.0)

    def test_parent_stage_not_completed_does_not_create_child(self):
        with TemporaryDirectory() as temp:
            config_file = Path(temp) / "config.yaml"
            config_file.write_text('experiment:\n  date: "20240503"\n')
            config = StandardConfig.load(config_file)
            parent = Path(temp) / "parent"
            initialize_run(parent, config.data, "synthetic", {})
            child = Path(temp) / "child"
            with self.assertRaises(ValueError):
                execute_from_association(config, input_run=parent, until="gates",
                                         run_dir=child)
            self.assertFalse(child.exists())


if __name__ == "__main__":
    unittest.main()
