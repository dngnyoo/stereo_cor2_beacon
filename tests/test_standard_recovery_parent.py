"""Recovery-only adapter guard tests; no real CME data needed."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd

from cme_pipeline.core.run_layout import initialize_run
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.standard_recovery_parent import (
    execute_from_gates, validate_saved_gate_decisions,
)


class RecoveryParentTests(unittest.TestCase):
    def make_config(self, root, *, with_radius=True):
        file = Path(root) / "config.yaml"
        recovery = "  inner_radius_px: 16.326\n" if with_radius else ""
        file.write_text(
            'experiment:\n  date: "20240503"\n'
            'recovery:\n  method: signed_dynamic_pa\n' + recovery
        )
        return StandardConfig.load(file)

    @staticmethod
    def association():
        return SimpleNamespace(family_members={"FAM_T01": [1], "FAM_T02": [2]})

    def test_saved_gate_ids_are_authoritative(self):
        summary = pd.DataFrame([
            {"family_id": "FAM_T01", "status": "ADMITTED", "anchor_count": 1,
             "max_detected_patches": 20},
            {"family_id": "FAM_T02", "status": "REJECTED", "anchor_count": 1,
             "max_detected_patches": 4},
        ])
        admitted, tracks = validate_saved_gate_decisions(
            summary, {"family_ids": ["FAM_T01"], "track_ids": [1]},
            self.association(), config=None
        )
        self.assertEqual(admitted, ["FAM_T01"])
        self.assertEqual(tracks, [1])

    def test_inconsistent_admitted_ids_rejected(self):
        summary = pd.DataFrame([
            {"family_id": "FAM_T01", "status": "ADMITTED",
             "anchor_count": 1, "max_detected_patches": 12},
            {"family_id": "FAM_T02", "status": "REJECTED",
             "anchor_count": 1, "max_detected_patches": 2},
        ])
        with self.assertRaises(ValueError):
            validate_saved_gate_decisions(
                summary, {"family_ids": ["FAM_T02"], "track_ids": [2]},
                self.association(), config=None
            )

    def test_missing_radius_rejected_before_run_created(self):
        with TemporaryDirectory() as root:
            cfg = self.make_config(root, with_radius=False)
            out = Path(root) / "child"
            with self.assertRaisesRegex(ValueError, "inner_radius_px"):
                execute_from_gates(cfg, input_run=Path(root) / "nonexistent",
                                   until="recovery", run_dir=out)
            self.assertFalse(out.exists())

    def test_incomplete_parent_rejected_before_run_created(self):
        with TemporaryDirectory() as root:
            cfg = self.make_config(root)
            parent = Path(root) / "parent"
            initialize_run(parent, cfg.data, "test", {})
            out = Path(root) / "child"
            with self.assertRaises(ValueError):
                execute_from_gates(cfg, input_run=parent, until="recovery",
                                   run_dir=out)
            self.assertFalse(out.exists())

    def test_reject_other_until(self):
        with TemporaryDirectory() as root:
            cfg = self.make_config(root)
            with self.assertRaises(ValueError):
                execute_from_gates(cfg, input_run=Path(root) / "no",
                                   until="gates", run_dir=Path(root) / "child")


if __name__ == "__main__":
    unittest.main()
