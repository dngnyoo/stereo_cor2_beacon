"""Preflight guards for explicit parent Tracking Run (no science files needed)."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.core.run_layout import initialize_run
from cme_pipeline.standard_tracking_parent import execute_from_tracking, resolve_tracking_parent


class TrackingParentGuardTests(unittest.TestCase):
    def make_config(self, tmp):
        file = Path(tmp) / "config.yaml"
        file.write_text('experiment:\n  date: "20240503"\ntracking:\n  method: a5b\n')
        return StandardConfig.load(file)

    def test_parent_missing(self):
        with TemporaryDirectory() as tmp:
            cfg = self.make_config(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_tracking_parent(cfg, Path(tmp) / "missing")

    def test_parent_without_completed_tracking_rejected(self):
        with TemporaryDirectory() as tmp:
            cfg = self.make_config(tmp)
            parent = Path(tmp) / "parent"
            initialize_run(parent, cfg.data, "sample", {})
            child = Path(tmp) / "child"
            with self.assertRaises(ValueError):
                execute_from_tracking(cfg, input_run=parent, until="gates",
                                      run_dir=child)
            self.assertFalse(child.exists())

    def test_unsupported_until_rejected_before_file_io(self):
        with TemporaryDirectory() as tmp:
            cfg = self.make_config(tmp)
            with self.assertRaises(ValueError):
                execute_from_tracking(cfg, input_run=Path(tmp) / "missing",
                                      until="tracking", run_dir=Path(tmp) / "child")


if __name__ == "__main__":
    unittest.main()
