"""Smoke tests for standard run isolation with a synthetic preprocessing import.

No science algorithms or real FITS are run by this test module.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.core.run_layout import verify_stage
from cme_pipeline.standard_runner import execute_to_gates


class StandardRunnerTests(unittest.TestCase):
    @staticmethod
    def _setup(tmp: str):
        base = Path(tmp)
        source = base / "preprocessed_fixture"
        source.mkdir()
        for name in (
            "north_up_RD_DNs_per_second.npy",
            "north_up_pair_valid_masks.npy",
            "01_rd_manifest.csv",
            "target_wcs_header.txt",
            "preprocessing_metadata.json",
        ):
            (source / name).write_bytes(b"fixture")
        (source / "preprocessing_metadata.json").write_text(
            json.dumps({"requested_date": "20240503", "unit": "DN/s", "msb_applied": False})
        )
        yaml = base / "baseline.yaml"
        yaml.write_text(
            'experiment:\n  date: "20240503"\n'
            'input:\n  preprocessed_dir: "' + str(source) + '"\n'
            'visualization:\n  default: "off"\n'
            'preprocessing:\n  method: dual_archive_rd\n'
        )
        return StandardConfig.load(yaml), base / "runs" / "fixture"

    def test_one_run_one_stage_and_checksum(self):
        with TemporaryDirectory() as tmp:
            cfg, run = self._setup(tmp)
            output = execute_to_gates(cfg, until="preprocessing", run_dir=run)
            self.assertEqual(run, output)
            self.assertTrue((run / "01_preprocessing" / "data" / "01_rd_manifest.csv").exists())
            self.assertFalse((run / "02_detection").exists())
            self.assertEqual(verify_stage(run, "preprocessing")["status"], "COMPLETED")
            manifest = json.loads((run / "run_manifest.json").read_text())
            self.assertEqual(manifest["stages"]["preprocessing"], "COMPLETED")
            with self.assertRaises(FileExistsError):
                execute_to_gates(cfg, until="preprocessing", run_dir=run)

    def test_visualization_not_silently_skipped(self):
        with TemporaryDirectory() as tmp:
            cfg, run = self._setup(tmp)
            with self.assertRaises(NotImplementedError):
                execute_to_gates(cfg, until="preprocessing", run_dir=run, visualize="all")
            self.assertFalse(run.exists())

    def test_missing_input_cannot_create_run(self):
        with TemporaryDirectory() as tmp:
            cfg, run = self._setup(tmp)
            (Path(tmp) / "preprocessed_fixture").rename(Path(tmp) / "unavailable")
            with self.assertRaises(FileNotFoundError):
                execute_to_gates(cfg, until="preprocessing", run_dir=run)
            self.assertFalse(run.exists())


if __name__ == "__main__":
    unittest.main()
