"""Stage 09 official SEEDS-only validation preflight regression."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cme_pipeline.core.run_layout import initialize_run
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.standard_seeds_validation_parent import (
    execute_from_catalog, validation_parameters
)


class SEEDSValidationParentTests(unittest.TestCase):
    @staticmethod
    def config(folder, reference=None, *, method="seeds"):
        path = Path(folder) / "cfg.yaml"
        section = f"  seeds_frames_csv: {reference}\n" if reference else ""
        path.write_text(
            'experiment:\n  date: "20240503"\n'
            'validation:\n'
            f'  method: {method}\n' + section
        )
        return StandardConfig.load(path)

    def test_missing_reference_path_fails_closed(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp)
            with self.assertRaisesRegex(ValueError, "seeds_frames_csv"):
                validation_parameters(cfg)

    def test_reference_file_absent(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp, str(Path(temp) / "missing.csv"))
            with self.assertRaises(FileNotFoundError):
                validation_parameters(cfg)

    def test_reject_other_catalog_methods(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp, method="external_catalogs")
            with self.assertRaisesRegex(ValueError, "SEEDS only"):
                validation_parameters(cfg)

    def test_reject_nonseeds_catalogs(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp)
            cfg.data["validation"]["catalogs"] = ["SEEDS", "CACTus"]
            with self.assertRaises(ValueError):
                validation_parameters(cfg)

    def test_reject_bad_threshold_before_loading_parent(self):
        with TemporaryDirectory() as temp:
            reference = Path(temp) / "seeds_frames.csv"
            reference.write_text("DATE,TIME,BPA,PASTART,PAEND,HMLEAD,parent_event_id\n")
            cfg = self.config(temp, str(reference))
            cfg.data["validation"]["min_mean_iou"] = -0.5
            with self.assertRaisesRegex(ValueError, "threshold"):
                validation_parameters(cfg)

    def test_missing_catalog_parent_does_not_create_child(self):
        with TemporaryDirectory() as temp:
            reference = Path(temp) / "seeds_frames.csv"
            reference.write_text(
                "DATE,TIME,BPA,PASTART,PAEND,HMLEAD,parent_event_id\n"
                "2024/05/03,01:00:00,80,50,110,4.0,SEEDS_COR2A_20240503T010000_W060_V0500_P080\n"
            )
            cfg = self.config(temp, str(reference))
            parent = Path(temp) / "parent"
            initialize_run(parent, cfg.data, "test", {})
            out = Path(temp) / "new"
            with self.assertRaises(ValueError):
                execute_from_catalog(cfg, input_run=parent,
                                     until="validation", run_dir=out)
            self.assertFalse(out.exists())

    def test_wrong_until_rejected(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp)
            with self.assertRaises(ValueError):
                execute_from_catalog(cfg, input_run=Path(temp) / "missing",
                                     until="catalog", run_dir=Path(temp) / "new")


if __name__ == "__main__":
    unittest.main()
