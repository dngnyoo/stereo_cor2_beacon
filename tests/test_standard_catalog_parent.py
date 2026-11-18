"""Stage 08 Catalog preflight tests; no catalog speed guessed or interpolated."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from cme_pipeline.core.run_layout import initialize_run
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.standard_catalog_parent import execute_from_segmentation, resolve_segmentation_parent


class CatalogParentTests(unittest.TestCase):
    @staticmethod
    def config(folder, with_rsun=True):
        file = Path(folder) / "cfg.yaml"
        radius = "  rsun_px: 58.0\n" if with_rsun else ""
        file.write_text('experiment:\n  date: "20240503"\n'
                        'catalog:\n  method: event_kinematics\n' + radius)
        return StandardConfig.load(file)

    def test_no_rsun_does_not_create_output(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp, with_rsun=False)
            child = Path(temp) / "child"
            with self.assertRaisesRegex(ValueError, "catalog.rsun_px"):
                execute_from_segmentation(cfg, input_run=Path(temp) / "missing",
                                          until="catalog", run_dir=child)
            self.assertFalse(child.exists())

    def test_invalid_rsun_does_not_create_output(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp)
            cfg.data["catalog"]["rsun_px"] = 0
            child = Path(temp) / "child"
            with self.assertRaises(ValueError):
                execute_from_segmentation(cfg, input_run=Path(temp) / "missing",
                                          until="catalog", run_dir=child)
            self.assertFalse(child.exists())

    def test_incomplete_parent_rejected(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp)
            parent = Path(temp) / "parent"
            initialize_run(parent, cfg.data, "synthetic", {})
            child = Path(temp) / "child"
            with self.assertRaises(ValueError):
                execute_from_segmentation(cfg, input_run=parent,
                                          until="catalog", run_dir=child)
            self.assertFalse(child.exists())

    def test_unknown_until_rejected(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp)
            with self.assertRaises(ValueError):
                execute_from_segmentation(cfg, input_run=Path(temp) / "missing",
                                          until="segmentation", run_dir=Path(temp) / "child")

    def test_unknown_catalog_parameter_rejected(self):
        with TemporaryDirectory() as temp:
            cfg = self.config(temp)
            cfg.data["catalog"]["unsupported_option"] = 1
            with self.assertRaises(ValueError):
                execute_from_segmentation(cfg, input_run=Path(temp) / "missing",
                                          until="catalog", run_dir=Path(temp) / "child")


if __name__ == "__main__":
    unittest.main()
