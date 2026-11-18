"""Architecture contract smoke tests; independent of legacy execution."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from cme_pipeline.core.standard_stages import (
    STAGES, STAGE_BY_NAME, resolve_visualization_mode, stages_until, stage_data_path
)
from cme_pipeline.core.run_layout import initialize_run, complete_stage, verify_stage


class StandardStageTests(unittest.TestCase):
    def test_order_and_contracts(self):
        self.assertEqual(len(STAGES), 9)
        self.assertEqual(len(set(s.folder for s in STAGES)), 9)
        self.assertEqual(stages_until("gates")[-1].name.value, "gates")

    def test_visualization_precedence(self):
        cfg = {"visualization": {"default": "off"},
               "tracking": {"visualization": {"mode": "diagnostic"}}}
        self.assertEqual(resolve_visualization_mode("detection", config=cfg), "off")
        self.assertEqual(resolve_visualization_mode("tracking", config=cfg), "diagnostic")
        self.assertEqual(resolve_visualization_mode("tracking", config=cfg, cli_override="all"), "all")

    def test_run_layout_isolated(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "one_run"
            initialize_run(root, {"experiment": "unit_test"}, "abc123", {"source": "synthetic"})
            with self.assertRaises(FileExistsError):
                initialize_run(root, {}, "abc123", {})
            spec = STAGE_BY_NAME["preprocessing"]
            outputs = {}
            for name in spec.produces:
                dest = stage_data_path(root, "preprocessing") / (name + ".bin")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"example")
                outputs[name] = str(dest.relative_to(root))
            complete_stage(root, "preprocessing", {"fits_sources": "input.fits"},
                           outputs, "fixture", {})
            self.assertEqual(verify_stage(root, "preprocessing")["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
