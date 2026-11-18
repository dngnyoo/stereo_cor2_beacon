"""Explicit parent-Run provenance tests without running astronomical computation."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cme_pipeline.core.run_layout import initialize_run, complete_stage
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.standard_input_run import resolve_detection_parent, execute_from_detection


class ExplicitInputRunTests(unittest.TestCase):
    def _fixture(self, base):
        base = Path(base)
        conf = base / "cfg.yaml"
        conf.write_text(
            'experiment:\n  date: "20240503"\n'
            'detection:\n  method: d4\n'
            'tracking:\n  method: a5b\n'
        )
        cfg = StandardConfig.load(conf)
        parent = base / "parent"
        initialize_run(parent, cfg.data, "parentSHA", {"source": "synthetic"})
        data = parent / "02_detection" / "data" / "raw"
        data.mkdir(parents=True)
        names = {
            "patch_activation": "patch_state_p04.npy",
            "component_labels": "patch_component_labels.npy",
            "frame_qc": "frame_detection_summary.csv",
            "detection_scores": "raw_scores.csv",
        }
        outputs = {}
        for key, filename in names.items():
            file = data / filename
            file.write_text("fake artifact")
            outputs[key] = str(file.relative_to(parent))
        (data / "detection_metadata.json").write_text(json.dumps({
            "detector": "d4", "requested_date": "20240503",
        }))
        complete_stage(parent, "detection", {}, outputs, "d4", cfg.data["detection"])
        return cfg, parent, data

    def test_verified_explicit_parent(self):
        with TemporaryDirectory() as temp:
            cfg, parent, data = self._fixture(temp)
            resolved, detector, meta = resolve_detection_parent(cfg, parent)
            self.assertEqual(resolved, parent)
            self.assertEqual(detector, data)
            self.assertIn("patch_activation", meta["outputs"])

    def test_modified_parent_artifact_rejected(self):
        with TemporaryDirectory() as temp:
            cfg, parent, data = self._fixture(temp)
            (data / "patch_state_p04.npy").write_text("tampered")
            with self.assertRaises(ValueError):
                resolve_detection_parent(cfg, parent)

    def test_no_child_created_when_input_differs(self):
        with TemporaryDirectory() as temp:
            cfg, parent, _ = self._fixture(temp)
            cfg.data["detection"]["score_mode"] = "positive"
            child = Path(temp) / "child"
            with self.assertRaises(ValueError):
                execute_from_detection(cfg, input_run=parent, until="tracking", run_dir=child)
            self.assertFalse(child.exists())

    def test_detection_parent_not_valid_for_detection_until(self):
        with TemporaryDirectory() as temp:
            cfg, parent, _ = self._fixture(temp)
            with self.assertRaises(ValueError):
                execute_from_detection(cfg, input_run=parent, until="detection",
                                       run_dir=Path(temp) / "child")


if __name__ == "__main__":
    unittest.main()
