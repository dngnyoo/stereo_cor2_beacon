"""Orchestrator control-flow tests with all science and plotting mocked."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.standard_full_orchestrator import (
    STEPS, run_full_standard, _apply_verified_geometry,
)


class FullStandardOrchestrationTests(TestCase):
    def _config(self, folder):
        path = Path(folder) / "standard.yaml"
        path.write_text(
            'experiment:\n  date: "20240510"\n'
            'input:\n'
            '  raw_dir: /raw/20240510\n'
            '  warmup_days: 2\n'
            'visualization:\n  default: "off"\n'
            'catalog:\n  method: event_kinematics\n'
            '  rsun_px: 8.525\n'
            'validation:\n  method: seeds\n'
        )
        return StandardConfig.load(path)

    def test_no_implicit_overwrite(self):
        with TemporaryDirectory() as folder:
            cfg = self._config(folder)
            with self.assertRaises(FileExistsError):
                run_full_standard(cfg, until="validation", run_dir=folder)

    def test_full_chain_stage_order(self):
        self.assertEqual(
            [s.stage for s in STEPS],
            ["detection", "tracking", "association", "gates",
             "recovery", "segmentation", "catalog", "validation"]
        )
        self.assertEqual(
            [s.parent_stage for s in STEPS][1:],
            ["detection", "tracking", "association", "gates",
             "recovery", "segmentation", "catalog"]
        )

    def test_missing_full_catalog_geometry_for_import_fails(self):
        with TemporaryDirectory() as folder:
            cfg = self._config(folder)
            cfg.data["input"] = {"preprocessed_dir": str(Path(folder) / "input")}
            cfg.data["catalog"].pop("rsun_px")
            out = Path(folder) / "out"
            with self.assertRaisesRegex(ValueError, "rsun_px"):
                run_full_standard(cfg, until="validation", run_dir=out)
            self.assertFalse(out.exists())

    def test_mocked_full_execution_preserves_explicit_stage_parents(self):
        with TemporaryDirectory() as folder:
            cfg = self._config(folder)
            root = Path(folder) / "full"
            calls = []
            def prep(config, *, until, run_dir, visualize):
                self.assertEqual(until, "detection")
                self.assertEqual(visualize, "off")
                Path(run_dir).mkdir()
                calls.append(("detection", None, str(run_dir)))
                return Path(run_dir)
            def child(stage):
                def call(config, *, input_run, until, run_dir, visualize):
                    self.assertEqual(until, stage)
                    self.assertEqual(visualize, "off")
                    Path(run_dir).mkdir()
                    calls.append((stage, str(input_run), str(run_dir)))
                    return Path(run_dir)
                return call
            import cme_pipeline.standard_input_run as detect
            import cme_pipeline.standard_tracking_parent as track
            import cme_pipeline.standard_association_parent as assoc
            import cme_pipeline.standard_recovery_parent as gates
            import cme_pipeline.standard_segmentation_parent as recovery
            import cme_pipeline.standard_catalog_parent as seg
            import cme_pipeline.standard_seeds_validation_parent as cat
            stages = [(detect, "execute_from_detection", "tracking"),
                      (track, "execute_from_tracking", "association"),
                      (assoc, "execute_from_association", "gates"),
                      (gates, "execute_from_gates", "recovery"),
                      (recovery, "execute_from_recovery", "segmentation"),
                      (seg, "execute_from_segmentation", "catalog"),
                      (cat, "execute_from_catalog", "validation")]
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch("cme_pipeline.standard_runner.execute_to_gates", side_effect=prep))
                # The orchestrator imports function aliases at invocation time.
                for module, function, stage in stages:
                    stack.enter_context(patch.object(module, function, side_effect=child(stage)))
                stack.enter_context(patch("cme_pipeline.standard_full_orchestrator.execute_to_gates", side_effect=prep))
                stack.enter_context(patch("cme_pipeline.standard_full_orchestrator.verify_stage"))
                stack.enter_context(patch("cme_pipeline.standard_full_orchestrator._ensure_seeds"))
                stack.enter_context(patch("cme_pipeline.standard_full_orchestrator._apply_verified_geometry"))
                stack.enter_context(patch("cme_pipeline.standard_full_visualization.render_full_stage", return_value=[]))
                actual = run_full_standard(cfg, until="validation", run_dir=root)
            self.assertEqual(actual, root)
            self.assertEqual(len(calls), 8)
            for current, previous in zip(calls[1:], calls[:-1]):
                self.assertEqual(current[1], previous[2])
            summary = json.loads((root / "full_run_summary.json").read_text())
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(len(summary["steps"]), 9)
            self.assertEqual([x["stage"] for x in summary["steps"]],
                             ["preprocessing", "detection", "tracking", "association",
                              "gates", "recovery", "segmentation", "catalog", "validation"])


if __name__ == "__main__":
    import unittest
    unittest.main()
