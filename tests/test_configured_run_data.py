"""Basic regression checks for config-driven live C3/A5b execution."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cme_pipeline.configured_run_data import run_configured_data
from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import resolve_stages


class ConfiguredRunDataTests(unittest.TestCase):
    def test_2010_c3_a5b_config_resolves_stage_registry(self):
        cfg = PipelineConfig.from_yaml("configs/dec20101212_c3_a5b.yaml")
        self.assertEqual(cfg.detection.method, "c3")
        self.assertEqual(cfg.tracking.method, "a5b")
        self.assertEqual(cfg.association.method, "identity")
        self.assertAlmostEqual(cfg.geometry.rsun_px, 8.163)
        names = [stage.name for stage in resolve_stages(cfg)]
        self.assertEqual(names, [
            "tracking", "recovery", "association", "birth_gate",
            "size_gate", "segmentation", "products", "validation",
        ])

    def test_legacy_config_defaults_to_c0_without_geometry(self):
        cfg = PipelineConfig.from_yaml("configs/baseline_rule_b.yaml")
        self.assertEqual(cfg.detection.method, "c0")
        self.assertEqual(cfg.tracking.method, "e1")
        self.assertIsNone(cfg.geometry.rsun_px)

    def test_live_requires_verified_pixel_solar_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig()
            cfg.repo_root = tmp
            cfg.tracking.method = "a5b"
            cfg.detection.method = "c3"
            cfg.geometry.rsun_px = None
            config_path = Path(tmp) / "config.yaml"
            cfg.to_yaml(config_path)
            with self.assertRaisesRegex(ValueError, "geometry.rsun_px"):
                run_configured_data(
                    config_path=config_path, date="20101212",
                    run_name="missing_scale", detection_dir=Path(tmp) / "detection",
                )

    def test_cli_detector_override_cannot_silently_replace_yaml(self):
        with self.assertRaisesRegex(ValueError, "conflicts with detection.method"):
            run_configured_data(
                config_path="configs/dec20101212_c3_a5b.yaml",
                date="20101212", run_name="not_executed",
                detection_dir="unused", method="c0",
            )


if __name__ == "__main__":
    unittest.main()
