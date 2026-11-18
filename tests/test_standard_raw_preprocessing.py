"""Stage 01 raw local-first selection tests; no network or scientific work."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.core.run_layout import verify_stage
from cme_pipeline.standard_runner import execute_to_gates, PREPROCESSING_FILES


class StandardRawInputTests(unittest.TestCase):
    def _config(self, folder, *, raw_dir=None):
        path = Path(folder) / "config.yaml"
        source_line = f'  raw_dir: "{raw_dir}"\n' if raw_dir is not None else ""
        path.write_text(
            'experiment:\n  date: "20240510"\n'
            'input:\n' + source_line +
            '  warmup_days: 2\n'
            'visualization:\n  default: "off"\n'
            'preprocessing:\n  method: dual_archive_rd\n'
        )
        return StandardConfig.load(path)

    @staticmethod
    def _fake_preprocess(manifest, *, mask_path=None):
        out = Path(manifest).parent / "preprocessed"
        out.mkdir()
        for filename in PREPROCESSING_FILES.values():
            (out / filename).write_text("fixture")
        return out

    def test_local_path_is_used_without_downloading(self):
        with TemporaryDirectory() as folder:
            raw = Path(folder) / "raw"
            raw.mkdir()
            config = self._config(folder, raw_dir=raw)
            output = Path(folder) / "run"
            def fake_prepare(**kwargs):
                self.assertEqual(kwargs["source"], "local")
                self.assertEqual(Path(kwargs["input_dir"]), raw)
                self.assertEqual(kwargs["date"], "20240510")
                self.assertEqual(kwargs["warmup_days"], 2)
                manifest = Path(kwargs["output_dir"]) / "input_manifest.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text("{}")
                return manifest
            with patch("cme_pipeline.data_preparation.prepare_source", side_effect=fake_prepare) as called, patch(
                "cme_pipeline.north_up_preprocessing.run_no_msb_north_up",
                side_effect=self._fake_preprocess
            ):
                self.assertEqual(execute_to_gates(config, until="preprocessing", run_dir=output), output)
            self.assertEqual(called.call_count, 1)
            self.assertEqual(verify_stage(output, "preprocessing")["implementation"],
                             "dual_archive_rd:raw_local")

    def test_no_raw_path_uses_download(self):
        with TemporaryDirectory() as folder:
            config = self._config(folder)
            output = Path(folder) / "run"
            def fake_prepare(**kwargs):
                self.assertEqual(kwargs["source"], "download")
                self.assertIsNone(kwargs["input_dir"])
                manifest = Path(kwargs["output_dir"]) / "input_manifest.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text("{}")
                return manifest
            with patch("cme_pipeline.data_preparation.prepare_source", side_effect=fake_prepare) as called, patch(
                "cme_pipeline.north_up_preprocessing.run_no_msb_north_up",
                side_effect=self._fake_preprocess
            ):
                execute_to_gates(config, until="preprocessing", run_dir=output)
            self.assertEqual(called.call_count, 1)
            self.assertEqual(verify_stage(output, "preprocessing")["implementation"],
                             "dual_archive_rd:raw_download")

    def test_bad_explicit_local_path_never_falls_back_to_download(self):
        with TemporaryDirectory() as folder:
            config = self._config(folder, raw_dir=Path(folder) / "missing")
            output = Path(folder) / "run"
            with patch("cme_pipeline.data_preparation.prepare_source") as called:
                with self.assertRaises(FileNotFoundError):
                    execute_to_gates(config, until="preprocessing", run_dir=output)
            called.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
