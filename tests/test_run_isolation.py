"""Unit tests for Run Directory and Cache Isolation."""

import tempfile
import unittest
from pathlib import Path

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext


class TestRunIsolation(unittest.TestCase):
    def test_run_isolation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_a = PipelineConfig(run_name="experiment_alpha", output_base_dir=tmp_dir)
            cfg_b = PipelineConfig(run_name="experiment_beta", output_base_dir=tmp_dir)

            ctx_a = PipelineContext(cfg_a)
            ctx_b = PipelineContext(cfg_b)

            # Directories must be completely separate
            self.assertNotEqual(ctx_a.run_dir, ctx_b.run_dir)
            self.assertEqual(ctx_a.run_dir.name, "experiment_alpha")
            self.assertEqual(ctx_b.run_dir.name, "experiment_beta")

            # Caches must be completely isolated
            ctx_a.cache.save_output("tracking", "hash_alpha", {"val": "alpha"})
            self.assertTrue(ctx_a.cache.is_valid("tracking", "hash_alpha"))
            self.assertFalse(ctx_b.cache.is_valid("tracking", "hash_alpha"))


if __name__ == "__main__":
    unittest.main()

