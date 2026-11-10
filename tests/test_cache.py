"""Unit tests for StageCache and invalidation logic."""

import tempfile
import unittest
from pathlib import Path

from cme_pipeline.core.cache import StageCache


class TestStageCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.cache = StageCache(self.cache_dir)
        self.stage_order = ["tracking", "recovery", "association", "birth_gate", "size_gate", "segmentation", "validation"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cache_miss_initially(self):
        self.assertFalse(self.cache.is_valid("tracking", "hash123"))

    def test_save_and_hit(self):
        dummy_data = {"result": [1, 2, 3], "status": "OK"}
        self.cache.save_output("tracking", "hash123", dummy_data)

        self.assertTrue(self.cache.is_valid("tracking", "hash123"))
        self.assertFalse(self.cache.is_valid("tracking", "different_hash"))

        loaded = self.cache.load_output("tracking")
        self.assertEqual(loaded["result"], [1, 2, 3])

    def test_invalidate_single(self):
        self.cache.save_output("recovery", "hash_rec", {"patches": 100})
        self.assertTrue(self.cache.is_valid("recovery", "hash_rec"))

        self.cache.invalidate("recovery")
        self.assertFalse(self.cache.is_valid("recovery", "hash_rec"))

    def test_invalidate_from(self):
        # Cache 3 stages
        self.cache.save_output("tracking", "h_track", {"t": 1})
        self.cache.save_output("recovery", "h_rec", {"r": 2})
        self.cache.save_output("association", "h_assoc", {"a": 3})
        self.cache.save_output("size_gate", "h_size", {"s": 4})

        # Invalidate from association onwards
        self.cache.invalidate_from("association", self.stage_order)

        # tracking and recovery should still be valid
        self.assertTrue(self.cache.is_valid("tracking", "h_track"))
        self.assertTrue(self.cache.is_valid("recovery", "h_rec"))

        # association and size_gate should be invalidated
        self.assertFalse(self.cache.is_valid("association", "h_assoc"))
        self.assertFalse(self.cache.is_valid("size_gate", "h_size"))


if __name__ == "__main__":
    unittest.main()

