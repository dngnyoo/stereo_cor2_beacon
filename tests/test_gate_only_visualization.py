"""Focused checks for Family-color assignment in observed-only gate comparisons."""
import unittest
import numpy as np

from cme_pipeline.gate_only_visualization import _match_ids, _palette


class TestGateOnlyColors(unittest.TestCase):
    def test_overlap_reuses_color_even_when_track_ids_change(self):
        base = np.zeros((2, 64, 64), dtype=np.int32)
        exp = np.zeros_like(base)
        base[:, 10, 10] = 1
        base[:, 11, 11] = 2
        exp[:, 10, 10] = 19
        exp[:, 11, 11] = 28
        exp[:, 15, 15] = 30  # new Family; unique color
        color_base, color_exp = _match_ids(base, exp)
        self.assertEqual(color_base[1], color_exp[19])
        self.assertEqual(color_base[2], color_exp[28])
        self.assertNotIn(color_exp[30], color_base.values())

    def test_color_values_stay_in_rgb_range(self):
        for i in range(150):
            self.assertTrue(all(0 <= c <= 1 for c in _palette(i)))


if __name__ == "__main__":
    unittest.main()
