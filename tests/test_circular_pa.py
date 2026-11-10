"""Unit tests for Circular PA Geometry around 0/360 degrees."""

import unittest
import numpy as np

from cme_pipeline.utils.geometry import (
    circular_diff_deg,
    circular_sector_overlap,
    minimal_circular_enclosing_arc,
)


class TestCircularPA(unittest.TestCase):
    def test_circular_difference_wraparound(self):
        # Difference between 5 deg and 355 deg should be +10 deg or -10 deg depending on direction
        diff_fwd = circular_diff_deg(5.0, 355.0)
        self.assertAlmostEqual(diff_fwd, 10.0, places=5)

        diff_bwd = circular_diff_deg(355.0, 5.0)
        self.assertAlmostEqual(diff_bwd, -10.0, places=5)

        # Opposite angles (180 deg)
        diff_opp = abs(circular_diff_deg(0.0, 180.0))
        self.assertAlmostEqual(diff_opp, 180.0, places=5)

    def test_minimal_circular_enclosing_arc(self):
        # Arc straddling North (0 deg): 350 deg, 0 deg, 10 deg
        angles = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
        arc = minimal_circular_enclosing_arc(angles)
        self.assertAlmostEqual(arc["arc_span"], 20.0, places=4)
        self.assertAlmostEqual(arc["arc_center"], 0.0, places=4)
        self.assertAlmostEqual(arc["arc_start"], 350.0, places=4)
        self.assertAlmostEqual(arc["arc_end"], 10.0, places=4)

    def test_circular_sector_overlap(self):
        # Sector 1: CPA 5 deg, width 20 deg (range 355 to 15 deg)
        # Sector 2: CPA 355 deg, width 20 deg (range 345 to 5 deg)
        # Overlap should be from 355 to 5 deg = 10 deg
        res = circular_sector_overlap(5.0, 20.0, 355.0, 20.0)
        self.assertGreater(res["iou"], 0.0)
        self.assertAlmostEqual(res["inter_deg"], 10.0, delta=0.1)
        self.assertAlmostEqual(res["union_deg"], 30.0, delta=0.1)
        self.assertAlmostEqual(res["iou"], 10.0 / 30.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()

