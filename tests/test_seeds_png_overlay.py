"""Quick visual overlay tests; no COR2 raw files required."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.visualize_seeds_overlay import (
    axes_bounds, png_xy, sector_points, overlay_frame,
)


class TestSeedsPngOverlay(unittest.TestCase):
    def test_png_axes_position(self):
        b = axes_bounds((864, 840))
        self.assertAlmostEqual(b[0], 104.4)
        self.assertAlmostEqual(b[1], 100.8)
        self.assertAlmostEqual(b[2], 655.2)
        x, y = png_xy(128, 128, b)
        self.assertAlmostEqual(x, 432.0)
        self.assertAlmostEqual(y, 428.4)
        with self.assertRaises(ValueError):
            axes_bounds((256, 256))

    def test_pa_orientation(self):
        b = axes_bounds((864, 840))
        cx, cy = png_xy(128, 128, b)
        north = sector_points(0, 2, 22, 60, 128, 128, b)
        east = sector_points(90, 2, 22, 60, 128, 128, b)
        self.assertLess(min(p[1] for p in north), cy)
        self.assertLess(min(p[0] for p in east), cx)

    def test_preserves_original_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pair_030_20101212T052424Z_families.png"
            target = root / "out.png"
            Image.new("RGB", (864, 840), "black").save(source)
            original = source.read_bytes()
            seeds = pd.DataFrame([{
                "parent_event_id": "S1",
                "time": pd.Timestamp("2010-12-12T05:24:00Z"),
                "seeds_bpa": 110., "seeds_width": 55.,
                "seeds_height": 7.,
            }])
            self.assertEqual(overlay_frame(source, target, seeds), 1)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(target.is_file())
            with Image.open(target) as output:
                self.assertEqual(output.size, (864, 840))


if __name__ == "__main__":
    unittest.main()
