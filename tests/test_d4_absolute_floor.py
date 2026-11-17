"""D4 is opt-in; C3 cannot silently inherit an absolute P04 floor."""
import unittest

from cme_pipeline.live_detection import run_detection


class TestD4AbsoluteFloor(unittest.TestCase):
    def test_d4_requires_floor(self):
        with self.assertRaisesRegex(ValueError, "requires finite"):
            run_detection("/does/not/exist", method="d4", floor_mode="outer_p95")

    def test_d4_requires_outer_p95(self):
        with self.assertRaisesRegex(ValueError, "requires floor_mode=outer_p95"):
            run_detection("/does/not/exist", method="d4", floor_mode="frame_median",
                          absolute_p04_floor=0.5)

    def test_d4_rejects_invalid_floor(self):
        for val in (-0.1, float("nan"), float("inf")):
            with self.subTest(val=val), self.assertRaises(ValueError):
                run_detection("/does/not/exist", method="d4", floor_mode="outer_p95",
                              absolute_p04_floor=val)

    def test_c3_rejects_d4_only_floor(self):
        with self.assertRaisesRegex(ValueError, "only for D4"):
            run_detection("/does/not/exist", method="c3", floor_mode="outer_p95",
                          absolute_p04_floor=0.5)

    def test_c0_rejects_d4_only_floor(self):
        with self.assertRaisesRegex(ValueError, "only for D4"):
            run_detection("/does/not/exist", method="c0", absolute_p04_floor=0.5)


if __name__ == "__main__":
    unittest.main()
