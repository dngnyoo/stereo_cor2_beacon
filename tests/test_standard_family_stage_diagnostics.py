"""Read-only Stage 05-07 visualization helper regression tests."""
import unittest
import numpy as np
import pandas as pd

from cme_pipeline.family_stage_diagnostics import _family_id, _indices, _xy


class FamilyStageDiagnosticTests(unittest.TestCase):
    def test_identity_family_naming(self):
        self.assertEqual(_family_id(3), "FAM_T03")
        self.assertEqual(_family_id(27), "FAM_T27")

    def test_diagnostic_samples_include_first_last_and_max(self):
        manifest = pd.DataFrame({
            "end_time": [
                "2024-05-09T23:30:00Z",
                "2024-05-10T00:00:00Z",
                "2024-05-10T00:30:00Z",
                "2024-05-10T01:00:00Z",
                "2024-05-10T01:30:00Z",
                "2024-05-11T00:00:00Z",
            ],
        })
        scores = np.array([999., 1., 8., 3., 2., 999.])
        diagnostic = _indices(manifest, "20240510", "diagnostic", scores)
        self.assertTrue({1, 2, 4}.issubset(set(diagnostic)))
        self.assertTrue(set(diagnostic).issubset({1, 2, 3, 4}))
        all_frames = _indices(manifest, "20240510", "all", scores)
        np.testing.assert_array_equal(all_frames, [1, 2, 3, 4])

    def test_solar_pa_zero_points_north(self):
        x, y = _xy(20., 0., (126.5, 127.5))
        self.assertAlmostEqual(x, 126.5)
        self.assertAlmostEqual(y, 147.5)

    def test_bad_family_palette_fails(self):
        from cme_pipeline.family_stage_diagnostics import _palette
        with self.assertRaises(ValueError):
            _palette("unresolved")
        self.assertEqual(_palette("FAM_T03"), _palette("FAM_T03"))


if __name__ == "__main__":
    unittest.main()
