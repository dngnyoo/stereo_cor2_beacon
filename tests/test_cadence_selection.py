"""Offline strict COR2 d7 source-selection tests."""
import unittest
from datetime import datetime, timedelta

from cme_pipeline.cadence_selection import select_d7_pairs


def observation(minute_offset):
    t = datetime(2024, 5, 10, 0, 23, 54) + timedelta(minutes=minute_offset)
    return {"file": f"/tmp/sta_{t:%Y%m%d_%H%M%S}_d7c2.fts",
            "date_obs": t.isoformat()}


class TestD7Cadence(unittest.TestCase):
    def test_valid_30min_pairs_and_gap(self):
        selection = select_d7_pairs([
            observation(60), observation(0), observation(30), observation(120)
        ])
        self.assertEqual(selection["n_selected"], 4)
        self.assertEqual(selection["n_pairs"], 2)
        self.assertEqual(selection["n_gaps"], 1)
        self.assertEqual(selection["gaps"][0]["estimated_missing_slots"], 1)
        self.assertEqual(selection["pairs"][0]["delta_t_sec"], 1800)

    def test_off_phase_excluded(self):
        selection = select_d7_pairs([observation(0), observation(15), observation(30)])
        self.assertEqual(selection["n_selected"], 2)
        self.assertEqual(selection["n_pairs"], 1)

    def test_filename_time_mismatch_rejected(self):
        bad = observation(0)
        bad["date_obs"] = "2024-05-10T00:26:00"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            select_d7_pairs([bad])

    def test_non_d7_rejected(self):
        bad = observation(0)
        bad["file"] = bad["file"].replace("_d7c2", "_d4c2")
        with self.assertRaisesRegex(ValueError, "Non-d7"):
            select_d7_pairs([bad])

    def test_duplicate_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-unique"):
            select_d7_pairs([observation(0), observation(0)])


if __name__ == "__main__":
    unittest.main()
