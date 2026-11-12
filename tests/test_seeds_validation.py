"""SEEDS matching must be independent, one-to-one and time-aware."""
import unittest

import pandas as pd

from scripts.validate_seeds import _read_seeds, validate_seeds


class TestSeedsValidation(unittest.TestCase):
    def test_one_to_one_and_unmatched(self):
        seeds = pd.DataFrame([
            {"parent_event_id": "S1", "timestamp_utc": pd.Timestamp("2010-12-12T12:00Z"),
             "seeds_bpa": 359., "seeds_width": 30., "seeds_lead": 3.},
            {"parent_event_id": "S2", "timestamp_utc": pd.Timestamp("2010-12-12T12:30Z"),
             "seeds_bpa": 90., "seeds_width": 20., "seeds_lead": 3.},
        ])
        det = pd.DataFrame([
            {"event_id": "D1", "pair_index": 1,
             "timestamp_utc": pd.Timestamp("2010-12-12T12:00Z"),
             "cpa_deg": 1., "angular_width_deg": 30., "front_rsun": 3.1},
            {"event_id": "D2", "pair_index": 2,
             "timestamp_utc": pd.Timestamp("2010-12-12T12:30Z"),
             "cpa_deg": 210., "angular_width_deg": 20., "front_rsun": 3.2},
        ])
        pairs, statuses, summary = validate_seeds(seeds, det)
        self.assertEqual(summary["matched_events"], 1)
        self.assertEqual(summary["unmatched_detected_events"], 1)
        self.assertEqual(summary["unmatched_seeds_events"], 1)
        self.assertEqual(int(pairs["assigned_match"].sum()), 1)
        self.assertEqual(statuses["status"].value_counts()["MATCHED"], 2)

    def test_observation_window_excludes_outside_detections(self):
        seeds = pd.DataFrame([
            {"parent_event_id": "S", "timestamp_utc": pd.Timestamp("2010-12-12T12:00Z"),
             "seeds_bpa": 30., "seeds_width": 50., "seeds_lead": 3.}
        ])
        det = pd.DataFrame([
            {"event_id": "D", "pair_index": 9,
             "timestamp_utc": pd.Timestamp("2010-12-12T17:00Z"),
             "cpa_deg": 30., "angular_width_deg": 50., "front_rsun": 3.},
        ])
        pairs, statuses, summary = validate_seeds(seeds, det)
        self.assertTrue(pairs.empty)
        self.assertEqual(summary["detected_events_outside_evaluation_window"], 1)
        self.assertIsNone(summary["precision_conditional"])
        self.assertEqual(summary["recall_conditional"], 0.)

    def test_angle_wrap_and_partial_overlap(self):
        seeds = pd.DataFrame([
            {"parent_event_id": "S", "timestamp_utc": pd.Timestamp("2010-12-12T12:00Z"),
             "seeds_bpa": 359., "seeds_width": 30., "seeds_lead": 4.}
        ])
        det = pd.DataFrame([
            {"event_id": "D", "pair_index": 1,
             "timestamp_utc": pd.Timestamp("2010-12-12T12:00Z"),
             "cpa_deg": 1., "angular_width_deg": 30., "front_rsun": 4.5},
        ])
        pairs, _, summary = validate_seeds(seeds, det)
        self.assertEqual(summary["matched_events"], 1)
        self.assertGreater(pairs.iloc[0]["mean_angular_iou"], .8)
        self.assertAlmostEqual(pairs.iloc[0]["mean_radial_error_rsun"], .5)

    def test_zero_iou_never_matched(self):
        seeds = pd.DataFrame([
            {"parent_event_id": "S", "timestamp_utc": pd.Timestamp("2010-12-12T12:00Z"),
             "seeds_bpa": 10., "seeds_width": 20., "seeds_lead": 4.}
        ])
        det = pd.DataFrame([
            {"event_id": "D", "pair_index": 1,
             "timestamp_utc": pd.Timestamp("2010-12-12T12:00Z"),
             "cpa_deg": 190., "angular_width_deg": 20., "front_rsun": 4.},
        ])
        pairs, _, summary = validate_seeds(seeds, det)
        self.assertFalse(pairs.iloc[0]["assigned_match"])
        self.assertEqual(summary["matched_events"], 0)


if __name__ == "__main__":
    unittest.main()
