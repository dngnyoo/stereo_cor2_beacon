"""CACTus liftoff-window matching regression."""
import unittest

import pandas as pd

from scripts.validate_cactus import validate_cactus


def ref(duration=2):
    return pd.DataFrame([{
        "catalog_event_id": "R", "event_type": "CME",
        "onset_utc": "2024-05-03T09:00:00Z",
        "liftoff_duration_hours": duration,
        "cpa_deg": 30, "angular_width_deg": 20,
    }])


def frames(*rows):
    return pd.DataFrame(rows, columns=["event_id", "timestamp", "cpa_deg", "angular_width_deg"])


class TestLiftoff(unittest.TestCase):
    def test_only_common_frames(self):
        f = frames(("D", "2024-05-02T09:00:00Z", 180, 100),
                   ("D", "2024-05-03T09:30:00Z", 30, 20),
                   ("D", "2024-05-03T11:00:00Z", 30, 20),
                   ("D", "2024-05-03T12:00:00Z", 180, 100),
                   ("old", "2024-05-02T10:00:00Z", 30, 20))
        p, _, s = validate_cactus(ref(), f, time_mode="liftoff", evaluation_date="20240503")
        self.assertEqual(s["detected_events"], 1)
        self.assertEqual(s["matched_events"], 1)
        self.assertEqual(p.iloc[0].common_frames, 2)
        self.assertEqual(p.iloc[0].angular_iou, 1.0)

    def test_gap_not_overlap(self):
        f = frames(("D", "2024-05-03T08:00:00Z", 30, 20),
                   ("D", "2024-05-03T12:00:00Z", 30, 20))
        p, _, s = validate_cactus(ref(), f, time_mode="liftoff")
        self.assertTrue(p.empty)
        self.assertIsNone(s["mean_matched_angular_iou"])

    def test_zero_duration(self):
        f = frames(("D", "2024-05-03T09:00:00Z", 30, 20))
        _, _, s = validate_cactus(ref(0), f, time_mode="liftoff")
        self.assertEqual(s["matched_events"], 1)
        _, _, s = validate_cactus(ref(0), frames(("D", "2024-05-03T09:00:01Z", 30, 20)), time_mode="liftoff")
        self.assertEqual(s["matched_events"], 0)

    def test_invalid_duration(self):
        with self.assertRaises(ValueError):
            validate_cactus(ref(-1), frames(), time_mode="liftoff")
