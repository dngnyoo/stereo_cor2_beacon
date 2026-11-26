import unittest

import pandas as pd

from scripts.validate_cactus import angular_iou, sector, validate_cactus


def ref(*rows):
    return pd.DataFrame(rows, columns=["catalog_event_id", "event_type", "onset_utc",
                                       "cpa_deg", "angular_width_deg"])


def frames(*rows):
    return pd.DataFrame(rows, columns=["event_id", "timestamp", "cpa_deg", "angular_width_deg"])


T = "2010-12-12T04:54:00Z"


class TestCactusValidation(unittest.TestCase):
    def test_wrap_and_halo(self):
        self.assertAlmostEqual(angular_iou(sector(0, 20), sector(355, 10)), .5)
        self.assertAlmostEqual(angular_iou(sector(0, 360), sector(50, 90)), .25)
        self.assertEqual(angular_iou(sector(0, 10), sector(10, 10)), 0)

    def test_union_does_not_fill_gap(self):
        p, _, s = validate_cactus(ref(("R", "CME", T, 30, 60)),
                                 frames(("D", T, 5, 10), ("D", T, 55, 10)))
        self.assertAlmostEqual(p.iloc[0].angular_iou, 1/3)
        self.assertAlmostEqual(s["mean_matched_angular_iou"], 1/3)

    def test_one_to_one_and_flow(self):
        p, status, s = validate_cactus(
            ref(("R1", "CME", T, 10, 20), ("R2", "CME", T, 10, 10),
                ("F", "Flow", T, 10, 20)),
            frames(("D", T, 10, 20)))
        self.assertEqual(s["matched_events"], 1)
        self.assertEqual(s["unmatched_reference_events"], 1)
        self.assertEqual(p.loc[p.matched].iloc[0].catalog_event_id, "R1")
        self.assertNotIn("F", status.event_id.tolist())

    def test_zero_vs_unavailable_mean(self):
        p, _, s = validate_cactus(ref(("R", "CME", T, 180, 10)),
                                  frames(("D", T, 0, 10)))
        self.assertEqual(p.iloc[0].angular_iou, 0)
        self.assertIsNone(s["mean_matched_angular_iou"])
        p, _, s = validate_cactus(ref(("R", "CME", T, 0, 10)),
                                  frames(("D", "2010-12-12T09:00:00Z", 0, 10)))
        self.assertTrue(p.empty)
        self.assertIsNone(s["mean_matched_angular_iou"])

    def test_empty_inputs(self):
        p, _, s = validate_cactus(ref(), frames(("D", T, 0, 10)))
        self.assertTrue(p.empty)
        self.assertEqual(s["unmatched_detected_events"], 1)
        self.assertIsNone(s["mean_matched_angular_iou"])
        _, _, s = validate_cactus(ref(("R", "CME", T, 0, 10)), frames())
        self.assertEqual(s["unmatched_reference_events"], 1)
