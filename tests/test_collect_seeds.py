"""Offline tests of SEEDS download normalization; no HTTP requests."""
import unittest

from scripts.collect_seeds import normalize


class TestSeedsCollector(unittest.TestCase):
    def test_filters_date_and_strips_old_candidate_metrics(self):
        raw = (
            b"parent_event_id,seeds_timestamp,is_evaluable,seeds_bpa,seeds_width,"
            b"seeds_height,seeds_lead,angular_iou,candidate_cpa\n"
            b"S1,2010-12-12T05:24:00,True,359,30,4.1,4.4,0.99,100\n"
            b"S2,2010-12-13T05:24:00,True,60,20,4.3,4.7,0.77,70\n"
        )
        df = normalize(raw, "20101212")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["seeds_bpa"], 359)
        self.assertNotIn("angular_iou", df.columns)
        self.assertNotIn("candidate_cpa", df.columns)

    def test_rejects_html_and_event_only_csv(self):
        for content in (b"<html>server unavailable</html>",
                        b"event_id,time,pa,width\n1,2010-12-12,90,20\n"):
            with self.assertRaises(ValueError):
                normalize(content, "20101212")

    def test_rejects_bad_width(self):
        raw = (
            b"parent_event_id,seeds_timestamp,is_evaluable,seeds_bpa,seeds_width,seeds_height\n"
            b"S1,2010-12-12T05:24:00,True,100,800,4.2\n"
        )
        with self.assertRaisesRegex(ValueError, "width"):
            normalize(raw, "20101212")


if __name__ == "__main__":
    unittest.main()
