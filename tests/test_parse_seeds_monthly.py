"""SEEDS monthly TXT parser regression using a real user-provided sample."""
import unittest

from scripts.parse_seeds_monthly import parse_monthly

SAMPLE = """2010/12/11 20:24:00   2 280  30   207 -------
2010/12/12 04:54:00   8  91 138   499     5.7
2010/12/12 09:24:00   3  76 113   401     7.2
2010/12/12 11:24:00   2 113  32  2901 -------
2010/12/12 12:54:00   8 154 111   270    16.6
2010/12/12 13:54:00   4  40  39   248    21.6
2010/12/12 17:24:00   5 117  34   373     2.1
2010/12/14 15:54:00   2  39  74   647 -------
"""


class TestMonthlyParser(unittest.TestCase):
    def test_real_dec12_records_and_stable_ids(self):
        rows = parse_monthly(SAMPLE, "20101212")
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            rows.iloc[0]["event_id"],
            "SEEDS_COR2A_20101212T045400_W138_V0499_P091",
        )
        self.assertEqual(
            rows.iloc[2]["event_id"],
            "SEEDS_COR2A_20101212T112400_W032_V2901_P113",
        )
        self.assertEqual(rows.iloc[0]["observation_count"], 8)
        self.assertEqual(rows.iloc[2]["speed_km_s"], 2901.0)
        self.assertTrue(rows.iloc[2]["auxiliary_raw"].startswith("-"))

    def test_2024_unwrapped_position_angles_keep_source_ids(self):
        sample = (
            "2024/05/03 01:53:30   3  99 113    90     7.2\\n"
            "2024/05/03 03:53:30   3 399 123  1023   251.8\\n"
            "2024/05/03 06:53:30   2 374  71  1217 -------\\n"
        )
        df = parse_monthly(sample, "20240503")
        self.assertEqual(df.pa_deg.tolist(), [99, 399, 374])
        self.assertTrue(df.iloc[1].event_id.endswith("_P399"))
        self.assertTrue(df.iloc[2].event_id.endswith("_P374"))
        self.assertEqual([int(p) % 360 for p in df.pa_deg], [99, 39, 14])

    def test_rejects_pa_outside_three_digit_source_field(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            parse_monthly("2024/05/03 03:53:30 3 1000 123 1023 251.8", "20240503")

    def test_rejects_unexpected_schema(self):
        with self.assertRaisesRegex(ValueError, "Malformed"):
            parse_monthly("2010/12/12 04:54:00 8 91 138\n", "20101212")

    def test_rejects_missing_date_records(self):
        with self.assertRaisesRegex(ValueError, "No SEEDS"):
            parse_monthly(SAMPLE, "20101213")


if __name__ == "__main__":
    unittest.main()
