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

    def test_rejects_unexpected_schema(self):
        with self.assertRaisesRegex(ValueError, "Malformed"):
            parse_monthly("2010/12/12 04:54:00 8 91 138\n", "20101212")

    def test_rejects_missing_date_records(self):
        with self.assertRaisesRegex(ValueError, "No SEEDS"):
            parse_monthly(SAMPLE, "20101213")


if __name__ == "__main__":
    unittest.main()
