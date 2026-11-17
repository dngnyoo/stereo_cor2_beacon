"""Regression tests from authentic 2010-12-12 GMU SEEDS detail sample."""
import tempfile
import unittest
from pathlib import Path

from scripts.parse_seeds_details import parse_detail, parse_directory


DETAIL = b"""#DATE_CREATED: 2011/09/01 21:43:25
#DATE_OBS: 2010/12/12
#TIME_OBS: 04:54:00
#DETECTOR: cor2A
#OBSERVER: SEEDS
#IMAGE_TYPE: RUNNING DIFFERENCE
#NUM_FRAMES: 8
#ONSET1: 2010/12/12 04:13:55
#ONSET2: 2010/12/12 03:59:40
#ONSET2_RSUN:  1.00
#CEN_PA:  91
#WIDTH: 138
#SPEED:   499
#ACCEL:     5.7
#REMARK:
#DATE   TIME     HMLEAD     HMHEIGHT  HMFOLLOW   TEL    PASTART PAEND   BPA
2010/12/12 04:54:00    3.07      3.00      3.00  cor2A      97.0   126.0   110.0
2010/12/12 05:24:00    3.87      3.40      3.00  cor2A      79.0   134.0   112.0
2010/12/12 05:54:00    5.15      4.68      3.87  cor2A      72.0   145.0   120.0
2010/12/12 06:24:00    6.42      5.48      5.15  cor2A      66.0   152.0   126.0
2010/12/12 06:54:00    7.89      6.96      6.49  cor2A      27.0   156.0    33.0
2010/12/12 07:24:00    9.23      7.96      7.42  cor2A      25.0   160.0   125.0
2010/12/12 07:54:00   10.44      9.10      8.43  cor2A      23.0   160.0   127.0
2010/12/12 08:24:00   11.92     10.58      9.91  cor2A      22.0   160.0   124.0
"""
NAME = "20101212.045400.w138.v0499.p091.txt"


class TestParseSeedsDetails(unittest.TestCase):
    def test_event_and_all_eight_frames(self):
        event, frames = parse_detail(DETAIL, NAME)
        self.assertEqual(event["event_id"], "SEEDS_COR2A_20101212T045400_W138_V0499_P091")
        self.assertEqual(event["NUM_FRAMES"], 8)
        self.assertEqual(event["ONSET2_RSUN"], 1.0)
        self.assertIsNone(event["REMARK"])
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0]["HMLEAD"], 3.07)
        self.assertEqual(frames[0]["BPA"], 110.)
        self.assertEqual(frames[4]["BPA"], 33.)
        self.assertEqual(frames[-1]["PAEND"], 160.)

    def test_num_frames_mismatch_fails(self):
        bad = DETAIL.replace(b"#NUM_FRAMES: 8", b"#NUM_FRAMES: 9")
        with self.assertRaisesRegex(ValueError, "NUM_FRAMES mismatch"):
            parse_detail(bad, NAME)

    def test_wrong_filename_fails(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            parse_detail(DETAIL, "20101212.045400.w137.v0499.p091.txt")

    def test_directory_output_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / NAME).write_bytes(DETAIL)
            events, frames = parse_directory(path, "20101212")
            self.assertEqual(events.shape, (1, 18))
            self.assertEqual(frames.shape, (8, 11))
            self.assertEqual(events.iloc[0]["DETECTOR"], "cor2A")


if __name__ == "__main__":
    unittest.main()
