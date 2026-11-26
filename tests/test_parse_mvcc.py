import json
import tempfile
import unittest
from pathlib import Path
from scripts.parse_mvcc import daily_events, normalize, parse_date

ROW = ["2287","A","1508.6938_0Ac","2010-12-12","05:24:00","FP","85","107","81",
       "266","-16","105","-11","notes","1508.7250_0Bc","1508.6938_0Ac",
       "A/2010/12/101212/1508.6938_0A/","3.03","5.15","-17.37","48.73",
       "20101212_053900.txt","mass.png"]


class TestMVCC(unittest.TestCase):
    def test_coverage(self):
        with self.assertRaisesRegex(ValueError, "REFERENCE_UNAVAILABLE"):
            parse_date("20240503")

    def test_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unmeasured = ROW.copy()
            unmeasured[2] = "otherA"
            (root/"monthly.json").write_text(json.dumps(
                {"recordsFiltered":2,"data":[ROW,unmeasured]}))
            (root/ROW[21]).write_text(
                "A id: 1508.6938_0A\nTimes for STA: 1\n"
                "Center of Mass Time Series in STA\nTime CoM\n"
                "20101212_052400 408.22 476.63 507.97 511.51 105.67 3.13 108.50 4.47 55.00\n\n"
                "Center of Mass Time Series in STB\n")
            (root/"1508.6938_0A.corset.ht").write_text(
                "#DETECTOR: STEREO_A-COR2\n#EVENT_ID: 1508.6938_0A\n"
                "   3.6542010-12-12T05:24:00.  81.0 COR2  1 -122   19\n")
            events, frames, edges = normalize(root, "20101212")
            self.assertEqual(len(events), 2)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames.iloc[0].front_rsun, 4.47)
            self.assertEqual(edges.iloc[0].position_angle, 81.0)
            self.assertEqual(events.iloc[1].measurement_status, "NOT_EVALUABLE")

    def test_incomplete_month(self):
        with self.assertRaises(ValueError):
            daily_events({"recordsFiltered":2,"data":[ROW]}, "20101212")
