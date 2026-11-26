"""Regression using observed December 2010 CACTus records."""
import unittest

from scripts.parse_cactus_monthly import parse_monthly

SAMPLE = """# Instrument: SECCHI| Detector: cor2
  first cor2: 2010/11/30  00:24:00.005 20101130_002400_04c2A.fts
  last  cor2: 2011/01/01  23:24:00.008 20110101_232400_04c2A.fts
# CME | t0 | dt0| pa | da | v | dv | minv| maxv| halo?
0031|2010/12/12 14:24|03|119|028|0328|0019|0312|0367|
0030|2010/12/12 12:24|01|040|014|0324|0086|0231|0480|
0029|2010/12/12 09:24|02|257|012|0162|0002|0158|0164|
0028|2010/12/12 06:24|06|194|094|0337|0048|0231|0446|II
0027|2010/12/12 05:24|02|354|034|0390|0037|0328|0462|
0026|2010/12/12 04:54|02|164|074|0337|0083|0150|0416|
0025|2010/12/12 04:54|03|279|012|0238|0036|0211|0312|
0024|2010/12/12 04:54|06|072|134|0462|0094|0277|0625|II
0001|2010/11/30 07:54|00|031|008|0249|0024|0204|0265|
# Flow| t0 | dt0| pa | da | v | dv | minv| maxv| halo?
0003|2010/12/12 05:54|00|335|024|0277|0000|0277|0277|
"""


class TestCactusParser(unittest.TestCase):
    def test_dec12_records(self):
        table = parse_monthly(SAMPLE, "201012")
        day = table[table.onset_utc.str.startswith("2010-12-12")]
        self.assertEqual(day.event_type.value_counts().to_dict(), {"CME": 8, "Flow": 1})
        row = day[day.original_event_id.eq("0024")].iloc[0]
        self.assertEqual(row.catalog_event_id, "CACTUS_COR2A_201012_CME_0024")
        self.assertEqual((row.cpa_deg, row.angular_width_deg, row.median_speed_km_s), (72, 134, 462))
        self.assertEqual(row.halo_flag, "II")
        self.assertEqual(len(table), 10)
        self.assertEqual(table.iloc[-1].liftoff_duration_hours, 0)

    def test_rejects_bad_rows(self):
        for raw in (SAMPLE.replace("|134|", "|999|"),
                    SAMPLE.replace("_04c2A.fts", "_04c2B.fts"),
                    SAMPLE + "broken row\n",
                    SAMPLE + SAMPLE.splitlines()[-1] + "\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_monthly(raw, "201012")

    def test_empty_cme_section(self):
        raw = SAMPLE.splitlines()[:4]
        self.assertTrue(parse_monthly("\n".join(raw), "201012").empty)
