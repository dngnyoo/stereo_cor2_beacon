"""Source-based geometry and three-height SEEDS overlay regression tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from scripts.visualize_seeds_overlay import (
    load_seeds, load_target_geometry, nearest_fits_radius, overlay_frame,
)


class TestSeedsPhysicalOverlay(unittest.TestCase):
    def test_source_frame_values_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "frames.csv"
            file.write_text(
                "DATE,TIME,BPA,PASTART,PAEND,HMLEAD,HMHEIGHT,HMFOLLOW,parent_event_id\n"
                "2010/12/12,06:54:00,33,27,156,7.89,6.96,6.49,S1\n"
            )
            df = load_seeds(file)
            self.assertEqual(df.loc[0, "seeds_pa_start"], 27)
            self.assertEqual(df.loc[0, "seeds_pa_end"], 156)
            self.assertEqual(df.loc[0, "seeds_lead"], 7.89)
            self.assertEqual(df.loc[0, "seeds_height"], 6.96)
            self.assertEqual(df.loc[0, "seeds_follow"], 6.49)

    def test_run_wcs_center_and_arcsec_per_pixel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "acquisition/preprocessed"
            folder.mkdir(parents=True)
            w = WCS(naxis=2)
            w.wcs.crpix = [127.618033, 128.502729]
            w.wcs.crval = [0., 0.]
            w.wcs.cdelt = [.032666666666667, .032666666666667]
            w.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
            w.wcs.cunit = ["deg", "deg"]
            (folder / "target_wcs_header.txt").write_text(w.to_header().tostring(sep="\n"))
            sx, sy, scale = load_target_geometry(root)
            self.assertAlmostEqual(sx, 126.618033, places=4)
            self.assertAlmostEqual(sy, 127.502729, places=4)
            self.assertAlmostEqual(scale, 117.6, places=5)

    def test_fits_radius_and_missing_data_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "20101212"
            folder.mkdir()
            path = folder / "20101212_122424_d7c2A.fts"
            header = fits.Header()
            header["DETECTOR"] = "COR2"
            header["OBSRVTRY"] = "STEREO_A"
            header["RSUN"] = 992.635293319
            fits.PrimaryHDU(data=np.zeros((2, 2), dtype="float32"),
                            header=header).writeto(path)
            radius, actual = nearest_fits_radius(
                pd.Timestamp("2010-12-12T12:24:24Z"), root, 117.6)
            self.assertEqual(actual, path)
            self.assertAlmostEqual(radius, 992.635293319 / 117.6, places=7)
            with self.assertRaises(FileNotFoundError):
                nearest_fits_radius(
                    pd.Timestamp("2010-12-12T13:54:24Z"), root, 117.6,
                    max_delta_sec=60)

    def test_three_height_overlay_does_not_change_original_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pair_036_20101212T065424Z_families.png"
            target = root / "overlay.png"
            Image.new("RGB", (864, 840), "black").save(source)
            original = source.read_bytes()
            seeds = pd.DataFrame([{
                "parent_event_id": "S1",
                "time": pd.Timestamp("2010-12-12T06:54:00Z"),
                "seeds_bpa": 33., "seeds_width": 129.,
                "seeds_pa_start": 27., "seeds_pa_end": 156.,
                "seeds_height": 6.96, "seeds_lead": 7.89,
                "seeds_follow": 6.49,
            }])
            count = overlay_frame(source, target, seeds, rsun_px=992.635293319 / 117.6)
            self.assertEqual(count, 1)
            self.assertEqual(source.read_bytes(), original)
            with Image.open(target) as result:
                self.assertEqual(result.size, (864, 840))
                self.assertNotEqual(result.getpixel((432, 428)), None)


if __name__ == "__main__":
    unittest.main()
