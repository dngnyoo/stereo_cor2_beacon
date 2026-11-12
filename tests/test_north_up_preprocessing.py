"""Offline invariants for No-MSB COR2 calibration and WCS target construction."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from cme_pipeline.north_up_preprocessing import calibrate_no_msb, target_north_up_wcs


def wcs_header(n=256):
    w = WCS(naxis=2)
    w.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    w.wcs.cunit = ["arcsec", "arcsec"]
    w.wcs.crpix = [n / 2 + .5, n / 2 + .5]
    w.wcs.crval = [0, 0]
    w.wcs.cdelt = [-30., 30.]
    hdr = w.to_header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = n
    hdr["NAXIS2"] = n
    hdr["EXPTIME"] = 2.0
    return hdr


class TestNorthUpPreprocessing(unittest.TestCase):
    def test_calibration_keeps_dn_per_second(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.fts"
            hdr = wcs_header(16)
            fits.writeto(path, np.full((16, 16), 20., dtype=np.float32), hdr)
            def trim(data, header, silent=True):
                return data, header
            def bias(header, silent=True):
                return 4.
            data, out_hdr = calibrate_no_msb(
                path, trim_function=trim, bias_function=bias
            )
            self.assertTrue(np.allclose(data, 8.))
            self.assertEqual(out_hdr["BUNIT"], "DN/s")
            self.assertFalse(out_hdr["MSBAPP"])

    def test_north_up_target_has_north_y_and_west_x(self):
        hdr = wcs_header()
        target = target_north_up_wcs([hdr])
        self.assertLess(target.wcs.cdelt[0], 0)
        self.assertGreater(target.wcs.cdelt[1], 0)
        self.assertTrue(np.allclose(target.wcs.pc, np.eye(2)))
        self.assertTrue(np.allclose(target.wcs.crpix, [128.5, 128.5]))

    def test_invalid_exposure_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.fts"
            hdr = wcs_header(16)
            hdr["EXPTIME"] = 0.
            fits.writeto(path, np.ones((16, 16)), hdr)
            with self.assertRaisesRegex(ValueError, "EXPTIME"):
                calibrate_no_msb(
                    path,
                    trim_function=lambda d, h, silent=True: (d, h),
                    bias_function=lambda h, silent=True: 0.,
                )


if __name__ == "__main__":
    unittest.main()
