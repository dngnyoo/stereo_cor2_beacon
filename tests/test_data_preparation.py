"""Offline tests for the date/local image-input front end."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from astropy.io import fits
import numpy as np

from cme_pipeline.data_preparation import (
    inspect_fits, list_fits, prepare_source, validate_date,
)


class TestDataPreparation(unittest.TestCase):
    def test_date_is_strict(self):
        self.assertEqual(validate_date("20240510"), "20240510")
        for bad in ("20241301", "20240230", "2024-05-10", "2024051", "../DATA"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_date(bad)

    def test_local_fits_inventory_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "observations"
            image_dir.mkdir()
            for i in (1, 0):
                header = fits.Header()
                header["DATE-OBS"] = f"2024-05-10T0{i}:00:00"
                header["DETECTOR"] = "COR2"
                header["OBSRVTRY"] = "STEREO_A"
                header["EXPTIME"] = 2.0
                header["CRPIX1"] = 8.0
                header["CRPIX2"] = 8.0
                header["CDELT1"] = 1.0
                header["CDELT2"] = 1.0
                fits.writeto(image_dir / f"im_{i}.fts", np.ones((16, 16)), header)
            manifest = prepare_source(
                date="20240510", source="local", product="science_raw",
                input_dir=image_dir, output_dir=root / "prep",
            )
            obj = json.loads(manifest.read_text())
            self.assertEqual(obj["n_files"], 2)
            self.assertEqual(obj["status"], "INVENTORIED_NOT_PREPROCESSED")
            self.assertTrue(all(v["has_wcs"] for v in obj["files"]))
            self.assertEqual(obj["files"][0]["date_obs"], "2024-05-10T00:00:00")
            self.assertEqual(obj["files"][0]["shape"], [16, 16])

    def test_invalid_date_in_image_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.fts"
            header = fits.Header()
            header["DATE-OBS"] = "2024-05-11T00:00:00"
            fits.writeto(path, np.ones((3, 3)), header)
            with self.assertRaisesRegex(ValueError, "date mismatch"):
                inspect_fits([path], expected_date="20240510")

    def test_no_images_does_not_silently_succeed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No FITS"):
                list_fits(Path(tmp))


if __name__ == "__main__":
    unittest.main()
