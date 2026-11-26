"""Offline tests for lossless CACTus acquisition and existing-file reuse."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.fetch_cactus_monthly import acquire, month_url

RAW = b"""# Instrument: SECCHI| Detector: cor2
  first cor2: 2010/11/30 00:24:00.005 20101130_002400_04c2A.fts
# CME | t0 | dt0| pa | da | v | dv | minv| maxv| halo?
0024|2010/12/12 04:54|06|072|134|0462|0094|0277|0625|II
"""


class TestFetchCactus(unittest.TestCase):
    def test_url(self):
        self.assertEqual(month_url("20101212"),
                         "https://secchi.nrl.navy.mil/cactus/SECCHI-A/2010/12/out/cmecat.txt")

    def test_download_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fake(url, timeout=30):
                return RAW, url
            result = acquire("20101212", output_root=root, fetch=fake)
            dest = Path(result["directory"])
            self.assertEqual((dest / "cmecat.txt").read_bytes(), RAW)
            meta = json.loads((dest / "source_provenance.json").read_text())
            self.assertEqual(meta["files"]["cmecat.txt"]["sha256"], hashlib.sha256(RAW).hexdigest())
            with self.assertRaises(FileExistsError):
                acquire("20101212", output_root=root, fetch=fake)
            acquire("20101212", output_root=root, reuse_existing=True, fetch=None)
            (dest / "cmecat.txt").write_bytes(RAW + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                acquire("20101212", output_root=root, reuse_existing=True, fetch=None)

    def test_adopt_existing_and_invalid_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "2010/12"
            dest.mkdir(parents=True)
            (dest / "cmecat.txt").write_bytes(RAW)
            acquire("20101212", output_root=root, reuse_existing=True, fetch=None)
            meta = json.loads((dest / "source_provenance.json").read_text())
            self.assertIsNone(meta["files"]["cmecat.txt"]["resolved_url"])
            with self.assertRaises(ValueError):
                acquire("20240503", output_root=root, fetch=lambda url, timeout: (b"<html>", url))
            self.assertFalse((root / "2024/05").exists())
