"""Offline unit tests of GMU monthly endpoint and lossless archiving."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_seeds_monthly import FILES, acquire, month_url


class TestFetchSeedsMonthly(unittest.TestCase):
    def test_url_is_cor2_a_monthly(self):
        self.assertEqual(
            month_url("20101212", "monthly.txt"),
            "http://spaceweather.gmu.edu/seeds/secchi/detection_cor2/a/2010/12/monthly.txt",
        )
        with self.assertRaises(ValueError):
            month_url("20101212", "other.txt")

    def test_archives_raw_and_provenance_without_interpolation(self):
        seen = []
        def fake_fetch(url, timeout=30):
            seen.append(url)
            return (b"original unparsed text\n", url)

        with tempfile.TemporaryDirectory() as tmp:
            outcome = acquire("20101212", output_root=Path(tmp), fetch=fake_fetch)
            directory = Path(outcome["directory"])
            self.assertEqual(len(seen), 3)
            for name in FILES:
                self.assertEqual((directory / name).read_bytes(), b"original unparsed text\n")
            meta = json.loads((directory / "source_provenance.json").read_text())
            self.assertEqual(meta["parser_status"], "RAW_DOWNLOADED_SCHEMA_NOT_YET_VERIFIED")
            self.assertEqual(
                meta["files"]["monthly.txt"]["sha256"],
                hashlib.sha256(b"original unparsed text\n").hexdigest(),
            )
            with self.assertRaises(FileExistsError):
                acquire("20101212", output_root=Path(tmp), fetch=fake_fetch)

    def test_does_not_write_partial_download(self):
        def broken_fetch(url, timeout=30):
            if url.endswith("/daily.txt"):
                raise ConnectionError("unavailable")
            return (b"data\n", url)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConnectionError):
                acquire("20101212", output_root=Path(tmp), fetch=broken_fetch)
            self.assertFalse((Path(tmp) / "2010" / "12").exists())


if __name__ == "__main__":
    unittest.main()
