"""Offline SEEDS detail download tests with authentic monthly records."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_seeds_details import fetch_details


class TestFetchSeedsDetails(unittest.TestCase):
    def test_builds_exact_url_and_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monthly = root / "raw/2010/12/monthly.txt"
            monthly.parent.mkdir(parents=True)
            monthly.write_text(
                "2010/12/12 04:54:00   8  91 138   499     5.7\n"
                "2010/12/12 11:24:00   2 113  32  2901 -------\n",
                encoding="utf-8",
            )
            seen = []
            def fake_fetch(url, timeout=30):
                seen.append(url)
                return (b"# DATE_OBS: 2010/12/12\n", url)
            result = fetch_details(
                "20101212", raw_root=root / "raw",
                output_root=root / "details", fetch=fake_fetch,
            )
            self.assertEqual(result["detail_count"], 2)
            self.assertEqual(
                seen[0],
                "http://spaceweather.gmu.edu/seeds/secchi/detection_cor2/a/2010/12/"
                "20101212.045400.w138.v0499.p091.txt",
            )
            self.assertTrue(
                (root / "details/20101212/20101212.112400.w032.v2901.p113.txt").exists()
            )
            meta = json.loads(
                (root / "details/20101212/details_provenance.json").read_text()
            )
            self.assertEqual(len(meta["details"]), 2)
            self.assertEqual(meta["parser_status"], "DOWNLOADED_RAW_DETAILS_NOT_YET_PARSED")

    def test_failure_does_not_create_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monthly = root / "raw/2010/12/monthly.txt"
            monthly.parent.mkdir(parents=True)
            monthly.write_text(
                "2010/12/12 04:54:00 8 91 138 499 5.7\n"
                "2010/12/12 09:24:00 3 76 113 401 7.2\n",
                encoding="utf-8",
            )
            def fake_fetch(url, timeout=30):
                if ".092400." in url:
                    raise ConnectionError("offline")
                return (b"sample\n", url)
            with self.assertRaises(ConnectionError):
                fetch_details(
                    "20101212", raw_root=root / "raw",
                    output_root=root / "details", fetch=fake_fetch,
                )
            self.assertFalse((root / "details/20101212").exists())


if __name__ == "__main__":
    unittest.main()
