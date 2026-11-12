"""Offline detector tests for causal T24, C0/C3 hierarchy and unavailable states."""
import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from cme_pipeline.live_detection import (
    _c0_hierarchy, _pure_hierarchy, _threshold, run_detection,
)


class TestDetectorCore(unittest.TestCase):
    def test_robust_threshold(self):
        self.assertAlmostEqual(_threshold(np.ones(10)), 1.0)

    def test_hierarchy_is_intentionally_different(self):
        flat = {16: np.zeros((16, 16), bool),
                8: np.zeros((32, 32), bool),
                4: np.zeros((64, 64), bool)}
        flat[16][6, 6] = True
        flat[8][14, 14] = True  # p16 neighbor rather than same parent
        flat[4][28, 28] = True
        self.assertTrue(_c0_hierarchy(flat)[4][28, 28])
        self.assertFalse(_pure_hierarchy(flat)[4][28, 28])

    def test_warmup_and_requested_day_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "preprocessed"
            inp.mkdir()
            rng = np.random.default_rng(7)
            rd = np.stack([
                rng.normal(0, 1, (256, 256)).astype("f4")
                for _ in range(3)
            ])
            np.save(inp / "north_up_RD_DNs_per_second.npy", rd)
            np.save(inp / "north_up_pair_valid_masks.npy",
                    np.ones(rd.shape, dtype=bool))
            (inp / "preprocessing_metadata.json").write_text(json.dumps({
                "unit": "DN/s", "msb_applied": False, "requested_date": "20240511"
            }))
            rows = []
            start = datetime(2024, 5, 10, 0, 23, 54)
            for i in range(3):
                t = start + timedelta(hours=12 * i)
                rows.append({
                    "pair_index": i + 1,
                    "start_time": (t - timedelta(minutes=30)).isoformat(),
                    "end_time": t.isoformat(),
                    "date_str": int(t.strftime("%Y%m%d"))
                })
            with (inp / "01_rd_manifest.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            out = run_detection(inp, method="c3", requested_date="20240511")
            states = np.load(out / "patch_state_p04.npy")
            self.assertEqual(states.shape, (3, 64, 64))
            self.assertTrue(np.all(states[0] == -1))
            self.assertTrue(np.all(states[1] == -1))
            self.assertEqual(json.loads((out / "detection_metadata.json").read_text())["detector"], "c3")


if __name__ == "__main__":
    unittest.main()
