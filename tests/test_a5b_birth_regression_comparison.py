"""Small offline fixture for the legacy/new A5b result comparator."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_a5b_birth_regression import compare_runs


class TestA5bBirthRegressionComparator(unittest.TestCase):
    def test_new_outer_track_matches_legacy_inner_track_without_id_assumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            det = root / "detector"
            det.mkdir()
            state = np.zeros((2, 64, 64), dtype=np.int8)
            state[:, 10, 10] = 1
            state[:, 40, 40] = 1
            np.save(det / "patch_state_p04.npy", state)
            pd.DataFrame({"pair_index": [1, 2], "end_time": ["2024-05-10T00:00:00", "2024-05-10T00:30:00"]}).to_csv(
                det / "frame_detection_summary.csv", index=False
            )
            old = np.zeros_like(state, dtype=np.int32)
            new = np.zeros_like(state, dtype=np.int32)
            old[:, 10, 10] = 1
            new[:, 10, 10] = 3  # same physical track, different ID
            new[:, 40, 40] = 2  # newly born outer track
            for dirname, own, births in (
                ("legacy", old, [(1, 20.0)]),
                ("new", new, [(2, 80.0), (3, 20.0)]),
            ):
                run = root / dirname
                (run / "arrays").mkdir(parents=True)
                (run / "tables").mkdir()
                np.save(run / "arrays" / "track_ownership_masks.npy", own)
                pd.DataFrame({"pair_index": [1, 2]}).to_csv(
                    run / "tables" / "live_observation_manifest.csv", index=False
                )
                pd.DataFrame([
                    {"track_id": tid, "pair_index": 1, "state": "OBSERVED", "birth_r_min_px": radius}
                    for tid, radius in births
                ]).to_csv(run / "tables" / "live_tracking_states.csv", index=False)
                (run / "tracking_metadata.json").write_text(json.dumps({
                    "method": "a5b", "detection_dir": str(det), "requested_date": "20240510"
                }))
            result = compare_runs(root / "legacy", root / "new", root / "comparison")
            self.assertEqual(result["legacy_tracks"], 1)
            self.assertEqual(result["new_tracks"], 2)
            self.assertEqual(result["old_coverage_lost"], 0)
            self.assertEqual(result["new_coverage_gained"], 2)
            self.assertEqual(result["new_birth_radius_above_threshold"], 1)
            matches = pd.read_csv(root / "comparison" / "track_matches.csv")
            self.assertEqual(int(matches.iloc[0]["legacy_id"]), 1)
            self.assertEqual(int(matches.iloc[0]["new_id"]), 3)
            self.assertEqual(float(matches.iloc[0]["temporal_iou"]), 1.0)


if __name__ == "__main__":
    unittest.main()
