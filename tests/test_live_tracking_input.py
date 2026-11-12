"""Live detector-to-tracker contract tests, no NASA download required."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from cme_pipeline.live_tracking_input import load_live_tracking_input, observation_usable


class TestLiveTrackingInput(unittest.TestCase):
    def _case(self, folder: Path):
        prep = folder / "preprocessed"
        det = folder / "detection_c3_gaussian"
        prep.mkdir()
        det.mkdir()
        w = WCS(naxis=2)
        w.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
        w.wcs.cunit = ["arcsec", "arcsec"]
        w.wcs.crpix = [128.5, 128.5]
        w.wcs.crval = [0., 0.]
        w.wcs.cdelt = [-30., 30.]
        (prep / "target_wcs_header.txt").write_text(w.to_header().tostring(sep="\n"))
        np.save(prep / "north_up_RD_DNs_per_second.npy", np.ones((3, 256, 256), dtype="f4"))
        np.save(prep / "north_up_pair_valid_masks.npy", np.ones((3, 256, 256), dtype=bool))
        manifest = pd.DataFrame({
            "pair_index": [1, 2, 3], "date_str": [20240509, 20240510, 20240510],
            "start_time": ["2024-05-09T23:00:00", "2024-05-10T00:00:00", "2024-05-10T00:30:00"],
            "end_time": ["2024-05-09T23:30:00", "2024-05-10T00:30:00", "2024-05-10T01:00:00"],
            "delta_t_min": [30., 30., 30.]
        })
        manifest.to_csv(prep / "01_rd_manifest.csv", index=False)
        states = np.full((3, 64, 64), -1, dtype="i1")
        states[1] = 0
        states[1, 20, 20] = 1
        np.save(det / "patch_state_p04.npy", states)
        comps = np.full((3, 64, 64), -1, dtype="i4")
        comps[1] = 0
        comps[1, 20, 20] = 1
        np.save(det / "patch_component_labels.npy", comps)
        pd.DataFrame({
            "pair_index": [1, 2, 3], "end_time": manifest["end_time"],
            "status": ["WARMUP", "VALID", "GLOBAL_ACTIVATION"],
            "usable_detection": [False, True, False],
            "in_requested_date": [False, True, True],
        }).to_csv(det / "frame_detection_summary.csv", index=False)
        (det / "detection_metadata.json").write_text(json.dumps({
            "input_dir": str(prep), "requested_date": "20240510",
            "detector": "c3", "denoiser": "gaussian"
        }))
        return det

    def test_unavailable_is_not_empty_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            det = self._case(Path(temp))
            output = load_live_tracking_input(det)
            self.assertEqual(output.manifest_df["usable_detection"].tolist(),
                             [False, True, False])
            self.assertEqual(output.raw_component_labels[0, 0, 0], -1)
            self.assertFalse(output.patch_activation_grid4[0].any())
            self.assertFalse(observation_usable(output, 0))
            self.assertTrue(observation_usable(output, 1))
            self.assertEqual(output.dataset_metadata["n_unavailable"], 2)

    def test_reject_unavailable_inactive_zero_filled_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            det = self._case(Path(temp))
            path = det / "patch_state_p04.npy"
            states = np.load(path)
            states[0] = 0
            np.save(path, states)
            with self.assertRaisesRegex(ValueError, "Unavailable observations"):
                load_live_tracking_input(det)


if __name__ == "__main__":
    unittest.main()
