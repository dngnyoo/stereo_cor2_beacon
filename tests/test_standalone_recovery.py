"""Regression tests for independent CLI-stage Recovery from saved Gate decisions."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from cme_pipeline.run_recovery_data import run_saved_recovery


class TestStandaloneRecovery(unittest.TestCase):
    def test_requires_pure_gate_run(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "gates"
            run.mkdir()
            (run / "gates_summary.json").write_text(json.dumps({
                "recovery_executed": True,
            }))
            with self.assertRaisesRegex(ValueError, "pure Gate run"):
                run_saved_recovery(run, run_name="irrelevant", recovery_inner_radius_px=22.0)

    def test_consumes_saved_decisions_without_reexecuting_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            detector = root / "detection"
            detector.mkdir()
            trackdir = root / "tracking"
            (trackdir / "arrays").mkdir(parents=True)
            (trackdir / "tables").mkdir()
            gates = root / "gates"
            (gates / "stages" / "size_gate").mkdir(parents=True)

            (trackdir / "tracking_metadata.json").write_text(json.dumps({
                "method": "a5b", "requested_date": "20240510",
                "detection_dir": str(detector), "total_tracks": 2,
            }))
            (gates / "gates_summary.json").write_text(json.dumps({
                "recovery_executed": False,
                "source_tracking_run": str(trackdir),
                "source_detection_dir": str(detector),
                "date": "20240510", "birth_max_radius_px": 30.0,
                "min_detected_patches": 10, "min_observed_pairs": 3,
                "candidate_families": 2, "size_gate_admitted": 1,
            }))
            pd.DataFrame([
                {"family_id": "FAM_T01", "status": "ADMITTED",
                 "anchor_count": 1, "rejection_reason": "NONE"},
                {"family_id": "FAM_T02", "status": "REJECTED",
                 "anchor_count": 1, "rejection_reason": "SIZE"},
            ]).to_csv(gates / "stages" / "size_gate" / "family_summary.csv", index=False)

            owned = np.zeros((1, 64, 64), dtype=np.int32)
            owned[0, 0, 0] = 1
            owned[0, 0, 1] = 2
            np.save(trackdir / "arrays" / "track_ownership_masks.npy", owned)
            pd.DataFrame([
                {"pair_index": 1, "track_id": 1, "state": "OBSERVED",
                 "birth_r_min_px": 20, "central_pa": 10.0, "front_radius": 30.0},
                {"pair_index": 1, "track_id": 2, "state": "OBSERVED",
                 "birth_r_min_px": 20, "central_pa": 20.0, "front_radius": 30.0},
            ]).to_csv(trackdir / "tables" / "live_tracking_states.csv", index=False)

            detected = SimpleNamespace(
                patch_activation_grid4=owned > 0,
                raw_component_labels=owned.copy(),
                manifest_df=pd.DataFrame({"pair_index": [1]}),
                frame_cache=[{}], geometry=SimpleNamespace(occulter_inner_radius_px=22.0),
                dataset_metadata={},
            )
            with patch("cme_pipeline.run_recovery_data.load_live_tracking_input", return_value=detected), \
                 patch("cme_pipeline.run_recovery_data.adapt_tracking_output", side_effect=lambda obj, **kwargs: obj), \
                 patch("cme_pipeline.run_recovery_data.ApplyRecoveryAfterGatesStage") as rec_stage, \
                 patch("cme_pipeline.run_recovery_data.PipelineContext") as context:
                context.return_value.run_dir = root / "output"
                (root / "output").mkdir()
                rec_stage.return_value.run.return_value = SimpleNamespace(
                    total_recovered_patches=9,
                )
                with patch("cme_pipeline.stages.gates.v01_family_size_gate.FamilySizeGateStage.run",
                           side_effect=AssertionError("Gate was re-executed")), \
                     patch("cme_pipeline.stages.gates.v01_birth_radius_gate.FamilyBirthRadiusGateStage.run",
                           side_effect=AssertionError("Birth Gate was re-executed")):
                    run_saved_recovery(gates, run_name="test_standalone_recovery",
                                       recovery_inner_radius_px=22.0)
                passed = rec_stage.return_value.run.call_args[0][0]
                self.assertEqual(passed.admitted_track_ids, [1])
                self.assertEqual(passed.admitted_family_ids, ["FAM_T01"])
                self.assertEqual(json.loads(
                    (root / "output" / "recovery_summary.json").read_text()
                )["gate_decisions_recomputed"], False)


if __name__ == "__main__":
    unittest.main()
