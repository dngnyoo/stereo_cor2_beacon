"""The Family Gate must never execute Recovery; execution is a separate stage."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.stage import AssociationOutput, BirthGateOutput, TrackingOutput
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage
from cme_pipeline.stages.recovery.v02_apply_after_gates import ApplyRecoveryAfterGatesStage


class TestIndependentGateAndRecovery(unittest.TestCase):
    def test_gate_only_then_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ("stages", "tables", "arrays"):
                (p / name).mkdir()
            ctx = SimpleNamespace(
                stages_dir=p / "stages", tables_dir=p / "tables",
                arrays_dir=p / "arrays", info=lambda _: None,
            )
            own = np.zeros((3, 64, 64), dtype=np.int32)
            own[:, 10, 10] = 1
            own[:, 10, 11] = 1
            man = pd.DataFrame({"pair_index": [1, 2, 3]})
            states = pd.DataFrame([
                {"pair_index": pair, "track_id": 1, "state": "OBSERVED",
                 "component_id": 1} for pair in [1, 2, 3]
            ])
            tr = TrackingOutput(
                manifest_df=man, track_states_df=states,
                patch_component_labels=np.zeros_like(own),
                patch_activation_grid4=own > 0, frame_cache=[{}] * 3,
                geometry=None, total_tracks=1, track_ownership_masks=own,
            )
            assoc = AssociationOutput(
                tracking_output=tr, recovery_output=None,
                track_to_family={1: "FAM_T01"}, family_members={"FAM_T01": [1]},
                family_anchors={"FAM_T01": [1]},
                edges_df=pd.DataFrame(), track_membership_df=pd.DataFrame(),
                candidate_family_count=1,
            )
            birth = BirthGateOutput(
                association_output=assoc, birth_gated_family_ids=["FAM_T01"]
            )
            cfg = PipelineConfig()
            cfg.size_gate.min_detected_patches = 2
            cfg.size_gate.min_observed_pairs = 3
            with patch(
                "cme_pipeline.stages.recovery.v01_signed_pa_recovery."
                "run_dynamic_patch_recovery_for_tracks",
                side_effect=AssertionError("Gate invoked Recovery unexpectedly"),
            ):
                gated = FamilySizeGateStage().run(birth, cfg, ctx)
            self.assertEqual(gated.admitted_family_ids, ["FAM_T01"])
            self.assertIsNone(gated.recovered_patch_masks)
            self.assertEqual(gated.total_recovered_patches, 0)
            self.assertFalse((p / "arrays" / "recovered_patch_masks_pipeline_b_full_day.npy").exists())
            fake = np.zeros_like(own, dtype=bool)
            fake[:, 10, 10] = True
            with patch(
                "cme_pipeline.stages.recovery.v02_apply_after_gates."
                "DynamicPARecoveryRunner",
                return_value=lambda ids: fake.copy(),
            ) as runner_factory:
                recovered = ApplyRecoveryAfterGatesStage().run(gated, cfg, ctx)
            runner_factory.assert_called_once()
            self.assertEqual(recovered.total_recovered_patches, 3)
            self.assertTrue(np.array_equal(recovered.recovered_patch_masks, fake))
            self.assertIn("rec_FAM_T01", recovered.family_recovered_patch_masks)
            self.assertTrue((p / "arrays" / "recovered_patch_masks_pipeline_b_full_day.npy").is_file())
            self.assertIsNone(gated.recovered_patch_masks)


if __name__ == "__main__":
    unittest.main()
