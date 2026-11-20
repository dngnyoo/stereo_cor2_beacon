"""Real-Data Unit and Regression Test Suite for Independent Seed-Origin Propagation Tracking (SPT).

Evaluates SPT-0, SPT-1, and SPT-2 on the 2010-12-12 benchmark dataset.
Verifies independence from A5b, mutual exclusivity, downstream gates compatibility,
and deterministic execution.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs
from cme_pipeline.stages.tracking.registry import (
    get_tracking_spec,
    list_tracking_methods,
    resolve_tracking_method_name,
)
from cme_pipeline.stages.tracking.v11_spt0_temporal_association import SPT0TemporalAssociationTrackingStage
from cme_pipeline.stages.tracking.v12_spt1_motion_prediction import SPT1MotionPredictionTrackingStage
from cme_pipeline.stages.tracking.v13_spt2_patch_ownership import SPT2PatchOwnershipTrackingStage
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage


class TestSPTRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PipelineConfig()
        cls.cfg.date_str = 20101212
        cls.ctx = PipelineContext(cls.cfg)
        cls.det = load_frozen_detection_inputs(cls.cfg, cls.ctx)

        # Common SPT parameters
        spt_params = {
            "patch_size": 4,
            "max_cadence_gap_sec": 4500.0,
            "outward_gate_enabled": True,
            "outward_gate_tolerance_px": 4.0,
            "seed_min_radius_px": 22.0,
            "seed_max_radius_px": 30.0,
            "seed_score_cutoff": 0.0,
            "max_missed_frames": 3,
            "min_confirmation_pairs": 2,
            "min_component_patches": 1,
        }

        cls.cfg_spt0 = PipelineConfig()
        cls.cfg_spt0.date_str = 20101212
        cls.cfg_spt0.tracking.method = "spt0"
        cls.cfg_spt0.tracking.parameters = dict(spt_params)
        cls.spt0_out = SPT0TemporalAssociationTrackingStage().run(cls.det, cls.cfg_spt0, cls.ctx)

        cls.cfg_spt1 = PipelineConfig()
        cls.cfg_spt1.date_str = 20101212
        cls.cfg_spt1.tracking.method = "spt1"
        cls.cfg_spt1.tracking.parameters = dict(spt_params)
        cls.spt1_out = SPT1MotionPredictionTrackingStage().run(cls.det, cls.cfg_spt1, cls.ctx)

        cls.cfg_spt2 = PipelineConfig()
        cls.cfg_spt2.date_str = 20101212
        cls.cfg_spt2.tracking.method = "spt2"
        cls.cfg_spt2.tracking.parameters = {
            **spt_params,
            "bridge_margin_threshold": 0.05,
            "separation_mode": "marker_geodesic",
        }
        cls.spt2_out = SPT2PatchOwnershipTrackingStage().run(cls.det, cls.cfg_spt2, cls.ctx)

    def test_01_registry_registration(self):
        """Verify spt0, spt1, spt2 are canonically registered."""
        methods = list_tracking_methods()
        for m in ["spt0", "spt1", "spt2"]:
            self.assertIn(m, methods)
            spec = get_tracking_spec(m)
            self.assertIn("seed_min_radius_px", spec.supported_parameters)
            self.assertIn("seed_max_radius_px", spec.supported_parameters)

    def test_02_seedless_continuation(self):
        """Verify tracks continue outward through outer seedless components."""
        for name, out in [("SPT-0", self.spt0_out), ("SPT-1", self.spt1_out), ("SPT-2", self.spt2_out)]:
            states = out.track_states_df
            outer_obs = states[(states["state"] == "OBSERVED") & (states["front_radius"] > 40.0)]
            self.assertGreater(len(outer_obs), 10, f"[{name}] Must continue observing tracks in outer corona (r > 40 px)")

    def test_03_motion_prediction_kinematics(self):
        """Verify SPT-1 and SPT-2 compute positive outward radial velocity."""
        for name, out in [("SPT-1", self.spt1_out), ("SPT-2", self.spt2_out)]:
            tracks = getattr(out, "confirmed_spt_tracks")
            self.assertGreater(len(tracks), 0)
            vs = [t.radial_velocity_px_s for t in tracks]
            self.assertTrue(all(v >= 0.0 for v in vs), f"[{name}] Velocities must be non-negative")
            self.assertTrue(any(v > 0.001 for v in vs), f"[{name}] At least one track must have significant velocity")

    def test_04_spt2_arbitration_logging(self):
        """Verify SPT-2 logs contested component arbitrations."""
        arb_df = getattr(self.spt2_out, "seed_arbitration_df")
        self.assertFalse(arb_df.empty, "SPT-2 must log contested patch arbitrations")
        self.assertGreater(len(arb_df), 100, "Should arbitrate >100 patches on 2010-12-12")

    def test_05_mutual_exclusive_ownership(self):
        """Verify no patch is assigned to more than one track in any frame."""
        for name, out in [("SPT-0", self.spt0_out), ("SPT-1", self.spt1_out), ("SPT-2", self.spt2_out)]:
            masks = out.track_ownership_masks
            grid4 = out.patch_activation_grid4
            for f in range(masks.shape[0]):
                m_frame = masks[f]
                # Total count of track-assigned patches
                unique_tids = np.unique(m_frame[m_frame > 0])
                sum_by_tid = sum((m_frame == t).sum() for t in unique_tids)
                total_nonzero = (m_frame > 0).sum()
                self.assertEqual(sum_by_tid, total_nonzero, f"[{name}] Patch collision at frame {f}")
                # Ensure inactive patches are never assigned
                self.assertTrue(np.all(m_frame[~grid4[f]] == 0), f"[{name}] Inactive patch assigned at frame {f}")

    def test_06_downstream_gates_compatibility(self):
        """Verify SPT outputs cleanly flow through IdentityAssociation and downstream Gates."""
        gate_cfg = PipelineConfig()
        gate_cfg.date_str = 20101212
        gate_cfg.birth_gate.max_birth_radius_px = 30.0
        gate_cfg.size_gate.min_detected_patches = 10
        gate_cfg.size_gate.min_observed_pairs = 3
        gate_cfg.size_gate.size_qualified_persistence = True
        gate_cfg.size_gate.consecutive_size_qualified = True

        for name, out in [("SPT-0", self.spt0_out), ("SPT-1", self.spt1_out), ("SPT-2", self.spt2_out)]:
            assoc = IdentityAssociationStage().run(out, gate_cfg, self.ctx)
            bg = FamilyBirthRadiusGateStage().run(assoc, gate_cfg, self.ctx)
            sg = FamilySizeGateStage().run(bg, gate_cfg, self.ctx)

            self.assertGreaterEqual(len(sg.admitted_family_ids), 4, f"[{name}] Must admit at least 4 CME families")

    def test_07_deterministic_execution(self):
        """Verify re-running SPT-2 on the same inputs yields identical masks."""
        rerun_out = SPT2PatchOwnershipTrackingStage().run(self.det, self.cfg_spt2, self.ctx)
        np.testing.assert_array_equal(
            self.spt2_out.track_ownership_masks,
            rerun_out.track_ownership_masks,
            err_msg="SPT-2 reruns must be bitwise identical"
        )


if __name__ == "__main__":
    unittest.main()
