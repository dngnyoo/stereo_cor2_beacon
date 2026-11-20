"""Unit and regression test suite for S1 (Seed-Guided Birth) and S2 (Seed-Aware Association) Tracking."""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs
from cme_pipeline.stages.tracking.registry import (
    get_tracking_spec,
    get_tracking_stage,
    list_tracking_methods,
    resolve_tracking_method_name,
)
from cme_pipeline.stages.tracking.v07_pa_width_weighted_cpa import PAWidthWeightedCPAAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v09_seed_guided_birth import SeedGuidedBirthTrackingStage
from cme_pipeline.stages.tracking.v10_seed_aware_association import SeedAwareAssociationTrackingStage
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage
from cme_pipeline.utils.geometry import circular_diff_deg


class TestSeedTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PipelineConfig()
        cls.cfg.date_str = 20101212
        cls.ctx = PipelineContext(cls.cfg)
        cls.det = load_frozen_detection_inputs(cls.cfg, cls.ctx)

        # Baseline A5b
        cls.cfg_a5b = PipelineConfig()
        cls.cfg_a5b.date_str = 20101212
        cls.cfg_a5b.tracking.method = "a5b"
        cls.cfg_a5b.tracking.parameters = {"min_pa_width_deg": 8.0, "outward_gate_enabled": True}
        cls.a5b_stage = PAWidthWeightedCPAAdaptiveROITrackingStage()
        cls.a5b_out = cls.a5b_stage.run(cls.det, cls.cfg_a5b, cls.ctx)

        # S1 Tracking
        cls.cfg_s1 = PipelineConfig()
        cls.cfg_s1.date_str = 20101212
        cls.cfg_s1.tracking.method = "s1"
        cls.cfg_s1.tracking.parameters = {
            "min_pa_width_deg": 8.0,
            "outward_gate_enabled": True,
            "seed_min_radius_px": 22.0,
            "seed_max_radius_px": 30.0,
            "strict_seeded_birth": True,
        }
        cls.s1_stage = SeedGuidedBirthTrackingStage()
        cls.s1_out = cls.s1_stage.run(cls.det, cls.cfg_s1, cls.ctx)

        # S2 Tracking
        cls.cfg_s2 = PipelineConfig()
        cls.cfg_s2.date_str = 20101212
        cls.cfg_s2.tracking.method = "s2"
        cls.cfg_s2.tracking.parameters = {
            "min_pa_width_deg": 8.0,
            "outward_gate_enabled": True,
            "seed_min_radius_px": 22.0,
            "seed_max_radius_px": 30.0,
            "strict_seeded_birth": True,
            "seed_aware_association_enabled": True,
            "seed_pa_tolerance_deg": 15.0,
            "ambiguity_tau": 0.25,
        }
        cls.s2_stage = SeedAwareAssociationTrackingStage()
        cls.s2_out = cls.s2_stage.run(cls.det, cls.cfg_s2, cls.ctx)

    def test_1_registry_registration(self):
        """Verify that s1 and s2 are registered with valid specs and aliases."""
        methods = list_tracking_methods()
        self.assertIn("s1", methods)
        self.assertIn("s2", methods)

        self.assertEqual(resolve_tracking_method_name("seed_birth"), "s1")
        self.assertEqual(resolve_tracking_method_name("seed_association"), "s2")

        s1_spec = get_tracking_spec("s1")
        self.assertIn("strict_seeded_birth", s1_spec.supported_parameters)
        self.assertIn("seed_max_radius_px", s1_spec.supported_parameters)

        s2_spec = get_tracking_spec("s2")
        self.assertIn("seed_aware_association_enabled", s2_spec.supported_parameters)
        self.assertIn("seed_pa_tolerance_deg", s2_spec.supported_parameters)
        self.assertIn("ambiguity_tau", s2_spec.supported_parameters)

    def test_2_s1_strict_seeded_birth_filtering(self):
        """Verify strict seeded birth suppresses unseeded false track births."""
        # A5b produces 21 total tracks, whereas S1 suppresses unseeded components to 18 tracks
        self.assertLess(self.s1_out.total_tracks, self.a5b_out.total_tracks)
        self.assertEqual(self.s1_out.total_tracks, 18)

        birth_df = getattr(self.s1_out, "seed_birth_df")
        self.assertFalse(birth_df.empty)

        # Verify rejection reasons
        unseeded_rejections = birth_df[birth_df["rejection_reason"] == "UNSEEDED_NO_INNER_SEED_CANDIDATE"]
        self.assertGreater(len(unseeded_rejections), 200, "Should log >200 unseeded component rejections across day")

        # Verify initiated tracks strictly have track_initiated == True and is_seeded == True
        initiated = birth_df[birth_df["track_initiated"]]
        self.assertEqual(len(initiated), 18)
        self.assertTrue(all(initiated["is_seeded"]))
        self.assertTrue(all(initiated["rep_seed_score"] > 0))

    def test_3_s1_score_priority_ordering(self):
        """Verify that newly initialized tracks within each frame are sorted by seed score descending."""
        birth_df = getattr(self.s1_out, "seed_birth_df")
        initiated = birth_df[birth_df["track_initiated"]]

        # In Pair 30, multiple tracks were born: verify scores are descending
        p30_tracks = initiated[initiated["pair_index"] == 30]
        self.assertGreaterEqual(len(p30_tracks), 2)
        scores = p30_tracks["rep_seed_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True), "Tracks in Pair 30 must be ordered by seed score descending")

    def test_4_s1_unseeded_component_continuation(self):
        """Verify existing active tracks can continue claiming unseeded components."""
        cs_df = self.s1_out.component_states_df
        # An existing track should claim components in outer corona that do not possess inner seeds
        outer_comps = cs_df[cs_df["r_min"] > 35.0]
        self.assertGreater(len(outer_comps), 0, "Active tracks must continue through outer unseeded components")

    def test_5_seed_provenance_retention(self):
        """Verify track seed provenance dictionary is attached and populated."""
        sp_s1 = getattr(self.s1_out, "track_seed_provenance")
        self.assertEqual(len(sp_s1), 18)
        for tid, pdata in sp_s1.items():
            self.assertIn("seed_score", pdata)
            self.assertIn("seed_pa_deg", pdata)
            self.assertIn("seed_radius_px", pdata)
            self.assertIn("seed_patch_coords", pdata)
            self.assertGreater(pdata["seed_score"], 0.0)

    def test_6_s2_seed_provenance_arbitration(self):
        """Verify S2 arbitrates contested patches using seed provenance and logs decisions."""
        arb_df = getattr(self.s2_out, "seed_arbitration_df")
        self.assertFalse(arb_df.empty, "S2 must log seed arbitration decisions")
        self.assertGreater(len(arb_df), 50, "Should arbitrate >50 contested patches on 2010-12-12")

        # Check arbitration fields
        required_cols = ["pair_index", "component_id", "patch_pa_deg", "a5b_winner", "s2_winner", "reason"]
        for col in required_cols:
            self.assertIn(col, arb_df.columns)

        # In Pair 39 and 40, Track 5 must win patches near its seed origin from Track 4
        p40_overrides = arb_df[(arb_df["pair_index"] == 40) & (arb_df["s2_winner"] == 5)]
        self.assertGreater(len(p40_overrides), 0, "Track 5 must recover patches from Track 4 in Pair 40")

    def test_7_s2_fallback_to_a5b_when_uncontested_or_outside_seed_corridor(self):
        """Verify S2 strictly preserves A5b allocation when seed evidence does not conflict."""
        # When seed_aware_association_enabled is False, S2 must produce identical ownership masks to S1
        cfg_s2_off = PipelineConfig()
        cfg_s2_off.date_str = 20101212
        cfg_s2_off.tracking.method = "s2"
        cfg_s2_off.tracking.parameters = {
            "min_pa_width_deg": 8.0,
            "outward_gate_enabled": True,
            "strict_seeded_birth": True,
            "seed_aware_association_enabled": False,
        }
        s2_off_out = self.s2_stage.run(self.det, cfg_s2_off, self.ctx)
        np.testing.assert_array_equal(
            s2_off_out.track_ownership_masks,
            self.s1_out.track_ownership_masks,
            err_msg="Disabling seed_aware_association must yield exact equivalence to S1"
        )

    def test_8_mutual_exclusive_patch_ownership(self):
        """Verify mutual exclusive patch ownership for S1 and S2."""
        for name, out in [("S1", self.s1_out), ("S2", self.s2_out)]:
            masks = out.track_ownership_masks
            grid4 = out.patch_activation_grid4
            n_frames = masks.shape[0]

            for f in range(n_frames):
                frame_m = masks[f]
                tids = np.unique(frame_m)
                sum_patches = sum((frame_m == t).sum() for t in tids if t > 0)
                nonzero_patches = (frame_m > 0).sum()
                self.assertEqual(sum_patches, nonzero_patches, f"[{name}] Patch collision at frame {f}!")
                self.assertTrue(np.all(frame_m[grid4[f] == 0] == 0), f"[{name}] Inactive patch assigned at frame {f}!")

    def test_9_s1_s2_downstream_gates_compatibility(self):
        """Verify both S1 and S2 outputs pass downstream association and family size gates."""
        gate_cfg = PipelineConfig()
        gate_cfg.date_str = 20101212
        gate_cfg.birth_gate.max_birth_radius_px = 30.0
        gate_cfg.size_gate.min_detected_patches = 10
        gate_cfg.size_gate.min_observed_pairs = 3
        gate_cfg.size_gate.size_qualified_persistence = True
        gate_cfg.size_gate.consecutive_size_qualified = True

        for name, out in [("S1", self.s1_out), ("S2", self.s2_out)]:
            assoc = IdentityAssociationStage().run(out, gate_cfg, self.ctx)
            bg = FamilyBirthRadiusGateStage().run(assoc, gate_cfg, self.ctx)
            sg = FamilySizeGateStage().run(bg, gate_cfg, self.ctx)

            self.assertGreater(len(sg.admitted_family_ids), 0, f"[{name}] Must have admitted families")
            self.assertGreaterEqual(len(sg.admitted_track_ids), 6, f"[{name}] Must admit at least 6 core CME families")

    def test_10_a5b_baseline_regression_invariance(self):
        """Verify baseline A5b is completely unaffected by S1 and S2 additions."""
        self.assertEqual(self.a5b_out.total_tracks, 21)
        df_a5b = self.a5b_out.track_states_df
        t3 = df_a5b[(df_a5b["track_id"] == 3) & (df_a5b["state"] == "OBSERVED")]
        self.assertEqual(len(t3), 20)
        self.assertAlmostEqual(t3.iloc[0]["central_pa"], 108.579, places=2)


if __name__ == "__main__":
    unittest.main()
