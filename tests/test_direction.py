"""Comprehensive Unit Tests for Birth-Anchored Direction Model and Gating."""

import unittest
from datetime import datetime, timedelta
import numpy as np

from cme_pipeline.core.direction import (
    compute_solar_root_pa,
    create_birth_direction_reference,
    update_birth_direction_reference,
)
from cme_pipeline.core.stage import BirthDirectionReference
from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg


class TestBirthDirectionModel(unittest.TestCase):
    def test_circular_mean_wraparound(self):
        """Test circular mean on S^1 around 0/360 degrees."""
        # 350 deg and 10 deg -> should average to 0.0 deg (360.0 % 360.0)
        mean_angle = circular_mean_deg(np.array([350.0, 10.0]))
        self.assertAlmostEqual(mean_angle, 0.0, places=5)

        # 355 deg, 0 deg, 5 deg -> 0.0 deg
        mean_3 = circular_mean_deg(np.array([355.0, 0.0, 5.0]))
        self.assertAlmostEqual(mean_3, 0.0, places=5)

        # Opposite angles (e.g. 0 deg and 180 deg) - test graceful handling
        mean_opp = circular_mean_deg(np.array([0.0, 180.0]))
        self.assertTrue(np.isfinite(mean_opp))

    def test_two_frame_causal_locking(self):
        """Test two-frame causal locking: Frame 1 provisional, Frame 2 confirmed."""
        t0 = datetime(2010, 12, 12, 5, 24, 0)
        t1 = t0 + timedelta(minutes=30)

        # Frame 1: Birth observation near solar surface (r_min = 20.0 px <= 30 px)
        ref = create_birth_direction_reference(
            track_id=15,
            birth_pair_id=30,
            birth_time=t0,
            birth_r_min_px=20.0,
            birth_pa=100.0,
            central_pa=100.0,
            angular_extent=25.0,
            anchor_threshold_px=30.0,
        )

        self.assertEqual(ref.track_id, 15)
        self.assertTrue(ref.is_anchor)
        self.assertEqual(ref.status, "PROVISIONAL")
        self.assertIsNone(ref.confirmation_time)
        self.assertAlmostEqual(ref.reference_cpa, 100.0, places=3)

        # Frame 2: Second consecutive observation at t1 (30 min later, within 75 min)
        # CPA_2 = 110.0 deg (difference = 10.0 deg <= 45.0 deg)
        ref_locked = update_birth_direction_reference(
            ref=ref,
            current_pair_id=31,
            current_time=t1,
            current_cpa=110.0,
            is_consecutive=True,
            max_lock_deviation_deg=45.0,
        )

        self.assertEqual(ref_locked.status, "LOCKED")
        self.assertEqual(ref_locked.confirmation_time, t1)
        self.assertAlmostEqual(ref_locked.cpa_first, 100.0, places=3)
        self.assertAlmostEqual(ref_locked.cpa_second, 110.0, places=3)
        # S^1 circular mean of 100 and 110 is 105.0
        self.assertAlmostEqual(ref_locked.reference_cpa, 105.0, places=3)
        self.assertTrue(any("LOCKED_CIRCULAR_MEAN" in flag for flag in ref_locked.flags))

    def test_reference_immutability_once_locked(self):
        """Test that once locked, subsequent frames cannot mutate the reference direction."""
        t0 = datetime(2010, 12, 12, 5, 24, 0)
        t1 = t0 + timedelta(minutes=30)
        t2 = t0 + timedelta(minutes=60)
        t3 = t0 + timedelta(minutes=90)

        ref = create_birth_direction_reference(
            track_id=15, birth_pair_id=30, birth_time=t0,
            birth_r_min_px=20.0, birth_pa=100.0, central_pa=100.0, angular_extent=20.0
        )
        ref_locked = update_birth_direction_reference(
            ref=ref, current_pair_id=31, current_time=t1, current_cpa=110.0,
            is_consecutive=True
        )
        self.assertEqual(ref_locked.status, "LOCKED")
        locked_cpa = ref_locked.reference_cpa

        # Frame 3: Track moves, instantaneous CPA shifts to 135 deg
        ref_frame3 = update_birth_direction_reference(
            ref=ref_locked, current_pair_id=32, current_time=t2, current_cpa=135.0,
            is_consecutive=True
        )
        self.assertAlmostEqual(ref_frame3.reference_cpa, locked_cpa, places=5)

        # Frame 4: Instantaneous CPA shifts to 80 deg
        ref_frame4 = update_birth_direction_reference(
            ref=ref_frame3, current_pair_id=33, current_time=t3, current_cpa=80.0,
            is_consecutive=True
        )
        self.assertAlmostEqual(ref_frame4.reference_cpa, locked_cpa, places=5)
        self.assertEqual(ref_frame4.status, "LOCKED")

    def test_observation_gap_boundary(self):
        """Test that observation gaps > 75 min terminate/unconfirm provisional reference."""
        t0 = datetime(2010, 12, 12, 0, 54, 0)
        t_gap = t0 + timedelta(minutes=270)  # 4.5 hour gap (Pair 29 to Pair 30)

        ref = create_birth_direction_reference(
            track_id=1, birth_pair_id=29, birth_time=t0,
            birth_r_min_px=22.0, birth_pa=90.0, central_pa=90.0, angular_extent=20.0
        )

        # Gap present: is_consecutive = False
        ref_after_gap = update_birth_direction_reference(
            ref=ref, current_pair_id=30, current_time=t_gap, current_cpa=95.0,
            is_consecutive=False
        )

        self.assertEqual(ref_after_gap.status, "UNCONFIRMED")
        self.assertIn("NON_CONSECUTIVE_CADENCE_GAP", ref_after_gap.flags)

    def test_no_future_leakage(self):
        """Test strict causal processing: references at step k depend only on past/current steps."""
        t0 = datetime(2010, 12, 12, 5, 24, 0)
        t1 = t0 + timedelta(minutes=30)

        ref_k = create_birth_direction_reference(
            track_id=10, birth_pair_id=1, birth_time=t0,
            birth_r_min_px=15.0, birth_pa=45.0, central_pa=45.0, angular_extent=10.0
        )
        # At step k, ref_k has no knowledge of future t1 or cpa at t1
        self.assertIsNone(ref_k.cpa_second)
        self.assertIsNone(ref_k.confirmation_time)
        self.assertEqual(ref_k.reference_cpa, 45.0)

    def test_directional_tracking_gating_logic(self):
        """Test candidate gating with directional angular difference tolerance <= 30 deg."""
        ref_cpa = 100.0

        # Candidate 1: CPA 115.0 deg -> diff = 15.0 deg <= 30.0 deg (ACCEPTED)
        diff_1 = abs(circular_diff_deg(115.0, ref_cpa))
        self.assertLessEqual(diff_1, 30.0)

        # Candidate 2: CPA 140.0 deg -> diff = 40.0 deg > 30.0 deg (REJECTED if no overlap)
        diff_2 = abs(circular_diff_deg(140.0, ref_cpa))
        self.assertGreater(diff_2, 30.0)

    def test_independent_anchor_separation_logic(self):
        """Test that non-collinear anchors (Delta CPA > 45 deg) separate into independent families."""
        parent_cpa = 105.0  # East CME
        child_cpa_collinear = 112.0  # East CME flank (diff = 7 deg <= 45 deg)
        child_cpa_non_collinear = 8.0  # North streamer / micro-eruption (diff = 97 deg > 45 deg)

        diff_collinear = abs(circular_diff_deg(child_cpa_collinear, parent_cpa))
        self.assertLessEqual(diff_collinear, 45.0)

        diff_non_collinear = abs(circular_diff_deg(child_cpa_non_collinear, parent_cpa))
        self.assertGreater(diff_non_collinear, 45.0)


if __name__ == "__main__":
    unittest.main()
