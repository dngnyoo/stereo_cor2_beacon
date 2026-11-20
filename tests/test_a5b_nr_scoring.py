"""Unit tests for redesigned normalized scoring (A5b-NR)."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

import numpy as np

from cme_pipeline.stages.tracking.a5b_nr_scoring import (
    circular_diff_deg,
    compute_angular_consistency,
    compute_detection_confidence,
    compute_kinematic_consistency,
    compute_nr_score,
    compute_spatial_continuity,
)


class TestA5bNRScoring(unittest.TestCase):
    def test_spatial_continuity_direct_overlap(self):
        m1 = np.zeros((64, 64), dtype=bool)
        m2 = np.zeros((64, 64), dtype=bool)
        m1[10:20, 10:20] = True  # 100 patches
        m2[10:20, 10:20] = True  # identical mask -> Dice = 1.0

        c, diag = compute_spatial_continuity(m1, m2)
        self.assertEqual(diag["category"], "DIRECT_OVERLAP")
        self.assertAlmostEqual(c, 1.0, places=5)
        self.assertAlmostEqual(diag["dice"], 1.0, places=5)

        # Partial direct overlap
        m3 = np.zeros((64, 64), dtype=bool)
        m3[15:25, 10:20] = True  # 50 overlapping patches, total 100 + 100 = 200 -> Dice = 0.5
        c_part, diag_part = compute_spatial_continuity(m1, m3)
        self.assertEqual(diag_part["category"], "DIRECT_OVERLAP")
        self.assertAlmostEqual(diag_part["dice"], 0.5, places=5)
        self.assertAlmostEqual(c_part, 0.75, places=5)
        self.assertGreater(c, c_part)
        self.assertGreater(c_part, 0.5)

    def test_spatial_continuity_adjacent_and_non_contact(self):
        m1 = np.zeros((64, 64), dtype=bool)
        m1[10:20, 10:20] = True

        # Adjacent (touches border, no direct overlap)
        m_adj = np.zeros((64, 64), dtype=bool)
        m_adj[20:25, 10:20] = True
        c_adj, diag_adj = compute_spatial_continuity(m1, m_adj)
        self.assertEqual(diag_adj["category"], "ADJACENT_OVERLAP")
        self.assertGreaterEqual(c_adj, 0.3)
        self.assertLessEqual(c_adj, 0.5)

        # Non-contact at distance 2 patches
        m_dist2 = np.zeros((64, 64), dtype=bool)
        m_dist2[22:25, 10:20] = True
        c_dist2, diag_dist2 = compute_spatial_continuity(m1, m_dist2)
        self.assertEqual(diag_dist2["category"], "NON_CONTACT_PROXIMITY")
        self.assertGreater(c_dist2, 0.0)
        self.assertLess(c_dist2, 0.3)

        # Non-contact at distance 6 patches
        m_dist6 = np.zeros((64, 64), dtype=bool)
        m_dist6[26:30, 10:20] = True
        c_dist6, diag_dist6 = compute_spatial_continuity(m1, m_dist6)
        self.assertGreater(c_dist6, 0.0)
        self.assertLess(c_dist6, c_dist2)

        # Strict hierarchy: Direct > Adjacent > Non-contact
        self.assertGreater(c_adj, c_dist2)
        self.assertGreater(c_dist2, c_dist6)

    def test_angular_consistency_circular_wraparound(self):
        # 1 deg difference across 0/360 boundary
        a_near, diag_near = compute_angular_consistency(1.0, 359.0, 30.0)
        self.assertAlmostEqual(diag_near["delta_cpa_deg"], 2.0, places=4)
        self.assertGreater(a_near, 0.95)

        # Identical CPA
        a_exact, _ = compute_angular_consistency(180.0, 180.0, 30.0)
        self.assertAlmostEqual(a_exact, 1.0, places=5)

        # 30 deg difference
        a_30, _ = compute_angular_consistency(150.0, 180.0, 30.0)
        self.assertGreater(a_exact, a_30)
        self.assertGreater(a_30, 0.0)

        # 90 deg difference: small but strictly positive (no zero-collapse)
        a_90, _ = compute_angular_consistency(90.0, 180.0, 30.0)
        self.assertGreater(a_90, 0.0)
        self.assertLess(a_90, a_30)

    def test_kinematic_consistency_causal_prediction(self):
        t0 = datetime(2010, 12, 12, 6, 0, 0)
        t1 = datetime(2010, 12, 12, 6, 20, 0)  # dt = 1200 s
        t2 = datetime(2010, 12, 12, 6, 40, 0)  # dt = 1200 s

        # Observed velocity: 6 px over 1200 s = 0.005 px/s
        history = [(t0, 50.0), (t1, 56.0)]
        k_pred, diag_pred = compute_kinematic_consistency(62.0, 56.0, history, t2)
        self.assertAlmostEqual(diag_pred["predicted_r_front"], 62.0, places=3)
        self.assertAlmostEqual(k_pred, 1.0, places=4)

        # Inward displacement: lower K but strictly positive (no zero-collapse)
        k_inward, diag_inward = compute_kinematic_consistency(52.0, 56.0, history, t2)
        self.assertGreater(k_inward, 0.0)
        self.assertLess(k_inward, k_pred)

        # Newborn track (no velocity history): does not collapse or fail
        newborn_history = [(t1, 56.0)]
        k_newborn, diag_newborn = compute_kinematic_consistency(62.0, 56.0, newborn_history, t2)
        self.assertFalse(diag_newborn["has_velocity_history"])
        self.assertGreater(k_newborn, 0.5)
        self.assertLessEqual(k_newborn, 1.0)

    def test_detection_confidence_scaling(self):
        m = np.zeros((64, 64), dtype=bool)
        m[10:15, 10:15] = True
        scores = np.full((64, 64), 2.0, dtype=np.float32)

        # Exactly at threshold: D = 0.5
        d_th, diag_th = compute_detection_confidence(m, scores, 2.0)
        self.assertAlmostEqual(d_th, 0.5, places=4)

        # Above threshold: D increases monotonically toward 1.0
        scores_hi = np.full((64, 64), 6.0, dtype=np.float32)
        d_hi, diag_hi = compute_detection_confidence(m, scores_hi, 2.0)
        self.assertGreater(d_hi, d_th)
        self.assertLessEqual(d_hi, 1.0)

        # Fallback when continuous scores absent
        d_none, _ = compute_detection_confidence(m, None, 2.0)
        self.assertAlmostEqual(d_none, 0.5, places=4)

    def test_geometric_mean_nr_score(self):
        # Balanced perfect score
        s_perfect = compute_nr_score(1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(s_perfect, 1.0, places=5)

        # Typical candidate score
        s_typical = compute_nr_score(0.8, 0.7, 0.9, 0.6)
        expected = (0.8 * 0.7 * 0.9 * 0.6) ** 0.25
        self.assertAlmostEqual(s_typical, expected, places=5)

        # Extreme non-contact weak signal candidate: NO zero-collapse
        s_weak = compute_nr_score(0.05, 0.1, 0.15, 0.5)
        self.assertGreater(s_weak, 0.0)
        self.assertLess(s_weak, s_typical)


if __name__ == "__main__":
    unittest.main()
