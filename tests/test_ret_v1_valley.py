r"""Unit and Synthetic Validation Suite for RET-V1 Symmetric Valley-Based Angular Lobe."""

import numpy as np
import pytest

from experiments.ret_tracking.ret_evidence_engine import (
    PolarGridGeometry,
    detect_angular_candidates,
)
from experiments.ret_tracking.ret_tracker import RETConfig, RadialEvidenceTracker


@pytest.fixture
def test_grid():
    return PolarGridGeometry(sx=128.0, sy=128.0)


def create_synthetic_angular_profile(n_pa: int = 360, bg_level: float = 1.0) -> np.ndarray:
    return np.full(n_pa, bg_level, dtype=np.float32)


def add_gaussian_peak(profile: np.ndarray, center_deg: float, amplitude: float, sigma_deg: float):
    n_pa = len(profile)
    pa_bins = np.arange(n_pa, dtype=np.float32)
    # circular difference
    diff = (pa_bins - center_deg + 180.0) % 360.0 - 180.0
    profile += (amplitude * np.exp(-0.5 * (diff / sigma_deg) ** 2)).astype(np.float32)


def make_dummy_polar_ev(n_r: int = 101, n_pa: int = 360, peak_r_idx: int = 40):
    ev = np.full((n_r, n_pa), 0.1, dtype=np.float32)
    ev[peak_r_idx : peak_r_idx + 10, :] = 2.0
    return ev


# Test 1: Symmetric peak with equally distant valleys
def test_1_symmetric_peak_equally_distant_valleys(test_grid):
    prof = create_synthetic_angular_profile()
    # Three peaks: 30 deg, 90 deg, 150 deg -> peak at 90 has valleys equally spaced at 60 deg and 120 deg
    add_gaussian_peak(prof, 30.0, 50.0, 5.0)
    add_gaussian_peak(prof, 90.0, 50.0, 5.0)
    add_gaussian_peak(prof, 150.0, 50.0, 5.0)

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="valley_symmetric")

    c90 = [c for c in cands if abs(c.peak_pa_deg - 90.0) < 2.0][0]
    assert c90.left_valley_pa_deg is not None
    assert c90.right_valley_pa_deg is not None
    # Distance to 60 deg and 120 deg is 30 deg each
    assert abs(c90.left_valley_dist_deg - c90.right_valley_dist_deg) < 2.0
    assert abs(c90.selected_half_width_deg - 30.0) < 2.0
    assert abs(c90.pa_width_deg - 60.0) < 4.0
    assert not c90.fallback_used


# Test 2: Asymmetric peak with closer left valley
def test_2_asymmetric_closer_left_valley(test_grid):
    prof = create_synthetic_angular_profile()
    # Peaks at 60 deg, 90 deg, 180 deg
    # Peak at 90: left peak at 60 (valley ~75, d_L ~15), right peak at 180 (valley ~135, d_R ~45)
    add_gaussian_peak(prof, 60.0, 50.0, 4.0)
    add_gaussian_peak(prof, 90.0, 50.0, 4.0)
    add_gaussian_peak(prof, 180.0, 50.0, 4.0)

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="valley_symmetric")

    c90 = [c for c in cands if abs(c.peak_pa_deg - 90.0) < 2.0][0]
    assert c90.left_valley_dist_deg < c90.right_valley_dist_deg
    assert c90.selected_half_width_deg == c90.left_valley_dist_deg
    assert c90.pa_width_deg == 2.0 * c90.selected_half_width_deg
    # Width bounded by left valley
    assert c90.pa_width_deg < 40.0


# Test 3: Asymmetric peak with closer right valley
def test_3_asymmetric_closer_right_valley(test_grid):
    prof = create_synthetic_angular_profile()
    # Peaks at 0 deg, 90 deg, 120 deg
    # Peak at 90: left peak at 0 (valley ~45, d_L ~45), right peak at 120 (valley ~105, d_R ~15)
    add_gaussian_peak(prof, 0.0, 50.0, 4.0)
    add_gaussian_peak(prof, 90.0, 50.0, 4.0)
    add_gaussian_peak(prof, 120.0, 50.0, 4.0)

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="valley_symmetric")

    c90 = [c for c in cands if abs(c.peak_pa_deg - 90.0) < 2.0][0]
    assert c90.right_valley_dist_deg < c90.left_valley_dist_deg
    assert c90.selected_half_width_deg == c90.right_valley_dist_deg
    assert c90.pa_width_deg == 2.0 * c90.selected_half_width_deg
    assert c90.pa_width_deg < 40.0


# Test 4: Multiple neighboring peaks
def test_4_multiple_neighboring_peaks(test_grid):
    prof = create_synthetic_angular_profile()
    # 5 peaks evenly separated by 45 deg
    peaks = [45.0, 90.0, 135.0, 180.0, 225.0]
    for p in peaks:
        add_gaussian_peak(prof, p, 40.0, 4.0)

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="valley_symmetric")

    assert len(cands) >= 5
    for c in cands:
        assert c.selected_half_width_deg is not None
        assert c.selected_half_width_deg > 0.0
        assert not c.fallback_used


# Test 5: Circular boundary crossing
def test_5_circular_boundary_crossing(test_grid):
    prof = create_synthetic_angular_profile()
    # Peaks at 350 deg, 20 deg, 90 deg
    # Peak at 350 wraps past 360 to 20 deg (span 30 deg)
    add_gaussian_peak(prof, 350.0, 50.0, 4.0)
    add_gaussian_peak(prof, 20.0, 50.0, 4.0)
    add_gaussian_peak(prof, 90.0, 50.0, 4.0)

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="valley_symmetric")

    c350 = [c for c in cands if abs((c.peak_pa_deg - 350.0 + 180) % 360 - 180) < 2.0][0]
    c20 = [c for c in cands if abs(c.peak_pa_deg - 20.0) < 2.0][0]

    assert c350.right_valley_pa_deg is not None
    # Valley between 350 and 20 is around 5 deg
    assert abs((c350.right_valley_pa_deg - 5.0 + 180) % 360 - 180) < 5.0
    assert abs((c20.left_valley_pa_deg - 5.0 + 180) % 360 - 180) < 5.0


# Test 6: Flat valley plateau
def test_6_flat_valley_plateau(test_grid):
    prof = create_synthetic_angular_profile(bg_level=2.0)
    # Peak at 50 deg and 200 deg. Between 90 deg and 160 deg is a flat plateau at 2.0
    add_gaussian_peak(prof, 50.0, 30.0, 4.0)
    add_gaussian_peak(prof, 200.0, 30.0, 4.0)

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="valley_symmetric")

    c50 = [c for c in cands if abs(c.peak_pa_deg - 50.0) < 2.0][0]
    c200 = [c for c in cands if abs(c.peak_pa_deg - 200.0) < 2.0][0]

    # For c50, right valley on plateau must pick index closest to 50 deg
    # For c200, left valley on plateau must pick index closest to 200 deg
    # Verify tie-breaking selects closest position to each peak
    assert c50.right_valley_dist_deg is not None
    assert c200.left_valley_dist_deg is not None
    assert c50.right_valley_pa_deg < c200.left_valley_pa_deg


# Test 7: Single-peak fallback
def test_7_single_peak_fallback(test_grid):
    prof = create_synthetic_angular_profile()
    # Single isolated peak at 120 deg
    add_gaussian_peak(prof, 120.0, 60.0, 6.0)

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="valley_symmetric")

    assert len(cands) == 1
    c = cands[0]
    assert c.fallback_used
    assert c.left_valley_pa_deg is None
    assert c.right_valley_pa_deg is None
    assert c.pa_width_deg >= 8.0


# Test 8: Narrow-lobe rejection
def test_8_narrow_lobe_rejection(test_grid):
    prof = create_synthetic_angular_profile()
    # Peaks at 98 deg, 100 deg, 102 deg are too close for find_peaks(distance=15).
    # To test narrow lobe rejection: two peaks separated by 16 deg (at 85 deg and 101 deg) with valleys at ~93 deg (d=8 deg -> h=8, W=16).
    # If half-width h < 4.0 deg (W < 8 deg), it must be rejected.
    # Suppose a peak at 90 deg has valleys at 87 deg and 93 deg (d=3 deg -> h=3 -> W=6 deg < 8 deg).
    # We construct a synthetic profile where valleys are at 3 deg distance:
    add_gaussian_peak(prof, 90.0, 40.0, 1.0)
    add_gaussian_peak(prof, 70.0, 40.0, 1.0)
    add_gaussian_peak(prof, 110.0, 40.0, 1.0)
    # Put artificially deep artificial dips at 87 deg and 93 deg
    prof[87] = -10.0
    prof[93] = -10.0

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, min_lobe_width_deg=8.0, lobe_width_mode="valley_symmetric")

    # Peak at 90 deg has h = min(3, 3) = 3 deg -> W = 6 deg < 8 deg -> must be rejected
    c90 = [c for c in cands if abs(c.peak_pa_deg - 90.0) < 2.0]
    assert len(c90) == 0  # rejected


# Test 9: Absence of overlapping lobe interiors
def test_9_absence_of_overlapping_lobe_interiors(test_grid):
    prof = create_synthetic_angular_profile()
    peaks = [30.0, 70.0, 120.0, 190.0, 270.0]
    for p in peaks:
        add_gaussian_peak(prof, p, 45.0, 4.0)

    polar_ev = make_dummy_polar_ev()
    cands = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="valley_symmetric")

    cands_sorted = sorted(cands, key=lambda c: c.peak_pa_deg)
    for i in range(len(cands_sorted) - 1):
        c_curr = cands_sorted[i]
        c_next = cands_sorted[i + 1]
        # In non-wrapping case: curr end must be <= next start
        if c_curr.peak_pa_deg < c_next.peak_pa_deg:
            assert c_curr.pa_end_deg <= c_next.pa_start_deg + 1e-3


# Test 10: Preservation of baseline results when prominence mode is selected
def test_10_preservation_of_baseline_results(test_grid):
    prof = create_synthetic_angular_profile()
    add_gaussian_peak(prof, 40.0, 50.0, 8.0)
    add_gaussian_peak(prof, 120.0, 30.0, 5.0)

    polar_ev = make_dummy_polar_ev()
    # Explicit default call vs explicit prominence call
    cands_default = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0)
    cands_prom = detect_angular_candidates(prof, polar_ev, test_grid, min_prominence=3.0, lobe_width_mode="prominence")

    # Verify exact match between default and prominence mode
    assert len(cands_default) == len(cands_prom)
    for cd, cp in zip(cands_default, cands_prom):
        assert cd.peak_pa_deg == cp.peak_pa_deg
        assert cd.pa_width_deg == cp.pa_width_deg
        assert cd.r_front_px == cp.r_front_px
        assert cd.left_valley_pa_deg is None
        assert not cd.fallback_used
