"""Redesigned Normalized Evidence Scoring for Experimental A5b-NR Tracking.

Replaces heuristic A5b matching terms (linear overlap ratio, clipped angular
difference, binary inward/outward step, leading-front radius bonus) with four
physically interpretable, bounded [0, 1] evidence components:
    S_NR = (C * A * K * D)**(1/4)

Evidence components:
- C: Spatial Continuity (Dice overlap for direct contact, boundary fraction
     for adjacency, and continuous Gaussian distance kernel for non-contact).
- A: Angular Consistency (circular Gaussian kernel on delta-CPA).
- K: Kinematic Consistency (causal predicted radial front from velocity history
     and elapsed time).
- D: Detection Confidence (patch-level D4 detection score excess over threshold).

All terms are strictly positive for eligible candidates, eliminating the 48.3%
multiplicative zero-collapse observed in A5b-N.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

STRUCT_8 = np.ones((3, 3), dtype=bool)


def circular_diff_deg(a: float, b: float) -> float:
    """Signed circular difference a - b in [-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def compute_spatial_continuity(
    prev_mask: np.ndarray,
    cand_mask: np.ndarray,
    *,
    prev_dilation: Optional[np.ndarray] = None,
    sigma_s_patches: float = 4.0,
) -> Tuple[float, dict[str, Any]]:
    """Compute bounded spatial continuity C in (0, 1].

    - Direct overlap: C in (0.5, 1.0] based on Dice similarity.
    - 8-connected adjacency: C in [0.3, 0.5] based on shared boundary contact fraction.
    - Non-contact: C in (0, 0.3) based on Gaussian spatial distance kernel.
    """
    prev_n = int(prev_mask.sum())
    cand_n = int(cand_mask.sum())

    if prev_n == 0 or cand_n == 0:
        return 0.5, {
            "category": "UNINITIALIZED_FALLBACK",
            "direct_overlap_count": 0,
            "adjacent_overlap_count": 0,
            "dice": 0.0,
            "min_distance_patches": float("nan"),
        }

    do_cnt = int(np.logical_and(prev_mask, cand_mask).sum())
    if do_cnt > 0:
        dice = (2.0 * do_cnt) / float(prev_n + cand_n)
        dice_clamped = min(1.0, max(0.0, dice))
        c = 0.5 + 0.5 * dice_clamped
        return c, {
            "category": "DIRECT_OVERLAP",
            "direct_overlap_count": do_cnt,
            "adjacent_overlap_count": do_cnt,
            "dice": dice,
            "min_distance_patches": 0.0,
        }

    if prev_dilation is None:
        prev_dilation = binary_dilation(prev_mask, STRUCT_8)

    adj_cnt = int(np.logical_and(prev_dilation, cand_mask).sum())
    if adj_cnt > 0:
        f_adj = min(1.0, float(adj_cnt) / max(1.0, min(prev_n, cand_n)))
        c = 0.3 + 0.2 * f_adj
        return c, {
            "category": "ADJACENT_OVERLAP",
            "direct_overlap_count": 0,
            "adjacent_overlap_count": adj_cnt,
            "dice": 0.0,
            "min_distance_patches": 1.0,
        }

    # Non-contact candidate satisfying PA and radial corridor constraints
    dist_map = distance_transform_edt(~prev_mask)
    min_dist = float(np.min(dist_map[cand_mask]))
    # Adjacent patches are distance 1.0; non-contact patches have min_dist >= sqrt(2)
    d_excess = max(0.0, min_dist - 1.0)
    c = 0.3 * math.exp(-(d_excess**2) / (2.0 * (sigma_s_patches**2)))
    c = max(1e-4, min(0.3, c))
    return c, {
        "category": "NON_CONTACT_PROXIMITY",
        "direct_overlap_count": 0,
        "adjacent_overlap_count": 0,
        "dice": 0.0,
        "min_distance_patches": min_dist,
    }


def compute_angular_consistency(
    cand_cpa: float,
    track_cpa: float,
    roi_half_width: float,
    *,
    min_sigma_pa: float = 12.0,
    max_sigma_pa: float = 30.0,
) -> Tuple[float, dict[str, Any]]:
    """Compute bounded circular angular consistency A in (0, 1].

    Uses Gaussian circular difference kernel. Scale is regularized by observed
    angular width without letting wide lobes inflate uncertainty unbounded.
    """
    diff = abs(circular_diff_deg(cand_cpa, track_cpa))
    sigma_pa = max(min_sigma_pa, min(max_sigma_pa, 0.5 * roi_half_width))
    a = math.exp(-(diff**2) / (2.0 * (sigma_pa**2)))
    a = max(1e-4, min(1.0, a))
    return a, {
        "delta_cpa_deg": diff,
        "sigma_pa_deg": sigma_pa,
    }


def compute_kinematic_consistency(
    cand_r_front: float,
    prev_r_front: float,
    observed_history: List[Tuple[datetime, float]],
    current_timestamp: datetime,
    *,
    nominal_velocity_px_s: float = 0.005,
    max_velocity_px_s: float = 0.025,
) -> Tuple[float, dict[str, Any]]:
    """Compute bounded causal kinematic consistency K in (0, 1].

    Predicts expected radial front position causal from prior observations
    and actual elapsed time. Missing velocity history defaults to nominal
    CME expansion speed with wider uncertainty so newborn tracks are not
    disfavored.
    """
    has_velocity_history = False
    if len(observed_history) >= 2:
        t_prev, r_prev = observed_history[-1]
        t_prev2, r_prev2 = observed_history[-2]
        dt_hist = (t_prev - t_prev2).total_seconds()
        dr_hist = r_prev - r_prev2
        if dt_hist > 0:
            v_est = min(max_velocity_px_s, max(0.0, dr_hist / dt_hist))
            has_velocity_history = True
        else:
            v_est = nominal_velocity_px_s
    elif len(observed_history) == 1:
        t_prev, r_prev = observed_history[-1]
        v_est = nominal_velocity_px_s
    else:
        r_prev = prev_r_front
        t_prev = current_timestamp
        v_est = nominal_velocity_px_s

    dt_curr = max(0.0, (current_timestamp - t_prev).total_seconds())
    predicted_r_front = r_prev + v_est * dt_curr

    if has_velocity_history:
        sigma_r = max(8.0, 0.15 * predicted_r_front)
    else:
        sigma_r = max(12.0, 0.25 * predicted_r_front)

    r_diff = cand_r_front - predicted_r_front
    k = math.exp(-(r_diff**2) / (2.0 * (sigma_r**2)))
    k = max(1e-4, min(1.0, k))
    return k, {
        "estimated_velocity_px_s": v_est,
        "predicted_r_front": predicted_r_front,
        "radial_error_px": r_diff,
        "sigma_r_px": sigma_r,
        "has_velocity_history": has_velocity_history,
    }


def compute_detection_confidence(
    cand_mask: np.ndarray,
    frame_patch_scores: Optional[np.ndarray],
    effective_threshold: Optional[float],
    *,
    fallback_confidence: float = 0.5,
) -> Tuple[float, dict[str, Any]]:
    """Compute bounded detection confidence D in [0.5, 1.0).

    Uses the median patch score of the candidate component relative to the
    effective D4 threshold. Neutral fallback 0.5 when continuous scores are
    not available.
    """
    if (
        frame_patch_scores is None
        or effective_threshold is None
        or not math.isfinite(effective_threshold)
    ):
        return fallback_confidence, {
            "comp_score_median": float("nan"),
            "effective_threshold": float("nan"),
            "threshold_excess": 0.0,
        }

    comp_scores = frame_patch_scores[cand_mask]
    if len(comp_scores) == 0:
        return fallback_confidence, {
            "comp_score_median": float("nan"),
            "effective_threshold": effective_threshold,
            "threshold_excess": 0.0,
        }

    comp_median = float(np.median(comp_scores))
    excess = max(0.0, comp_median - effective_threshold)
    tau_d = max(1.0, effective_threshold)
    d = 0.5 + 0.5 * (1.0 - math.exp(-excess / tau_d))
    d = max(0.1, min(1.0, d))
    return d, {
        "comp_score_median": comp_median,
        "effective_threshold": effective_threshold,
        "threshold_excess": excess,
    }


def compute_nr_score(
    c: float,
    a: float,
    k: float,
    d: float,
) -> float:
    """Compute equal-weight geometric-mean score S_NR = (C * A * K * D)**(1/4) in (0, 1]."""
    for name, val in [("C", c), ("A", a), ("K", k), ("D", d)]:
        if not (math.isfinite(val) and val > 0.0):
            raise ValueError(f"Evidence term {name}={val} must be finite and positive")
    log_s = 0.25 * (math.log(c) + math.log(a) + math.log(k) + math.log(d))
    return min(1.0, max(0.0, math.exp(log_s)))
