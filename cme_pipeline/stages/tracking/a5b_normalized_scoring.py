"""Bounded, equal-weight geometric-mean scores for experimental A5b-N.

This module changes only candidate PRIORITY. It deliberately preserves A5b's
candidate eligibility, patch arbitration, birth rules, and outward gate.
Scores are heuristics, NOT calibrated probabilities.
"""
from __future__ import annotations

import math


def normalized_candidate_score(
    overlap_score: float,
    angular_score: float,
    kinematic_score: float,
    front_radius_px: float,
    max_observable_radius_px: float,
    *,
    has_spatial_contact: bool,
) -> tuple[float, dict[str, float]]:
    """Return [0,1] match score and the bounded component scores.

    Direct/adjacent overlap is undefined as positive evidence when there is
    no spatial contact. In that case omit O rather than multiplying by zero;
    otherwise no-overlap candidates would be permanently disfavored.
    """
    if not (math.isfinite(max_observable_radius_px)
            and max_observable_radius_px > 0):
        raise ValueError("max_observable_radius_px must be finite and positive")

    def clip01(x: float) -> float:
        x = float(x)
        if not math.isfinite(x):
            raise ValueError("Candidate matching score must be finite")
        return min(1.0, max(0.0, x))

    # The original A5b overlap count can exceed prev_n due to dilation;
    # clipping preserves the original direct/adjacency definition.
    o = clip01(overlap_score)
    a = clip01(angular_score)
    k = clip01(kinematic_score)
    f = clip01(float(front_radius_px) / max_observable_radius_px)
    factors = (o, a, k, f) if has_spatial_contact else (a, k, f)
    score = math.prod(factors) ** (1.0 / len(factors))
    return score, {"O": o, "A": a, "K": k, "F": f}
