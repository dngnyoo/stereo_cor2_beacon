"""Birth Direction Reference and two-frame causal direction locking engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cme_pipeline.core.stage import BirthDirectionReference
from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg


def compute_solar_root_pa(
    pa_vals: np.ndarray, r_vals: np.ndarray, r_min_px: float, margin_px: float = 2.0
) -> float:
    """Computes Position Angle of the solar root (pixels closest to occulting edge)."""
    if len(pa_vals) == 0:
        return np.nan
    root_mask = r_vals <= (r_min_px + margin_px)
    if not np.any(root_mask):
        root_mask = r_vals <= (r_min_px + 5.0)
    if not np.any(root_mask):
        idx_min = int(np.argmin(r_vals))
        return float(pa_vals[idx_min] % 360.0)
    return circular_mean_deg(pa_vals[root_mask])


def create_birth_direction_reference(
    track_id: int,
    birth_pair_id: int,
    birth_time: datetime,
    birth_r_min_px: float,
    central_pa: float,
    angular_extent: float,
    birth_pa: Optional[float] = None,
    anchor_threshold_px: float = 30.0,
) -> BirthDirectionReference:
    """Initializes a PROVISIONAL BirthDirectionReference at the first observation."""
    is_anchor = float(birth_r_min_px) <= anchor_threshold_px
    b_pa = float(birth_pa % 360.0) if birth_pa is not None else float(central_pa % 360.0)
    cpa_1 = float(central_pa % 360.0)

    flags = []
    if is_anchor:
        flags.append("SOLAR_ROOT_ANCHOR")
    else:
        flags.append("UNANCHORED_FEATURE")

    return BirthDirectionReference(
        track_id=int(track_id),
        birth_pair_id=int(birth_pair_id),
        birth_time=birth_time,
        birth_r_min_px=float(birth_r_min_px),
        is_anchor=is_anchor,
        birth_pa=b_pa,
        cpa_first=cpa_1,
        cpa_second=None,
        reference_cpa=cpa_1,  # Provisional reference equals first CPA
        initial_angular_extent=float(angular_extent),
        status="PROVISIONAL",
        confirmation_time=None,
        confidence=1.0 if is_anchor else 0.5,
        flags=flags,
    )


def update_birth_direction_reference(
    ref: BirthDirectionReference,
    current_pair_id: int,
    current_time: datetime,
    current_cpa: float,
    is_consecutive: bool,
    max_lock_deviation_deg: float = 45.0,
) -> BirthDirectionReference:
    """Causally updates the BirthDirectionReference at the second observation.
    
    If already LOCKED, the reference is IMMUTABLE and cannot be altered.
    """
    if ref.status == "LOCKED":
        # Immutability principle: once locked, reference_cpa does NOT update
        return ref

    if ref.status == "PROVISIONAL":
        cpa_2 = float(current_cpa % 360.0)
        ref.cpa_second = cpa_2

        # Check consecutive observation condition
        if not is_consecutive:
            ref.status = "UNCONFIRMED"
            ref.flags.append("NON_CONSECUTIVE_CADENCE_GAP")
            ref.confidence *= 0.5
            return ref

        # Check angular deviation between first and second frame
        delta_pa = abs(float(circular_diff_deg(cpa_2, ref.cpa_first)))

        if delta_pa <= max_lock_deviation_deg:
            # Successfully lock two-frame circular mean reference
            locked_mean = circular_mean_deg([ref.cpa_first, cpa_2])
            ref.reference_cpa = float(locked_mean)
            ref.status = "LOCKED"
            ref.confirmation_time = current_time
            ref.flags.append(f"LOCKED_CIRCULAR_MEAN_DELTA_{delta_pa:.1f}DEG")
        else:
            # Excessive angular deviation: do not lock
            ref.status = "UNCONFIRMED"
            ref.flags.append(f"DIRECTION_DISCONTINUITY_DELTA_{delta_pa:.1f}DEG")
            ref.confidence *= 0.4

    return ref

