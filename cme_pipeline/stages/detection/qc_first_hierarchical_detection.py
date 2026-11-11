"""Pure per-frame QC-first hierarchical detection logic (original experimental condition C3).

This module intentionally does not read FITS, derive rolling thresholds, or change
the existing frozen benchmark pipeline. The caller supplies precomputed P95 patch
scores, validity masks and causal T24 thresholds. A later streaming stage can
reuse this kernel after computing T24 from preceding observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import label

SCALES = (16, 8, 4)
UNAVAILABLE = -1
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class QCFirstDetectionFrameResult:
    observation_status: str
    usable_detection: bool
    active_fraction_pre: float
    floor_p04: float
    effective_threshold_p04: float
    floor_bound_p04: bool
    pre_hierarchical: dict[int, np.ndarray]
    final_states: dict[int, np.ndarray]
    component_labels_p04: np.ndarray


def apply_hierarchical_gating(
    flat16: np.ndarray, flat08: np.ndarray, flat04: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gate 8-pixel patches by 16-pixel parents, then 4 by 8."""
    h16 = np.asarray(flat16, dtype=bool).copy()
    h08 = np.asarray(flat08, dtype=bool) & np.repeat(
        np.repeat(h16, 2, axis=0), 2, axis=1
    )
    h04 = np.asarray(flat04, dtype=bool) & np.repeat(
        np.repeat(h08, 2, axis=0), 2, axis=1
    )
    return h16, h08, h04


def detect_qc_first_frame(
    scores: Mapping[int, np.ndarray],
    accepted: Mapping[int, np.ndarray],
    t24: Mapping[int, float],
    *,
    input_status: str = "OK",
    max_active_fraction_pre: float = 0.70,
    min_floor_patches: int = 3,
) -> QCFirstDetectionFrameResult:
    """Apply pre-floor global-activation QC, then p04-only frame-median floor.

    `input_status` must be OK, WARMUP, NO_HISTORY or MISSING_DATA.
    Unusable observations carry -1 states, never valid-empty zero masks.
    No floor-adjusted scores are fed back into the causal T24 history.
    """
    if input_status not in {"OK", "WARMUP", "NO_HISTORY", "MISSING_DATA"}:
        raise ValueError(f"Unknown input_status: {input_status}")
    if not 0.0 <= max_active_fraction_pre <= 1.0:
        raise ValueError("max_active_fraction_pre must be between 0 and 1")
    if min_floor_patches < 1:
        raise ValueError("min_floor_patches must be positive")

    expected = {16: (16, 16), 8: (32, 32), 4: (64, 64)}
    score_arrays: dict[int, np.ndarray] = {}
    accepted_arrays: dict[int, np.ndarray] = {}
    for scale in SCALES:
        s = np.asarray(scores[scale], dtype=float)
        a = np.asarray(accepted[scale], dtype=bool)
        if s.shape != expected[scale] or a.shape != expected[scale]:
            raise ValueError(f"Invalid p{scale:02d} shape: {s.shape}, {a.shape}")
        score_arrays[scale] = s
        accepted_arrays[scale] = a & np.isfinite(s)

    thresholds_available = all(np.isfinite(t24[s]) for s in SCALES)
    status = input_status
    if status == "OK" and not thresholds_available:
        status = "NO_HISTORY"
    if status == "OK" and not np.any(accepted_arrays[4]):
        status = "MISSING_DATA"

    flat = {}
    for scale in SCALES:
        flat[scale] = np.zeros(expected[scale], dtype=bool)
        if status == "OK":
            a = accepted_arrays[scale]
            flat[scale][a] = score_arrays[scale][a] > float(t24[scale])

    h16, h08, h04 = apply_hierarchical_gating(flat[16], flat[8], flat[4])
    pre = {16: h16, 8: h08, 4: h04}
    valid_p04_count = int(accepted_arrays[4].sum())
    active_fraction = (
        float((h04 & accepted_arrays[4]).sum() / valid_p04_count)
        if valid_p04_count else float("nan")
    )

    if status == "OK" and active_fraction >= max_active_fraction_pre:
        status = "GLOBAL_ACTIVATION"
    usable = status == "OK"
    if usable:
        status = "VALID"

    final = {
        scale: np.full(expected[scale], UNAVAILABLE, dtype=np.int8)
        for scale in SCALES
    }
    floor = float("nan")
    effective = float("nan")
    bound = False
    labels = np.full(expected[4], UNAVAILABLE, dtype=np.int32)

    if usable:
        valid_scores = score_arrays[4][accepted_arrays[4]]
        base_threshold = float(t24[4])
        if len(valid_scores) >= min_floor_patches:
            floor = float(np.median(valid_scores))
            effective = max(base_threshold, floor)
            bound = floor > base_threshold
        else:
            effective = base_threshold

        flat04_final = accepted_arrays[4] & (score_arrays[4] > effective)
        parent08 = np.repeat(np.repeat(h08, 2, axis=0), 2, axis=1)
        final[16] = h16.astype(np.int8)
        final[8] = h08.astype(np.int8)
        final[4] = (flat04_final & parent08).astype(np.int8)
        labels, _ = label(final[4].astype(bool), structure=CONNECTIVITY_8)
        labels = labels.astype(np.int32)

    return QCFirstDetectionFrameResult(
        observation_status=status,
        usable_detection=usable,
        active_fraction_pre=active_fraction,
        floor_p04=floor,
        effective_threshold_p04=effective,
        floor_bound_p04=bound,
        pre_hierarchical=pre,
        final_states=final,
        component_labels_p04=labels,
    )
