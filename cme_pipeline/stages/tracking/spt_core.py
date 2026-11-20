"""Core Data Structures, Seed Detection, Region Growth, and Motion Prediction for SPT.

Standalone implementation for Seed-Origin Propagation Tracking (SPT).
Does NOT import, call, or rely on A5b tracking or association logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import label

from cme_pipeline.core.stage import GeometryContext
from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg

STRUCT_8 = np.ones((3, 3), bool)


class SPTLifecycleState(str, Enum):
    """Lifecycle state of an independent SPT track."""
    TENTATIVE = "TENTATIVE"                     # Newly seeded, awaiting temporal confirmation
    CONFIRMED = "CONFIRMED"                     # Confirmed CME track satisfying persistence
    TEMPORARILY_MISSING = "TEMPORARILY_MISSING" # Unobserved at current frame, within tolerance
    TERMINATED = "TERMINATED"                   # Terminated track (cadence gap or lack of support)


@dataclass(frozen=True)
class SeedProperties:
    """Immutable provenance and physical characteristics of a representative seed."""
    seed_id: int
    birth_pair_index: int
    birth_timestamp: datetime
    patch_x: int
    patch_y: int
    seed_pa_deg: float
    seed_radius_px: float
    seed_score: float
    component_id: int
    initial_patch_count: int


@dataclass
class ComponentGeometry:
    """Physical and geometric attributes of an 8-connected active patch component."""
    component_id: int
    patch_indices: List[Tuple[int, int]]        # (y, x) coordinates in 64x64 grid
    patch_count: int
    pa_values: np.ndarray                       # Array of PAs for each patch
    radius_values: np.ndarray                   # Array of radii for each patch
    scores: np.ndarray                          # Array of continuous D4 detection scores
    cpa_deg: float                              # Circular mean PA
    pa_min_deg: float                           # Minimal enclosing sector start
    pa_max_deg: float                           # Minimal enclosing sector end
    angular_width_deg: float                    # Circular angular width
    r_min_px: float                             # Innermost radial boundary
    r_max_px: float                             # Outermost radial boundary (leading front)
    r_mean_px: float                            # Mean radial distance
    centroid_x: float                           # Evidence-weighted centroid X
    centroid_y: float                           # Evidence-weighted centroid Y
    has_seed: bool = False
    representative_seed: Optional[SeedProperties] = None


@dataclass
class SPTTrackHistoryItem:
    """Record of a track's physical state at a single observation frame."""
    pair_index: int
    timestamp: datetime
    state: str                                  # "OBSERVED", "MISSED", "GAP"
    patch_indices: List[Tuple[int, int]]
    patch_count: int
    cpa_deg: float
    angular_width_deg: float
    pa_min_deg: float
    pa_max_deg: float
    r_min_px: float
    r_max_px: float                             # Leading front radius
    r_mean_px: float
    observed_radial_velocity_px_s: float        # Observed instantaneous velocity
    predicted_r_front_px: float                 # Prior predicted front radius
    predicted_cpa_deg: float                    # Prior predicted CPA
    associated_component_ids: List[int]


@dataclass
class SPTTrack:
    """Independent Seed-Origin Propagated Track object."""
    track_id: int
    lifecycle_state: SPTLifecycleState
    birth_pair_index: int
    birth_timestamp: datetime
    seed: SeedProperties
    
    # Dynamic angular state
    current_cpa: float
    tracked_cpa: float                          # Exponentially smoothed CPA
    predicted_cpa: float                        # Forward predicted CPA
    current_width: float
    pa_min: float
    pa_max: float
    
    # Dynamic radial state and kinematics
    current_r_min: float
    current_r_front: float
    predicted_r_front: float
    predicted_r_sigma: float
    radial_velocity_px_s: float                 # Estimated outward velocity (px/s)
    
    # Lifecycle counters
    consecutive_observed_pairs: int = 1
    consecutive_missed_pairs: int = 0
    total_observed_pairs: int = 1
    total_patches_accumulated: int = 0
    last_observed_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_observed_pair_index: int = 0
    termination_reason: Optional[str] = None
    
    # Observation history
    history: List[SPTTrackHistoryItem] = field(default_factory=list)
    owned_patches: Set[Tuple[int, int]] = field(default_factory=set)


# =====================================================================
# Seed Detection and Characterization
# =====================================================================

def detect_seeds_and_grow_components(
    active_grid: np.ndarray,
    scores_grid: np.ndarray,
    geom: GeometryContext,
    pair_index: int,
    timestamp: datetime,
    *,
    r_min_px: float = 22.0,
    r_seed_max_px: float = 30.0,
    seed_score_cutoff: float = 0.0,
    min_component_patches: int = 1,
) -> Tuple[List[ComponentGeometry], List[SeedProperties]]:
    """Identifies candidate seeds within near-Sun annulus and extracts all T24 components.

    For each 8-connected active component intersecting [r_min_px, r_seed_max_px]:
        p* = argmax_{p in C and r_min <= r_p <= r_seed_max} S_D4(p)
    Selects highest scoring eligible patch as representative seed.
    Preserves exact geometry and continuous detection score distributions.
    """
    if active_grid.shape != (64, 64) or scores_grid.shape != (64, 64):
        raise ValueError("Grids must have shape (64, 64)")

    labeled_arr, num_features = label(active_grid.astype(bool), structure=STRUCT_8)
    components: List[ComponentGeometry] = []
    seeds: List[SeedProperties] = []

    patch_rad = np.asarray(geom.patch_rad, dtype=np.float32)
    patch_pa = np.asarray(geom.patch_pa, dtype=np.float32)

    # Deterministic processing order by component label
    for comp_lbl in range(1, num_features + 1):
        coords_y, coords_x = np.where(labeled_arr == comp_lbl)
        n_patches = len(coords_y)
        if n_patches < min_component_patches:
            continue

        patch_coords = list(zip(coords_y.tolist(), coords_x.tolist()))
        r_vals = patch_rad[coords_y, coords_x]
        pa_vals = patch_pa[coords_y, coords_x]
        score_vals = scores_grid[coords_y, coords_x]

        # Component geometric properties
        r_min = float(np.min(r_vals))
        r_max = float(np.max(r_vals))
        r_mean = float(np.mean(r_vals))
        cpa = float(circular_mean_deg(pa_vals))

        # Circular sector bounds [pa_min, pa_max] and width
        # Compute minimal enclosing circular arc
        pa_sorted = np.sort(np.mod(pa_vals, 360.0))
        diffs = np.diff(pa_sorted)
        wrap_diff = (pa_sorted[0] + 360.0) - pa_sorted[-1]
        all_diffs = np.append(diffs, wrap_diff)
        max_gap_idx = int(np.argmax(all_diffs))
        max_gap = float(all_diffs[max_gap_idx])

        if max_gap >= 360.0:
            width = 0.0
            pa_min = cpa
            pa_max = cpa
        else:
            width = 360.0 - max_gap
            pa_min = float(pa_sorted[(max_gap_idx + 1) % len(pa_sorted)])
            pa_max = float((pa_min + width) % 360.0)

        # Evidence-weighted centroid
        w = np.maximum(score_vals, 0.01)
        w_sum = float(np.sum(w))
        centroid_x = float(np.sum(coords_x * w) / w_sum)
        centroid_y = float(np.sum(coords_y * w) / w_sum)

        # Identify candidate seeds within the near-Sun search annulus
        in_annulus = (r_vals >= r_min_px) & (r_vals <= r_seed_max_px)
        has_seed = False
        rep_seed: Optional[SeedProperties] = None

        if np.any(in_annulus):
            annulus_indices = np.where(in_annulus)[0]
            # Filter by score cutoff
            qualifying = [idx for idx in annulus_indices if score_vals[idx] >= seed_score_cutoff]
            if qualifying:
                # Deterministic tie-breaking: max score, then min radius, then min PA
                best_idx = max(
                    qualifying,
                    key=lambda idx: (score_vals[idx], -r_vals[idx], -pa_vals[idx])
                )
                sy, sx = coords_y[best_idx], coords_x[best_idx]
                seed_id = len(seeds) + 1
                rep_seed = SeedProperties(
                    seed_id=seed_id,
                    birth_pair_index=pair_index,
                    birth_timestamp=timestamp,
                    patch_x=int(sx),
                    patch_y=int(sy),
                    seed_pa_deg=float(patch_pa[sy, sx]),
                    seed_radius_px=float(patch_rad[sy, sx]),
                    seed_score=float(scores_grid[sy, sx]),
                    component_id=comp_lbl,
                    initial_patch_count=n_patches,
                )
                seeds.append(rep_seed)
                has_seed = True

        comp_geom = ComponentGeometry(
            component_id=comp_lbl,
            patch_indices=patch_coords,
            patch_count=n_patches,
            pa_values=pa_vals,
            radius_values=r_vals,
            scores=score_vals,
            cpa_deg=cpa,
            pa_min_deg=pa_min,
            pa_max_deg=pa_max,
            angular_width_deg=width,
            r_min_px=r_min,
            r_max_px=r_max,
            r_mean_px=r_mean,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            has_seed=has_seed,
            representative_seed=rep_seed,
        )
        components.append(comp_geom)

    return components, seeds


# =====================================================================
# Motion Prediction & Trajectory Kinematics (SPT-1 Core)
# =====================================================================

def update_track_kinematics(
    track: SPTTrack,
    current_time: datetime,
    observed_r_front_px: float,
    observed_cpa_deg: float,
    *,
    ema_alpha_cpa: float = 0.15,
) -> None:
    """Updates dynamic velocity, tracked CPA, and forward prediction models."""
    dt_sec = (current_time - track.last_observed_timestamp).total_seconds()
    if dt_sec <= 0:
        dt_sec = 1800.0  # Default 30 min nominal fallback

    # 1. Update Observed Velocity
    dr_px = observed_r_front_px - track.current_r_front
    instantaneous_v = max(0.0, dr_px / dt_sec)  # Enforce non-negative outward motion

    if len(track.history) <= 1:
        track.radial_velocity_px_s = instantaneous_v
    else:
        # Smooth velocity across recent observations (median or weighted average)
        recent_vs = [
            h.observed_radial_velocity_px_s for h in track.history[-3:]
            if h.state == "OBSERVED" and h.observed_radial_velocity_px_s > 0
        ]
        recent_vs.append(instantaneous_v)
        track.radial_velocity_px_s = float(np.median(recent_vs))

    # 2. Update Tracked CPA using Exponential Smoothing (handling 0/360 wrap)
    cpa_diff = (observed_cpa_deg - track.tracked_cpa + 180.0) % 360.0 - 180.0
    track.tracked_cpa = (track.tracked_cpa + ema_alpha_cpa * cpa_diff) % 360.0
    track.current_cpa = observed_cpa_deg

    # 3. Predict CPA for next timestep
    # If historical CPA drift is observed, project drift with damping
    if len(track.history) >= 2 and dt_sec > 0:
        prev_cpa = track.history[-1].cpa_deg
        drift_rate_deg_s = ((observed_cpa_deg - prev_cpa + 180.0) % 360.0 - 180.0) / dt_sec
        # Damped forward projection (max 5 deg per 30 min)
        projected_drift = np.clip(drift_rate_deg_s * 1800.0, -5.0, 5.0)
        track.predicted_cpa = (track.tracked_cpa + projected_drift) % 360.0
    else:
        track.predicted_cpa = track.tracked_cpa


def predict_track_forward(
    track: SPTTrack,
    target_time: datetime,
) -> Tuple[float, float, float]:
    """Predicts expected leading-front radius, radial uncertainty, and CPA at target_time.

    Returns:
        (predicted_r_front_px, predicted_sigma_r_px, predicted_cpa_deg)
    """
    dt_sec = max(0.0, (target_time - track.last_observed_timestamp).total_seconds())
    nominal_v = track.radial_velocity_px_s
    if nominal_v <= 0:
        # Default nominal coronal velocity: ~300 km/s in COR2 px (~0.003 px/s)
        nominal_v = 0.0035

    predicted_r = track.current_r_front + nominal_v * dt_sec
    # Radial prediction uncertainty grows with elapsed time
    base_sigma = 4.0  # Base uncertainty in patch pixels
    time_sigma = (dt_sec / 1800.0) * 3.0  # +3 px per 30 min gap
    predicted_sigma = float(base_sigma + time_sigma)

    return float(predicted_r), predicted_sigma, track.predicted_cpa
