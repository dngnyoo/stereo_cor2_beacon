"""Standalone Gaussian Evidence Tracker for CME Identification and Tracking.

Implements:
- G0: Raw pixel evidence tracking (sigma_s = 0)
- G1: Spatial Gaussian evidence tracking (sigma_s in {1, 2, 4, 6} px)
- G2: Causal temporal accumulation memory:
      H_t = alpha(dt) * H_{t-1} + (1 - alpha(dt)) * E_tilde_t
- G3: Motion-aware association discovering trajectory velocities dynamically:
      r_pred = r_{t-1} + v_r * dt

Completely independent of A5b tracks, T24 connected components, or prescribed velocities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg
from experiments.gaussian_tracking.gaussian_evidence_model import (
    PixelEvidenceConfig,
    EvidenceRegion,
    compute_pixel_evidence,
    apply_spatial_gaussian_spreading,
    extract_evidence_regions,
    compute_gaussian_evidence_overlap,
)


@dataclass
class TrackerConfig:
    """Configuration for Gaussian Evidence Tracker."""
    variant: str = "G1"  # 'G0', 'G1', 'G2', 'G3'
    sigma_s: float = 2.0  # Gaussian spreading scale (0.0 for G0)
    evidence_formulation: str = "positive"  # 'positive' or 'compressed'
    evidence_threshold: float = 0.5  # Detection threshold on spread evidence
    min_region_area_px: int = 8  # Minimum region pixel count
    tau_half_life_min: float = 40.0  # Temporal accumulation half-life for G2/G3 (minutes)
    birth_mode: str = "near_sun"  # 'near_sun' (r <= r_birth_max) or 'unrestricted'
    r_birth_max: float = 38.0  # Max radius for new track birth in pixels (~4.7 R_sun)
    r_min_px: float = 22.0  # Inner occulter boundary
    r_max_px: float = 122.5  # Outer coronagraph FOV boundary
    min_confirm_hits: int = 2  # Detections required to confirm track
    max_missed_frames: int = 2  # Max consecutive missed frames before termination
    max_dt_gap_min: float = 75.0  # Max observation gap before terminating active tracks
    max_pa_diff_deg: float = 25.0  # Max position angle difference allowed for association
    max_spatial_dist_px: float = 40.0  # Max Euclidean centroid jump allowed per step
    min_radial_velocity_px_hr: float = -6.0  # Strict outward motion tolerance (slight centroid noise)
    max_radial_velocity_px_hr: float = 120.0  # Physical CME speed upper bound in px/hr
    # Cost weights for bipartite association
    w_spatial: float = 1.0  # Centroid displacement weight
    w_pa: float = 1.2  # Angular deviation weight
    w_motion: float = 1.5  # Radial velocity / prediction error weight
    w_overlap: float = 1.0  # Continuous Gaussian overlap weight
    gating_cost_threshold: float = 50.0  # Upper bound cost for acceptable association


@dataclass
class TrackObservation:
    """Individual observation hit associated with a track."""
    frame_idx: int
    timestamp: datetime
    dt_hours: float
    region_id: int
    cx: float
    cy: float
    r: float
    pa: float
    r_front: float
    r_min: float
    pa_span: float
    total_evidence: float
    peak_evidence: float
    area_px: int
    v_r_instant: float  # px/hr
    v_r_smoothed: float  # px/hr
    predicted_r: Optional[float] = None
    association_cost: float = 0.0
    overlap_score: float = 0.0


@dataclass
class GaussianTrack:
    """State of an independently discovered CME track."""
    track_id: int
    status: str  # 'tentative', 'confirmed', 'terminated'
    observations: List[TrackObservation] = field(default_factory=list)
    missed_count: int = 0
    last_active_time: Optional[datetime] = None

    @property
    def latest(self) -> TrackObservation:
        return self.observations[-1]

    @property
    def first(self) -> TrackObservation:
        return self.observations[0]

    @property
    def hits(self) -> int:
        return len(self.observations)

    @property
    def current_r(self) -> float:
        return self.latest.r

    @property
    def current_pa(self) -> float:
        return self.latest.pa

    @property
    def current_vr(self) -> float:
        return self.latest.v_r_smoothed

    @property
    def mean_cpa(self) -> float:
        if not self.observations:
            return 0.0
        pas = [obs.pa for obs in self.observations]
        return float(circular_mean_deg(np.array(pas)))

    @property
    def cpa_jitter(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        m_pa = self.mean_cpa
        diffs = [circular_diff_deg(obs.pa, m_pa) for obs in self.observations]
        return float(np.std(diffs))

    @property
    def duration_hours(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        return (self.latest.timestamp - self.first.timestamp).total_seconds() / 3600.0

    @property
    def mean_radial_velocity(self) -> float:
        """Overall discovered radial velocity in px/hr."""
        if len(self.observations) < 2:
            return 0.0
        dt_hr = self.duration_hours
        if dt_hr <= 0:
            return 0.0
        dr = self.latest.r - self.first.r
        return dr / dt_hr

    @property
    def outward_consistency(self) -> float:
        """Fraction of sequential steps with non-negative outward radial displacement."""
        if len(self.observations) < 2:
            return 1.0
        outward_steps = sum(1 for obs in self.observations[1:] if obs.v_r_instant >= -0.5)
        return float(outward_steps / (len(self.observations) - 1))


@dataclass
class FrameTrackingResult:
    """Per-frame tracking diagnostic output."""
    frame_idx: int
    timestamp: datetime
    dt_from_prev_min: float
    raw_evidence: np.ndarray
    spread_evidence: np.ndarray
    accumulated_evidence: Optional[np.ndarray]
    extracted_regions: List[EvidenceRegion]
    active_confirmed_tracks: List[GaussianTrack]
    active_tentative_tracks: List[GaussianTrack]
    newly_born_tracks: List[int]
    matched_tracks: List[Tuple[int, int]]  # (track_id, region_id)
    terminated_tracks: List[int]


class GaussianEvidenceTracker:
    """Independent Spatial Gaussian Evidence Tracker.

    Discovers, links, and measures CME trajectories without any dependency on
    external track labels, patch components, or fixed speed assumptions.
    """

    def __init__(
        self,
        config: TrackerConfig,
        sx: float,
        sy: float,
        radius_raster: np.ndarray,
        pa_raster: np.ndarray,
    ):
        self.config = config
        self.sx = float(sx)
        self.sy = float(sy)
        self.radius_raster = radius_raster
        self.pa_raster = pa_raster

        # Ensure G0 has sigma_s = 0.0
        if self.config.variant == "G0":
            self.config.sigma_s = 0.0

        self.tracks: Dict[int, GaussianTrack] = {}
        self.next_track_id: int = 1
        self.previous_timestamp: Optional[datetime] = None
        self.accumulated_memory: Optional[np.ndarray] = None
        self.previous_regions: List[EvidenceRegion] = []
        self.previous_spread_ev: Optional[np.ndarray] = None

    def reset(self):
        """Resets tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.previous_timestamp = None
        self.accumulated_memory = None
        self.previous_regions.clear()
        self.previous_spread_ev = None

    def step(
        self,
        frame_idx: int,
        timestamp: datetime,
        z_map: np.ndarray,
        valid_mask: np.ndarray,
    ) -> FrameTrackingResult:
        """Processes one observation frame and updates track states."""
        # 1. Cadence and gap evaluation
        dt_hours = 0.5  # default 30 min
        dt_min = 30.0
        if self.previous_timestamp is not None:
            dt_sec = (timestamp - self.previous_timestamp).total_seconds()
            dt_hours = max(0.01, dt_sec / 3600.0)
            dt_min = dt_sec / 60.0

        # Terminate active tracks if observation gap exceeds max_dt_gap_min
        if self.previous_timestamp is not None and dt_min > self.config.max_dt_gap_min:
            for trk in self.tracks.values():
                if trk.status in ("tentative", "confirmed"):
                    trk.status = "terminated"
            # Reset temporal memory across large gap
            self.accumulated_memory = None

        # 2. Compute pixel evidence E_t(x, y)
        ev_raw = compute_pixel_evidence(
            z_map=z_map,
            valid_mask=valid_mask,
            formulation=self.config.evidence_formulation,
            r_min_px=self.config.r_min_px,
            r_max_px=self.config.r_max_px,
            radius_raster=self.radius_raster,
        )

        # 3. Spatial Gaussian spreading E_tilde_t(x, y)
        ev_spread = apply_spatial_gaussian_spreading(
            evidence=ev_raw,
            valid_mask=valid_mask,
            sigma_s=self.config.sigma_s,
            r_min_px=self.config.r_min_px,
            r_max_px=self.config.r_max_px,
            radius_raster=self.radius_raster,
        )

        # 4. Temporal accumulation memory (G2 & G3)
        ev_for_extraction = ev_spread
        accum_field: Optional[np.ndarray] = None

        if self.config.variant in ("G2", "G3"):
            tau_min = self.config.tau_half_life_min
            alpha = float(math.exp(-dt_min * math.log(2.0) / tau_min))
            if self.accumulated_memory is None:
                self.accumulated_memory = ev_spread.copy()
            else:
                self.accumulated_memory = (
                    alpha * self.accumulated_memory + (1.0 - alpha) * ev_spread
                ).astype(np.float32)

            accum_field = self.accumulated_memory.copy()
            if self.config.variant == "G2":
                # In G2, region extraction operates directly on accumulated evidence field
                ev_for_extraction = self.accumulated_memory

        # 5. Extract spatially coherent high-evidence regions
        regions = extract_evidence_regions(
            spread_evidence=ev_for_extraction,
            valid_mask=valid_mask,
            sx=self.sx,
            sy=self.sy,
            radius_raster=self.radius_raster,
            pa_raster=self.pa_raster,
            threshold=self.config.evidence_threshold,
            min_area_px=self.config.min_region_area_px,
            r_min_px=self.config.r_min_px,
            r_max_px=self.config.r_max_px,
        )

        # 6. Associate active tracks with detected regions
        active_track_ids = [
            tid for tid, trk in self.tracks.items()
            if trk.status in ("tentative", "confirmed")
        ]

        matched_tracks, newly_born, terminated = self._associate_and_update(
            frame_idx=frame_idx,
            timestamp=timestamp,
            dt_hours=dt_hours,
            active_track_ids=active_track_ids,
            regions=regions,
            ev_spread=ev_spread,
        )

        # 7. Collect diagnostics
        confirmed = [trk for trk in self.tracks.values() if trk.status == "confirmed"]
        tentative = [trk for trk in self.tracks.values() if trk.status == "tentative"]

        result = FrameTrackingResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            dt_from_prev_min=dt_min,
            raw_evidence=ev_raw,
            spread_evidence=ev_spread,
            accumulated_evidence=accum_field,
            extracted_regions=regions,
            active_confirmed_tracks=confirmed,
            active_tentative_tracks=tentative,
            newly_born_tracks=newly_born,
            matched_tracks=matched_tracks,
            terminated_tracks=terminated,
        )

        self.previous_timestamp = timestamp
        self.previous_regions = regions
        self.previous_spread_ev = ev_spread.copy()

        return result

    def _associate_and_update(
        self,
        frame_idx: int,
        timestamp: datetime,
        dt_hours: float,
        active_track_ids: List[int],
        regions: List[EvidenceRegion],
        ev_spread: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Performs bipartite association and handles birth/termination."""
        n_tracks = len(active_track_ids)
        n_regions = len(regions)

        matched_pairs: List[Tuple[int, int]] = []
        matched_track_set = set()
        matched_region_set = set()

        if n_tracks > 0 and n_regions > 0:
            cost_matrix = np.full((n_tracks, n_regions), 1e6, dtype=np.float64)

            for i, tid in enumerate(active_track_ids):
                trk = self.tracks[tid]
                last_obs = trk.latest

                # Predict radial position
                if self.config.variant == "G3":
                    # Discovered trajectory prediction: r_pred = r_prev + v_r * dt
                    # If v_r is not yet established, default to outward drift
                    v_pred = trk.current_vr if abs(trk.current_vr) > 0.1 else 8.0
                    r_pred = last_obs.r + v_pred * dt_hours
                else:
                    r_pred = last_obs.r

                for j, reg in enumerate(regions):
                    # 1. Angular PA continuity check
                    pa_diff = circular_diff_deg(reg.centroid_pa, last_obs.pa)
                    if pa_diff > self.config.max_pa_diff_deg:
                        continue  # Directional jump too large

                    # 2. Spatial displacement check
                    d_spatial = math.hypot(reg.centroid_x - last_obs.cx, reg.centroid_y - last_obs.cy)
                    if d_spatial > self.config.max_spatial_dist_px:
                        continue  # Spatial jump too far for single cadence step

                    # 3. Discovered radial velocity check
                    dr = reg.centroid_r - last_obs.r
                    vr_instant = dr / dt_hours

                    if vr_instant < self.config.min_radial_velocity_px_hr:
                        continue  # Inward collapse violation
                    if vr_instant > self.config.max_radial_velocity_px_hr:
                        continue  # Unphysically fast outward jump

                    # 4. Continuous Gaussian evidence overlap (if previous map available)
                    overlap = 0.0
                    if self.previous_spread_ev is not None and last_obs.region_id >= 0:
                        # Find previous region mask if available
                        prev_mask = None
                        for preg in self.previous_regions:
                            if preg.region_id == last_obs.region_id:
                                prev_mask = preg.mask
                                break
                        if prev_mask is not None:
                            overlap = compute_gaussian_evidence_overlap(
                                self.previous_spread_ev, prev_mask,
                                ev_spread, reg.mask,
                            )

                    # Compute combined affinity cost
                    cost = 0.0
                    cost += self.config.w_spatial * (d_spatial / 10.0)
                    cost += self.config.w_pa * (pa_diff / 5.0)

                    if self.config.variant == "G3":
                        # Motion-aware prediction cost
                        pred_err = abs(reg.centroid_r - r_pred)
                        cost += self.config.w_motion * (pred_err / 5.0)
                    else:
                        # Standard radial continuity cost
                        cost += self.config.w_motion * max(0.0, -vr_instant * 2.0)

                    # Continuous overlap discount (higher overlap reduces cost)
                    cost += self.config.w_overlap * (1.0 - overlap) * 5.0

                    if cost < self.config.gating_cost_threshold:
                        cost_matrix[i, j] = cost

            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r_idx, c_idx in zip(row_ind, col_ind):
                if cost_matrix[r_idx, c_idx] < self.config.gating_cost_threshold:
                    tid = active_track_ids[r_idx]
                    reg = regions[c_idx]
                    matched_pairs.append((tid, reg.region_id))
                    matched_track_set.add(tid)
                    matched_region_set.add(c_idx)

                    # Update track with matched observation
                    trk = self.tracks[tid]
                    dr = reg.centroid_r - trk.latest.r
                    vr_instant = dr / dt_hours
                    # Smooth discovered radial velocity: EMA beta = 0.5
                    vr_smooth = (
                        0.5 * trk.current_vr + 0.5 * vr_instant
                        if trk.hits >= 2
                        else vr_instant
                    )

                    r_pred_val = (
                        trk.latest.r + trk.current_vr * dt_hours
                        if self.config.variant == "G3"
                        else None
                    )

                    obs = TrackObservation(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        dt_hours=dt_hours,
                        region_id=reg.region_id,
                        cx=reg.centroid_x,
                        cy=reg.centroid_y,
                        r=reg.centroid_r,
                        pa=reg.centroid_pa,
                        r_front=reg.r_front,
                        r_min=reg.r_min,
                        pa_span=reg.pa_span,
                        total_evidence=reg.total_evidence,
                        peak_evidence=reg.peak_evidence,
                        area_px=reg.area_px,
                        v_r_instant=vr_instant,
                        v_r_smoothed=vr_smooth,
                        predicted_r=r_pred_val,
                        association_cost=float(cost_matrix[r_idx, c_idx]),
                        overlap_score=0.0,
                    )
                    trk.observations.append(obs)
                    trk.missed_count = 0
                    trk.last_active_time = timestamp

                    # Promotion from tentative to confirmed
                    if trk.status == "tentative" and trk.hits >= self.config.min_confirm_hits:
                        trk.status = "confirmed"

        # Handle unmatched active tracks
        terminated_ids: List[int] = []
        for tid in active_track_ids:
            if tid not in matched_track_set:
                trk = self.tracks[tid]
                trk.missed_count += 1
                # Check termination conditions:
                # 1. Missed count threshold exceeded
                # 2. CME structure reached outer boundary
                if trk.missed_count >= self.config.max_missed_frames:
                    trk.status = "terminated"
                    terminated_ids.append(tid)
                elif trk.latest.r >= (self.config.r_max_px - 5.0):
                    trk.status = "terminated"
                    terminated_ids.append(tid)

        # Handle unmatched candidate regions -> Track Birth
        newly_born_ids: List[int] = []
        for j, reg in enumerate(regions):
            if j not in matched_region_set:
                # Apply birth rule
                allow_birth = True
                if self.config.birth_mode == "near_sun":
                    # Require initiation near inner coronagraph boundary
                    if reg.centroid_r > self.config.r_birth_max:
                        allow_birth = False

                if allow_birth:
                    tid = self.next_track_id
                    self.next_track_id += 1

                    init_status = "tentative" if self.config.min_confirm_hits > 1 else "confirmed"
                    new_track = GaussianTrack(
                        track_id=tid,
                        status=init_status,
                        missed_count=0,
                        last_active_time=timestamp,
                    )
                    obs = TrackObservation(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        dt_hours=dt_hours,
                        region_id=reg.region_id,
                        cx=reg.centroid_x,
                        cy=reg.centroid_y,
                        r=reg.centroid_r,
                        pa=reg.centroid_pa,
                        r_front=reg.r_front,
                        r_min=reg.r_min,
                        pa_span=reg.pa_span,
                        total_evidence=reg.total_evidence,
                        peak_evidence=reg.peak_evidence,
                        area_px=reg.area_px,
                        v_r_instant=0.0,
                        v_r_smoothed=0.0,
                        predicted_r=None,
                        association_cost=0.0,
                        overlap_score=0.0,
                    )
                    new_track.observations.append(obs)
                    self.tracks[tid] = new_track
                    newly_born_ids.append(tid)

        return matched_pairs, newly_born_ids, terminated_ids

    def get_confirmed_tracks(self) -> List[GaussianTrack]:
        """Returns all confirmed tracks (both active and terminated)."""
        return [trk for trk in self.tracks.values() if trk.status == "confirmed" or (trk.status == "terminated" and trk.hits >= self.config.min_confirm_hits)]

    def get_all_tracks(self) -> List[GaussianTrack]:
        """Returns all tracks."""
        return list(self.tracks.values())

    def to_dataframe(self) -> pd.DataFrame:
        """Exports full tracking history as a tabular DataFrame."""
        rows = []
        for tid, trk in self.tracks.items():
            for seq_idx, obs in enumerate(trk.observations):
                rows.append({
                    "track_id": tid,
                    "status": trk.status,
                    "obs_seq": seq_idx,
                    "frame_idx": obs.frame_idx,
                    "timestamp": obs.timestamp.isoformat() if isinstance(obs.timestamp, datetime) else str(obs.timestamp),
                    "dt_hours": obs.dt_hours,
                    "region_id": obs.region_id,
                    "cx": obs.cx,
                    "cy": obs.cy,
                    "centroid_r_px": obs.r,
                    "centroid_pa_deg": obs.pa,
                    "r_front_px": obs.r_front,
                    "r_min_px": obs.r_min,
                    "pa_span_deg": obs.pa_span,
                    "total_evidence": obs.total_evidence,
                    "peak_evidence": obs.peak_evidence,
                    "area_px": obs.area_px,
                    "v_r_instant_px_hr": obs.v_r_instant,
                    "v_r_smoothed_px_hr": obs.v_r_smoothed,
                    "predicted_r_px": obs.predicted_r,
                    "association_cost": obs.association_cost,
                    "mean_cpa": trk.mean_cpa,
                    "cpa_jitter": trk.cpa_jitter,
                    "outward_consistency": trk.outward_consistency,
                    "total_hits": trk.hits,
                })
        return pd.DataFrame(rows)
