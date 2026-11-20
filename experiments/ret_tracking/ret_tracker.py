r"""Radial Evidence Tracker (RET) Engine for Standalone CME Identification and Tracking.

Implements:
- RET-0: Radial Evidence Only (No temporal tracking, candidate discovery only)
- RET-1: Two-Frame Outward Confirmation (CPA compatibility, front outward validation, independent birth/continuation)
- RET-2: Track-Specific Motion Prediction (dynamically discovered velocity, time-aware association, r_pred = r + v*dt)
- RET-3: Multi-Lobe and Overlap Handling (concurrent same-direction lobes, split/merge diagnostics, identity preservation)

Completely independent of A5b, Seed Tracking, or G1/G3 track assignments.
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
from experiments.ret_tracking.ret_evidence_engine import (
    PolarGridGeometry,
    RadialCandidate,
    FrameRadialEvidenceResult,
    compute_pixel_evidence,
    apply_spatial_gaussian_spreading,
    sample_polar_evidence,
    accumulate_radial_evidence,
    detect_angular_candidates,
)


@dataclass
class RETConfig:
    """Configuration for Radial Evidence Tracker (RET)."""
    variant: str = "RET-1"  # 'RET-0', 'RET-1', 'RET-2', 'RET-3'
    sigma_s: float = 2.0  # Gaussian spreading scale (0.0 = raw)
    evidence_formulation: str = "positive"  # 'positive' or 'compressed'
    min_prominence: float = 3.0  # Angular peak prominence threshold
    min_separation_deg: float = 15.0  # Minimum angular separation between lobes
    min_lobe_width_deg: float = 8.0  # Minimum angular lobe width
    max_lobe_width_deg: float = 120.0  # Maximum angular lobe width
    angular_smooth_sigma_deg: float = 2.0  # Angular Gaussian smoothing scale
    normalize_by_length: bool = True  # Normalize radial accumulation by valid length
    front_evidence_quantile: float = 0.20  # Front detection relative threshold
    front_min_threshold: float = 0.35  # Absolute minimum evidence for front support
    min_confirm_hits: int = 2  # Detections required to confirm track (2 or 3)
    max_missed_frames: int = 2  # Max consecutive missed frames before termination
    max_dt_gap_min: float = 75.0  # Max observation gap before terminating active tracks
    max_pa_diff_deg: float = 20.0  # Max circular CPA difference allowed for association
    min_radial_velocity_px_hr: float = -6.0  # Outward motion tolerance (slight centroid noise)
    max_radial_velocity_px_hr: float = 120.0  # Physical CME speed upper bound in px/hr
    birth_mode: str = "near_sun"  # 'near_sun' (r_front <= r_birth_max) or 'unrestricted'
    r_birth_max: float = 48.0  # Max front radius for new track initiation (~5.8 R_sun)
    r_min_px: float = 22.0  # Inner occulter boundary
    r_max_px: float = 122.5  # Outer coronagraph FOV boundary
    # Cost weights for bipartite association
    w_pa: float = 1.5  # CPA angular deviation weight
    w_front: float = 1.0  # Front radial displacement / prediction error weight
    w_width: float = 0.5  # Angular width similarity weight
    gating_cost_threshold: float = 40.0  # Max acceptable association cost


@dataclass
class RETTrackObservation:
    """Individual observation hit associated with an RET track."""
    frame_idx: int
    timestamp: datetime
    dt_hours: float
    candidate_id: int
    cpa_deg: float
    peak_pa_deg: float
    pa_width_deg: float
    pa_start_deg: float
    pa_end_deg: float
    r_front_px: float
    r_peak_px: float
    r_min_px: float
    total_lobe_evidence: float
    peak_evidence: float
    prominence: float
    v_r_instant: float  # px/hr
    v_r_smoothed: float  # px/hr
    predicted_r: Optional[float] = None
    association_cost: float = 0.0


@dataclass
class RETTrack:
    """State of an independently discovered RET CME track."""
    track_id: int
    status: str  # 'tentative', 'confirmed', 'terminated'
    observations: List[RETTrackObservation] = field(default_factory=list)
    missed_count: int = 0
    last_active_time: Optional[datetime] = None

    @property
    def latest(self) -> RETTrackObservation:
        return self.observations[-1]

    @property
    def first(self) -> RETTrackObservation:
        return self.observations[0]

    @property
    def hits(self) -> int:
        return len(self.observations)

    @property
    def current_cpa(self) -> float:
        return self.latest.cpa_deg

    @property
    def current_r_front(self) -> float:
        return self.latest.r_front_px

    @property
    def current_vr(self) -> float:
        return self.latest.v_r_smoothed

    @property
    def mean_cpa(self) -> float:
        if not self.observations:
            return 0.0
        cpas = [obs.cpa_deg for obs in self.observations]
        return float(circular_mean_deg(np.array(cpas)))

    @property
    def cpa_jitter(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        m_cpa = self.mean_cpa
        diffs = [circular_diff_deg(obs.cpa_deg, m_cpa) for obs in self.observations]
        return float(np.std(diffs))

    @property
    def duration_hours(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        return (self.latest.timestamp - self.first.timestamp).total_seconds() / 3600.0

    @property
    def mean_radial_velocity(self) -> float:
        """Overall discovered radial front velocity in px/hr."""
        if len(self.observations) < 2:
            return 0.0
        dt_hr = self.duration_hours
        if dt_hr <= 0:
            return 0.0
        dr = self.latest.r_front_px - self.first.r_front_px
        return dr / dt_hr

    @property
    def outward_consistency(self) -> float:
        """Fraction of sequential observations with non-negative outward displacement."""
        if len(self.observations) < 2:
            return 1.0
        outward_steps = sum(1 for obs in self.observations[1:] if obs.v_r_instant >= -0.5)
        return float(outward_steps / (len(self.observations) - 1))


@dataclass
class FrameRETResult:
    """Per-frame diagnostic tracking output."""
    frame_idx: int
    timestamp: datetime
    dt_from_prev_min: float
    evidence_result: FrameRadialEvidenceResult
    active_confirmed_tracks: List[RETTrack]
    active_tentative_tracks: List[RETTrack]
    newly_born_tracks: List[int]
    matched_tracks: List[Tuple[int, int]]  # (track_id, candidate_id)
    terminated_tracks: List[int]


class RadialEvidenceTracker:
    """Standalone Radial Evidence Tracking (RET) Engine."""

    def __init__(self, config: RETConfig, grid: PolarGridGeometry):
        self.config = config
        self.grid = grid

        self.tracks: Dict[int, RETTrack] = {}
        self.next_track_id: int = 1
        self.previous_timestamp: Optional[datetime] = None

    def reset(self):
        """Resets tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.previous_timestamp = None

    def step(
        self,
        frame_idx: int,
        timestamp: datetime,
        z_map: np.ndarray,
        valid_mask: np.ndarray,
    ) -> FrameRETResult:
        """Processes one observation frame and updates track states."""
        # 1. Cadence and observation gap evaluation
        dt_hours = 0.5
        dt_min = 30.0
        if self.previous_timestamp is not None:
            dt_sec = (timestamp - self.previous_timestamp).total_seconds()
            dt_hours = max(0.01, dt_sec / 3600.0)
            dt_min = dt_sec / 60.0

        # Terminate active tracks across large data outage
        if self.previous_timestamp is not None and dt_min > self.config.max_dt_gap_min:
            for trk in self.tracks.values():
                if trk.status in ("tentative", "confirmed"):
                    trk.status = "terminated"

        # 2. Compute pixel-level evidence E_t(x, y)
        ev_raw = compute_pixel_evidence(
            z_map=z_map,
            valid_mask=valid_mask,
            formulation=self.config.evidence_formulation,
        )

        # 3. Mask-normalized 2D Gaussian convolution E_tilde_t(x, y)
        ev_spread = apply_spatial_gaussian_spreading(
            evidence=ev_raw,
            valid_mask=valid_mask,
            sigma_s=self.config.sigma_s,
        )

        # 4. Transform evidence into polar coordinates E_tilde(r, theta)
        polar_ev, polar_val = sample_polar_evidence(
            spread_ev=ev_spread,
            valid_mask=valid_mask,
            grid=self.grid,
        )

        # 5. Radial ray accumulation A_t(theta)
        a_raw, a_smooth, val_len = accumulate_radial_evidence(
            polar_ev=polar_ev,
            polar_val=polar_val,
            grid=self.grid,
            angular_smooth_sigma_deg=self.config.angular_smooth_sigma_deg,
            normalize_by_length=self.config.normalize_by_length,
        )

        # 6. Detect angular evidence peaks and extract candidate radial fronts
        candidates = detect_angular_candidates(
            a_smooth=a_smooth,
            polar_ev=polar_ev,
            grid=self.grid,
            min_prominence=self.config.min_prominence,
            min_separation_deg=self.config.min_separation_deg,
            min_lobe_width_deg=self.config.min_lobe_width_deg,
            max_lobe_width_deg=self.config.max_lobe_width_deg,
            front_evidence_quantile=self.config.front_evidence_quantile,
            front_min_threshold=self.config.front_min_threshold,
        )

        ev_result = FrameRadialEvidenceResult(
            frame_idx=frame_idx,
            raw_evidence=ev_raw,
            spread_evidence=ev_spread,
            polar_evidence=polar_ev,
            polar_valid=polar_val,
            angular_evidence_raw=a_raw,
            angular_evidence_smoothed=a_smooth,
            valid_radial_support=val_len,
            candidates=candidates,
        )

        # In RET-0, do not perform temporal tracking
        if self.config.variant == "RET-0":
            self.previous_timestamp = timestamp
            return FrameRETResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                dt_from_prev_min=dt_min,
                evidence_result=ev_result,
                active_confirmed_tracks=[],
                active_tentative_tracks=[],
                newly_born_tracks=[],
                matched_tracks=[],
                terminated_tracks=[],
            )

        # 7. Associate active tracks with detected angular candidates
        active_track_ids = [
            tid for tid, trk in self.tracks.items()
            if trk.status in ("tentative", "confirmed")
        ]

        matched_tracks, newly_born, terminated = self._associate_and_update(
            frame_idx=frame_idx,
            timestamp=timestamp,
            dt_hours=dt_hours,
            active_track_ids=active_track_ids,
            candidates=candidates,
        )

        confirmed = [trk for trk in self.tracks.values() if trk.status == "confirmed"]
        tentative = [trk for trk in self.tracks.values() if trk.status == "tentative"]

        self.previous_timestamp = timestamp

        return FrameRETResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            dt_from_prev_min=dt_min,
            evidence_result=ev_result,
            active_confirmed_tracks=confirmed,
            active_tentative_tracks=tentative,
            newly_born_tracks=newly_born,
            matched_tracks=matched_tracks,
            terminated_tracks=terminated,
        )

    def _associate_and_update(
        self,
        frame_idx: int,
        timestamp: datetime,
        dt_hours: float,
        active_track_ids: List[int],
        candidates: List[RadialCandidate],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Bipartite matching between existing tracks and angular candidates."""
        n_tracks = len(active_track_ids)
        n_cands = len(candidates)

        matched_pairs: List[Tuple[int, int]] = []
        matched_track_set = set()
        matched_cand_set = set()

        if n_tracks > 0 and n_cands > 0:
            cost_matrix = np.full((n_tracks, n_cands), 1e6, dtype=np.float64)

            for i, tid in enumerate(active_track_ids):
                trk = self.tracks[tid]
                last_obs = trk.latest

                # Predict expected front radius
                if self.config.variant in ("RET-2", "RET-3"):
                    # Discovered motion prediction
                    v_pred = trk.current_vr if abs(trk.current_vr) > 0.1 else 10.0
                    r_pred = last_obs.r_front_px + v_pred * dt_hours
                else:
                    # RET-1: static front continuity
                    r_pred = last_obs.r_front_px

                for j, cand in enumerate(candidates):
                    # 1. Circular CPA continuity gate
                    d_cpa = abs(circular_diff_deg(cand.weighted_cpa_deg, last_obs.cpa_deg))
                    if d_cpa > self.config.max_pa_diff_deg:
                        continue

                    # 2. Front radial displacement gate
                    dr = cand.r_front_px - last_obs.r_front_px
                    vr_instant = dr / dt_hours

                    if vr_instant < self.config.min_radial_velocity_px_hr:
                        continue  # Inward collapse violation
                    if vr_instant > self.config.max_radial_velocity_px_hr:
                        continue  # Unphysically fast outward jump

                    # 3. Multi-lobe disambiguation (RET-3)
                    # If candidate front is significantly behind last observed front (> 8 px),
                    # and another candidate has an outward front, penalize or prevent association
                    if self.config.variant == "RET-3" and dr < -2.0:
                        continue

                    # Calculate combined affinity cost
                    cost = 0.0
                    cost += self.config.w_pa * (d_cpa / 5.0)

                    if self.config.variant in ("RET-2", "RET-3"):
                        pred_err = abs(cand.r_front_px - r_pred)
                        cost += self.config.w_front * (pred_err / 5.0)
                    else:
                        # RET-1 outward motion cost
                        cost += self.config.w_front * max(0.0, -vr_instant * 2.0)

                    # Angular width difference penalty
                    w_diff = abs(cand.pa_width_deg - last_obs.pa_width_deg)
                    cost += self.config.w_width * (w_diff / 10.0)

                    if cost < self.config.gating_cost_threshold:
                        cost_matrix[i, j] = cost

            # Solve linear assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r_idx, c_idx in zip(row_ind, col_ind):
                if cost_matrix[r_idx, c_idx] < self.config.gating_cost_threshold:
                    tid = active_track_ids[r_idx]
                    cand = candidates[c_idx]
                    matched_pairs.append((tid, cand.candidate_id))
                    matched_track_set.add(tid)
                    matched_cand_set.add(c_idx)

                    # Update track state
                    trk = self.tracks[tid]
                    dr = cand.r_front_px - trk.latest.r_front_px
                    vr_instant = dr / dt_hours
                    vr_smooth = (
                        0.5 * trk.current_vr + 0.5 * vr_instant
                        if trk.hits >= 2
                        else vr_instant
                    )

                    r_pred_val = (
                        trk.latest.r_front_px + trk.current_vr * dt_hours
                        if self.config.variant in ("RET-2", "RET-3")
                        else None
                    )

                    obs = RETTrackObservation(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        dt_hours=dt_hours,
                        candidate_id=cand.candidate_id,
                        cpa_deg=cand.weighted_cpa_deg,
                        peak_pa_deg=cand.peak_pa_deg,
                        pa_width_deg=cand.pa_width_deg,
                        pa_start_deg=cand.pa_start_deg,
                        pa_end_deg=cand.pa_end_deg,
                        r_front_px=cand.r_front_px,
                        r_peak_px=cand.r_peak_px,
                        r_min_px=cand.r_min_px,
                        total_lobe_evidence=cand.total_lobe_evidence,
                        peak_evidence=cand.peak_evidence,
                        prominence=cand.prominence,
                        v_r_instant=vr_instant,
                        v_r_smoothed=vr_smooth,
                        predicted_r=r_pred_val,
                        association_cost=float(cost_matrix[r_idx, c_idx]),
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
                if trk.missed_count >= self.config.max_missed_frames:
                    trk.status = "terminated"
                    terminated_ids.append(tid)
                elif trk.latest.r_front_px >= (self.config.r_max_px - 5.0):
                    trk.status = "terminated"
                    terminated_ids.append(tid)

        # Handle unmatched candidate lobes -> Track Birth
        newly_born_ids: List[int] = []
        for j, cand in enumerate(candidates):
            if j not in matched_cand_set:
                allow_birth = True
                if self.config.birth_mode == "near_sun":
                    # Require front to be within birth zone
                    if cand.r_front_px > self.config.r_birth_max:
                        allow_birth = False

                if allow_birth:
                    tid = self.next_track_id
                    self.next_track_id += 1

                    init_status = "tentative" if self.config.min_confirm_hits > 1 else "confirmed"
                    new_track = RETTrack(
                        track_id=tid,
                        status=init_status,
                        missed_count=0,
                        last_active_time=timestamp,
                    )
                    obs = RETTrackObservation(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        dt_hours=dt_hours,
                        candidate_id=cand.candidate_id,
                        cpa_deg=cand.weighted_cpa_deg,
                        peak_pa_deg=cand.peak_pa_deg,
                        pa_width_deg=cand.pa_width_deg,
                        pa_start_deg=cand.pa_start_deg,
                        pa_end_deg=cand.pa_end_deg,
                        r_front_px=cand.r_front_px,
                        r_peak_px=cand.r_peak_px,
                        r_min_px=cand.r_min_px,
                        total_lobe_evidence=cand.total_lobe_evidence,
                        peak_evidence=cand.peak_evidence,
                        prominence=cand.prominence,
                        v_r_instant=0.0,
                        v_r_smoothed=0.0,
                        predicted_r=None,
                        association_cost=0.0,
                    )
                    new_track.observations.append(obs)
                    self.tracks[tid] = new_track
                    newly_born_ids.append(tid)

        return matched_pairs, newly_born_ids, terminated_ids

    def get_confirmed_tracks(self) -> List[RETTrack]:
        """Returns all confirmed tracks (active and terminated)."""
        return [
            trk for trk in self.tracks.values()
            if trk.status == "confirmed" or (trk.status == "terminated" and trk.hits >= self.config.min_confirm_hits)
        ]

    def get_all_tracks(self) -> List[RETTrack]:
        return list(self.tracks.values())

    def to_dataframe(self) -> pd.DataFrame:
        """Exports full tracking history to pandas DataFrame."""
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
                    "candidate_id": obs.candidate_id,
                    "cpa_deg": obs.cpa_deg,
                    "peak_pa_deg": obs.peak_pa_deg,
                    "pa_width_deg": obs.pa_width_deg,
                    "pa_start_deg": obs.pa_start_deg,
                    "pa_end_deg": obs.pa_end_deg,
                    "r_front_px": obs.r_front_px,
                    "r_peak_px": obs.r_peak_px,
                    "r_min_px": obs.r_min_px,
                    "total_lobe_evidence": obs.total_lobe_evidence,
                    "peak_evidence": obs.peak_evidence,
                    "prominence": obs.prominence,
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
