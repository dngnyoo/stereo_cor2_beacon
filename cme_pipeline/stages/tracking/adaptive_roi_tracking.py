"""Birth-Seeded Adaptive PA-ROI Tracking Stage for CME Pipeline.

Implements:
- Birth Seed detection (birth_r_min <= 30.0 px)
- Immutable Original Birth PA and Immutable Original Birth CPA
- Initial PA Angular Extent & Initial Birth ROI
- Adaptive Active ROI expansion based on 8-connected patch growth
- Overlapping ROI patch ownership resolution (minimum circular distance to previous tracked_cpa)
- Patch conflict resolution logging (tie-breaking, AMBIGUOUS handling)
- Post-distribution 8-connectivity re-examination
- Post-assignment CPA EMA update (alpha = 0.1, strict non-circular causality)
- Track-specific Height-Time Map (HTM) profile extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import GeometryContext, PipelineStage, TrackingOutput
from cme_pipeline.stages.tracking.e1_tracking import E1TrackingStage
from cme_pipeline.utils.geometry import (
    circular_diff_deg,
    circular_mean_deg,
    minimal_circular_enclosing_arc,
)
from cme_pipeline.utils.io import compute_dict_hash

STRUCT_8 = np.ones((3, 3), bool)
MAX_CADENCE_GAP_SEC = 4500.0  # 75 min
BIRTH_GATE_THRESHOLD_PX = 30.0
ALPHA_CPA_EMA = 0.1
DEFAULT_ROI_HALF_WIDTH_DEG = 15.0
ROI_PADDING_DEG = 5.0
MAX_MISSED_FRAMES = 3


@dataclass
class HTMRecord:
    """Radial activity profile and kinematic metrics for a single frame."""
    pair_index: int
    timestamp: datetime
    raw_radial_activity: np.ndarray        # Histogram of active patches in PA ROI
    accepted_connected_patches: np.ndarray # Histogram of 8-connected patches
    assigned_track_components: np.ndarray  # Histogram of final owned patches
    observed_radial_front_px: float       # Max radius of observed patches
    observed_radial_min_px: float         # Min radius of observed patches
    temporal_continuity_score: float      # Overlap / kinematic continuity


@dataclass
class AdaptiveTrack:
    """Adaptive PA-ROI Track object with full lifecycle and HTM history."""
    track_id: int
    birth_pair_id: int
    birth_time: datetime
    birth_r_min_px: float
    is_anchor: bool

    # Immutable Birth Geometry
    original_birth_pa: float          # Direction of min-radius solar root patch (Immutable)
    original_birth_cpa: float         # Central PA of seed component at birth (Immutable)
    initial_angular_extent: float     # Initial angular span (deg)
    initial_birth_roi: Tuple[float, float] # (center_pa, half_width_deg)

    # Dynamic / Adaptive State
    adaptive_active_roi: Tuple[float, float] # (center_pa, half_width_deg)
    tracked_cpa: float                # EMA smoothed CPA (alpha=0.1)
    observed_cpa: float               # Instantaneous CPA of current observed component

    # History
    history: List[Dict[str, Any]] = field(default_factory=list)
    htm_history: List[HTMRecord] = field(default_factory=list)
    missed: int = 0
    terminated: bool = False


@dataclass
class PatchConflictRecord:
    """Record of patch ownership conflict resolution."""
    pair_index: int
    patch_r: int
    patch_c: int
    patch_pa: float
    patch_rad: float
    candidate_track_ids: List[int]
    winner_track_id: int
    min_pa_diff: float
    tie_break_applied: str
    status: str  # RESOLVED or AMBIGUOUS


def compute_radial_histogram(radii: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Computes histogram of radial positions along radial bins."""
    if len(radii) == 0:
        return np.zeros(len(bins) - 1, dtype=int)
    counts, _ = np.histogram(radii, bins=bins)
    return counts


class BirthSeededAdaptiveROITrackingStage(PipelineStage):
    """Birth-Seeded Adaptive PA-ROI Tracking Stage."""
    name: str = "tracking"
    version: str = "3.0.0"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": "adaptive_pa_roi",
            "alpha": ALPHA_CPA_EMA,
            "date_str": config.date_str,
            "outward_gate_enabled": getattr(config.tracking, "outward_gate_enabled", False),
            "outward_gate_tolerance_px": getattr(config.tracking, "outward_gate_tolerance_px", 4.0),
        }
        return compute_dict_hash(desc)

    def run(self, inputs: Any, config: PipelineConfig, context: PipelineContext) -> TrackingOutput:
        context.info("Running BirthSeededAdaptiveROITrackingStage (Adaptive PA-ROI Tracking)...")

        # 1. Base E1 Preprocessing (Frames, WCS, Geometry, Thresholds)
        base_stage = E1TrackingStage()
        base_out = base_stage.run(inputs, config, context)
        geom = base_out.geometry
        patch_pa = geom.patch_pa
        patch_rad = geom.patch_rad
        man_df = base_out.manifest_df
        grid4_all = base_out.patch_activation_grid4

        n_pairs = len(man_df)
        radial_bins = np.linspace(15.0, 130.0, 47)  # 2.5 px bins from occulter to FOV

        tracks: Dict[int, AdaptiveTrack] = {}
        next_track_id = 1

        all_conflict_records: List[PatchConflictRecord] = []
        all_rejection_records: List[Dict[str, Any]] = []
        track_state_rows: List[Dict[str, Any]] = []

        # 3D arrays to store per-frame track and component assignments
        patch_component_labels = np.zeros((n_pairs, 64, 64), dtype=np.int32)
        track_ownership_masks = np.zeros((n_pairs, 64, 64), dtype=np.int32)

        prev_time = None

        for step_idx in range(n_pairs):
            p_curr = int(man_df.iloc[step_idx]["pair_index"])
            ts_curr = datetime.fromisoformat(man_df.iloc[step_idx]["end_time"])
            cur_grid = grid4_all[step_idx]

            # Cadence gap check
            if prev_time is not None:
                dt_sec = (ts_curr - prev_time).total_seconds()
                if dt_sec > MAX_CADENCE_GAP_SEC:
                    context.info(f"Large cadence gap detected before Pair {p_curr} ({dt_sec/60:.1f} min). Terminating active tracks.")
                    for t in tracks.values():
                        if not t.terminated:
                            t.terminated = True
            prev_time = ts_curr

            # -------------------------------------------------------------
            # Step A: Identify candidates for existing active tracks
            # -------------------------------------------------------------
            active_tracks = [t for t in tracks.values() if not t.terminated]
            # Map of (r, c) -> list of candidate track IDs
            patch_candidates: Dict[Tuple[int, int], List[int]] = {}

            # Active patch coordinates in current frame
            active_rcs = list(zip(*np.nonzero(cur_grid)))

            for t in active_tracks:
                roi_center, roi_half = t.adaptive_active_roi
                last_obs = next((h for h in reversed(t.history) if h["state"] == "OBSERVED"), None)
                prev_mask = last_obs["mask"] if last_obs is not None else np.zeros((64, 64), bool)
                prev_front = last_obs["front_radius"] if last_obs is not None else 125.0
                prev_rmin = last_obs["radial_min"] if last_obs is not None else 15.0

                for r, c in active_rcs:
                    ppa = patch_pa[r, c]
                    prad = patch_rad[r, c]

                    # 1. Inside Adaptive PA ROI
                    pa_diff = abs(circular_diff_deg(ppa, roi_center))
                    if pa_diff <= roi_half:
                        # 2. Reasonable radial expansion corridor
                        # (Allow outward expansion and slight inward leeway for flank growth)
                        if prad >= (prev_rmin - 8.0) and prad <= (prev_front + 30.0):
                            patch_candidates.setdefault((r, c), []).append(t.track_id)
                        # 3. Spatial adjacency: 8-connected neighbor of previous mask
                        elif prev_mask[max(0, r-1):min(64, r+2), max(0, c-1):min(64, c+2)].any():
                            patch_candidates.setdefault((r, c), []).append(t.track_id)

            # -------------------------------------------------------------
            # Step B: Overlapping ROI Ownership & Conflict Resolution
            # -------------------------------------------------------------
            assigned_patches_by_track: Dict[int, List[Tuple[int, int]]] = {t.track_id: [] for t in active_tracks}

            for (r, c), cand_tids in patch_candidates.items():
                ppa = patch_pa[r, c]
                prad = patch_rad[r, c]

                if len(cand_tids) == 1:
                    # Single candidate: uncontested ownership
                    assigned_patches_by_track[cand_tids[0]].append((r, c))
                else:
                    # Multiple candidate tracks: resolve based on previous tracked_cpa
                    cand_diffs = []
                    for tid in cand_tids:
                        tr = tracks[tid]
                        diff = abs(circular_diff_deg(ppa, tr.tracked_cpa))
                        cand_diffs.append((diff, tid))

                    cand_diffs.sort(key=lambda x: x[0])
                    min_diff, winner_tid = cand_diffs[0]
                    runner_up_diff, runner_up_tid = cand_diffs[1]

                    tie_break = "NONE"
                    status = "RESOLVED"

                    # Tie-break if difference is negligible (< 0.2 deg)
                    if abs(runner_up_diff - min_diff) < 0.2:
                        # Priority 1: Check previous frame ownership
                        prev_winner_obs = next((h for h in reversed(tracks[winner_tid].history) if h["state"] == "OBSERVED"), None)
                        prev_runner_obs = next((h for h in reversed(tracks[runner_up_tid].history) if h["state"] == "OBSERVED"), None)

                        win_owned = prev_winner_obs is not None and prev_winner_obs["mask"][r, c]
                        run_owned = prev_runner_obs is not None and prev_runner_obs["mask"][r, c]

                        if run_owned and not win_owned:
                            winner_tid = runner_up_tid
                            tie_break = "PREVIOUS_FRAME_OWNERSHIP"
                        elif win_owned and not run_owned:
                            tie_break = "PREVIOUS_FRAME_OWNERSHIP"
                        else:
                            # Priority 2: Earlier birth / temporal seniority
                            if tracks[runner_up_tid].birth_pair_id < tracks[winner_tid].birth_pair_id:
                                winner_tid = runner_up_tid
                                tie_break = "TEMPORAL_SENIORITY"
                            else:
                                tie_break = "CLOSEST_CPA"
                            status = "AMBIGUOUS"

                    assigned_patches_by_track[winner_tid].append((r, c))

                    all_conflict_records.append(
                        PatchConflictRecord(
                            pair_index=p_curr,
                            patch_r=r,
                            patch_c=c,
                            patch_pa=float(ppa),
                            patch_rad=float(prad),
                            candidate_track_ids=cand_tids,
                            winner_track_id=winner_tid,
                            min_pa_diff=float(min_diff),
                            tie_break_applied=tie_break,
                            status=status,
                        )
                    )

            # -------------------------------------------------------------
            # Step C: 8-Connected Growth & Out-of-ROI Expansion
            # -------------------------------------------------------------
            # Create a label map for active components on full grid4
            lbl_img, n_comps = label(cur_grid, structure=STRUCT_8)

            claimed_patches_this_frame = set()

            for t in active_tracks:
                seed_coords = assigned_patches_by_track.get(t.track_id, [])
                if not seed_coords:
                    # Missed detection
                    t.missed += 1
                    if t.missed >= MAX_MISSED_FRAMES:
                        t.terminated = True
                    # Record PREDICTED_ONLY state
                    last_obs = next((h for h in reversed(t.history) if h["state"] == "OBSERVED"), None)
                    last_cpa = last_obs["central_pa"] if last_obs is not None else t.tracked_cpa
                    last_front = last_obs["front_radius"] if last_obs is not None else 125.0

                    track_state_rows.append({
                        "experiment": "Adaptive_PA_ROI_Tracking",
                        "track_id": t.track_id,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "component_id": -1,
                        "component_uid": f"p{p_curr}_pred",
                        "state": "PREDICTED_ONLY",
                        "central_pa": last_cpa,
                        "tracked_cpa": t.tracked_cpa,
                        "roi_center_pa": t.adaptive_active_roi[0],
                        "roi_half_width": t.adaptive_active_roi[1],
                        "pa_min": t.adaptive_active_roi[0] - t.adaptive_active_roi[1],
                        "pa_max": t.adaptive_active_roi[0] + t.adaptive_active_roi[1],
                        "radial_front": last_front,
                        "front_radius": last_front,
                        "radial_min": last_obs["radial_min"] if last_obs is not None else 20.0,
                        "area_pixels": 0,
                        "num_patches": 0,
                        "birth_r_min_px": t.birth_r_min_px,
                        "original_birth_pa": t.original_birth_pa,
                        "original_birth_cpa": t.original_birth_cpa,
                        "is_solar_root_anchor": t.is_anchor,
                        "outward_gate_evaluated": False,
                        "outward_gate_passed": True,
                        "radial_displacement": 0.0,
                    })
                    continue

                t.missed = 0

                # Identify which 8-connected components on full grid overlap the seed coordinates
                comp_ids = set(lbl_img[r, c] for r, c in seed_coords if lbl_img[r, c] > 0)
                # Expand 8-connectivity to all patches of these components
                expanded_mask = np.isin(lbl_img, list(comp_ids))

                # Ensure we do not usurp patches already uniquely won by another track's core
                for other_tid, other_coords in assigned_patches_by_track.items():
                    if other_tid != t.track_id:
                        for orc in other_coords:
                            if orc not in seed_coords:
                                expanded_mask[orc] = False

                # ---------------------------------------------------------
                # Step D: Post-Distribution 8-Connectivity Re-check
                # ---------------------------------------------------------
                # Check connected components within expanded_mask
                sub_lbl, n_sub = label(expanded_mask, structure=STRUCT_8)
                if n_sub > 1:
                    # Select the primary cluster containing the largest number of seeds
                    best_sub_id = 1
                    best_seed_count = 0
                    for s_id in range(1, n_sub + 1):
                        cnt = sum(1 for (r, c) in seed_coords if sub_lbl[r, c] == s_id)
                        if cnt > best_seed_count:
                            best_seed_count = cnt
                            best_sub_id = s_id
                    expanded_mask = (sub_lbl == best_sub_id)

                if expanded_mask.sum() == 0:
                    t.missed += 1
                    if t.missed >= MAX_MISSED_FRAMES:
                        t.terminated = True
                    last_obs = next((h for h in reversed(t.history) if h["state"] == "OBSERVED"), None)
                    last_cpa = last_obs["central_pa"] if last_obs is not None else t.tracked_cpa
                    last_front = last_obs["front_radius"] if last_obs is not None else 125.0
                    track_state_rows.append({
                        "experiment": "Adaptive_PA_ROI_Tracking",
                        "track_id": t.track_id,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "component_id": -1,
                        "component_uid": f"p{p_curr}_pred",
                        "state": "PREDICTED_ONLY",
                        "central_pa": last_cpa,
                        "tracked_cpa": t.tracked_cpa,
                        "roi_center_pa": t.adaptive_active_roi[0],
                        "roi_half_width": t.adaptive_active_roi[1],
                        "pa_min": t.adaptive_active_roi[0] - t.adaptive_active_roi[1],
                        "pa_max": t.adaptive_active_roi[0] + t.adaptive_active_roi[1],
                        "radial_front": last_front,
                        "front_radius": last_front,
                        "radial_min": last_obs["radial_min"] if last_obs is not None else 20.0,
                        "area_pixels": 0,
                        "num_patches": 0,
                        "birth_r_min_px": t.birth_r_min_px,
                        "original_birth_pa": t.original_birth_pa,
                        "original_birth_cpa": t.original_birth_cpa,
                        "is_solar_root_anchor": t.is_anchor,
                        "outward_gate_evaluated": False,
                        "outward_gate_passed": True,
                        "radial_displacement": 0.0,
                    })
                    continue

                # Geometry and Extent of candidate component
                comp_pas = patch_pa[expanded_mask]
                comp_rads = patch_rad[expanded_mask]
                arc = minimal_circular_enclosing_arc(comp_pas)
                obs_cpa = float(circular_mean_deg(comp_pas))
                obs_span = float(arc["arc_span"])
                obs_front = float(np.max(comp_rads))
                obs_min_rad = float(np.min(comp_rads))

                # Outward motion gate check
                outward_enabled = getattr(config.tracking, "outward_gate_enabled", False)
                outward_tol = getattr(config.tracking, "outward_gate_tolerance_px", 4.0)

                last_obs = next((h for h in reversed(t.history) if h["state"] == "OBSERVED"), None)
                gate_passed = True
                radial_disp = 0.0

                if last_obs is not None:
                    prev_front = float(last_obs["front_radius"])
                    radial_disp = obs_front - prev_front
                    if outward_enabled and radial_disp < -outward_tol:
                        gate_passed = False
                        all_rejection_records.append({
                            "pair_index": p_curr,
                            "timestamp": ts_curr.isoformat(),
                            "track_id": t.track_id,
                            "prev_front_radius": prev_front,
                            "cand_front_radius": obs_front,
                            "radial_displacement": radial_disp,
                            "tolerance_px": outward_tol,
                            "cand_num_patches": int(expanded_mask.sum()),
                            "cand_cpa": obs_cpa,
                            "action": "REJECTED_OUTWARD_GATE",
                        })

                if not gate_passed:
                    t.missed += 1
                    if t.missed >= MAX_MISSED_FRAMES:
                        t.terminated = True
                    last_cpa = last_obs["central_pa"] if last_obs is not None else t.tracked_cpa
                    last_front = last_obs["front_radius"] if last_obs is not None else 125.0

                    track_state_rows.append({
                        "experiment": "Adaptive_PA_ROI_Tracking",
                        "track_id": t.track_id,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "component_id": -1,
                        "component_uid": f"p{p_curr}_pred",
                        "state": "PREDICTED_ONLY",
                        "central_pa": last_cpa,
                        "tracked_cpa": t.tracked_cpa,
                        "roi_center_pa": t.adaptive_active_roi[0],
                        "roi_half_width": t.adaptive_active_roi[1],
                        "pa_min": t.adaptive_active_roi[0] - t.adaptive_active_roi[1],
                        "pa_max": t.adaptive_active_roi[0] + t.adaptive_active_roi[1],
                        "radial_front": last_front,
                        "front_radius": last_front,
                        "radial_min": last_obs["radial_min"] if last_obs is not None else 20.0,
                        "area_pixels": 0,
                        "num_patches": 0,
                        "birth_r_min_px": t.birth_r_min_px,
                        "original_birth_pa": t.original_birth_pa,
                        "original_birth_cpa": t.original_birth_cpa,
                        "is_solar_root_anchor": t.is_anchor,
                        "outward_gate_evaluated": True,
                        "outward_gate_passed": False,
                        "radial_displacement": radial_disp,
                    })
                    continue

                # Final assigned patch mask for track t in this frame
                t_patch_mask = expanded_mask
                t_patch_coords = list(zip(*np.nonzero(t_patch_mask)))
                for rc in t_patch_coords:
                    claimed_patches_this_frame.add(rc)

                # ---------------------------------------------------------
                # Step E: Update Adaptive Active ROI
                # ---------------------------------------------------------
                # ROI adapts to actual observed PA extent with padding
                new_roi_half = max(DEFAULT_ROI_HALF_WIDTH_DEG, obs_span / 2.0 + ROI_PADDING_DEG)
                t.adaptive_active_roi = (obs_cpa, new_roi_half)

                # ---------------------------------------------------------
                # Step F: Update CPA (EMA alpha = 0.1, strict post-assignment)
                # ---------------------------------------------------------
                cpa_diff = circular_diff_deg(obs_cpa, t.tracked_cpa)
                t.tracked_cpa = float((t.tracked_cpa + ALPHA_CPA_EMA * cpa_diff) % 360.0)
                t.observed_cpa = obs_cpa

                # Record in 3D masks
                track_ownership_masks[step_idx][t_patch_mask] = t.track_id
                patch_component_labels[step_idx][t_patch_mask] = t.track_id

                # Save history
                obs_record = {
                    "pair_index": p_curr,
                    "timestamp": ts_curr,
                    "state": "OBSERVED",
                    "mask": t_patch_mask,
                    "central_pa": obs_cpa,
                    "tracked_cpa": t.tracked_cpa,
                    "front_radius": obs_front,
                    "radial_min": obs_min_rad,
                    "num_patches": len(t_patch_coords),
                    "area_pixels": len(t_patch_coords) * 16,
                }
                t.history.append(obs_record)

                # ---------------------------------------------------------
                # Step G: Track-Specific HTM Data Accumulation
                # ---------------------------------------------------------
                # 1. Raw radial activity in current adaptive ROI
                roi_center, roi_half = t.adaptive_active_roi
                in_roi_mask = cur_grid & (np.abs(circular_diff_deg(patch_pa, roi_center)) <= roi_half)
                raw_hist = compute_radial_histogram(patch_rad[in_roi_mask], radial_bins)

                # 2. Accepted connected patches
                conn_hist = compute_radial_histogram(patch_rad[expanded_mask], radial_bins)

                # 3. Assigned track component patches
                assigned_hist = compute_radial_histogram(comp_rads, radial_bins)

                # 4. Temporal continuity score
                last_two = [h for h in t.history if h["state"] == "OBSERVED"]
                if len(last_two) >= 2:
                    m_prev = last_two[-2]["mask"]
                    m_curr = last_two[-1]["mask"]
                    # Jaccard overlap or 8-connectivity adjacency
                    inter = np.logical_and(m_prev, m_curr).sum()
                    union = np.logical_or(m_prev, m_curr).sum()
                    cont_score = float(inter / max(1, union))
                else:
                    cont_score = 1.0

                htm_rec = HTMRecord(
                    pair_index=p_curr,
                    timestamp=ts_curr,
                    raw_radial_activity=raw_hist,
                    accepted_connected_patches=conn_hist,
                    assigned_track_components=assigned_hist,
                    observed_radial_front_px=obs_front,
                    observed_radial_min_px=obs_min_rad,
                    temporal_continuity_score=cont_score,
                )
                t.htm_history.append(htm_rec)

                track_state_rows.append({
                    "experiment": "Adaptive_PA_ROI_Tracking",
                    "track_id": t.track_id,
                    "pair_index": p_curr,
                    "timestamp": ts_curr.isoformat(),
                    "component_id": t.track_id,
                    "component_uid": f"p{p_curr}_t{t.track_id}",
                    "state": "OBSERVED",
                    "central_pa": obs_cpa,
                    "tracked_cpa": t.tracked_cpa,
                    "roi_center_pa": t.adaptive_active_roi[0],
                    "roi_half_width": t.adaptive_active_roi[1],
                    "pa_min": float(arc["arc_start"]),
                    "pa_max": float(arc["arc_end"]),
                    "radial_front": obs_front,
                    "front_radius": obs_front,
                    "radial_min": obs_min_rad,
                    "area_pixels": len(t_patch_coords) * 16,
                    "num_patches": len(t_patch_coords),
                    "birth_r_min_px": t.birth_r_min_px,
                    "original_birth_pa": t.original_birth_pa,
                    "original_birth_cpa": t.original_birth_cpa,
                    "is_solar_root_anchor": t.is_anchor,
                    "outward_gate_evaluated": outward_enabled and (last_obs is not None),
                    "outward_gate_passed": True,
                    "radial_displacement": radial_disp,
                })

            # -------------------------------------------------------------
            # Step H: Birth Initialization for Unclaimed Patches
            # -------------------------------------------------------------
            unclaimed_grid = cur_grid.copy()
            for (r, c) in claimed_patches_this_frame:
                unclaimed_grid[r, c] = False

            uncl_lbl, n_uncl = label(unclaimed_grid, structure=STRUCT_8)
            for c_id in range(1, n_uncl + 1):
                c_mask = (uncl_lbl == c_id)
                c_rads = patch_rad[c_mask]
                c_pas = patch_pa[c_mask]
                min_r = float(np.min(c_rads))

                # Birth Seed Qualification: birth_r_min <= 30.0 px
                if min_r <= BIRTH_GATE_THRESHOLD_PX:
                    tid = next_track_id
                    next_track_id += 1

                    # Solar root patch direction
                    idx_min = int(np.argmin(c_rads))
                    orig_birth_pa = float(c_pas[idx_min])
                    orig_birth_cpa = float(circular_mean_deg(c_pas))

                    arc_birth = minimal_circular_enclosing_arc(c_pas)
                    init_extent = float(arc_birth["arc_span"])
                    init_roi_half = max(DEFAULT_ROI_HALF_WIDTH_DEG, init_extent / 2.0 + ROI_PADDING_DEG)
                    init_roi = (orig_birth_cpa, init_roi_half)

                    new_track = AdaptiveTrack(
                        track_id=tid,
                        birth_pair_id=p_curr,
                        birth_time=ts_curr,
                        birth_r_min_px=min_r,
                        is_anchor=True,
                        original_birth_pa=orig_birth_pa,
                        original_birth_cpa=orig_birth_cpa,
                        initial_angular_extent=init_extent,
                        initial_birth_roi=init_roi,
                        adaptive_active_roi=init_roi,
                        tracked_cpa=orig_birth_cpa,
                        observed_cpa=orig_birth_cpa,
                    )

                    # Initial observation record
                    obs_front = float(np.max(c_rads))
                    n_pts = int(c_mask.sum())
                    new_track.history.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "state": "OBSERVED",
                        "mask": c_mask,
                        "central_pa": orig_birth_cpa,
                        "tracked_cpa": orig_birth_cpa,
                        "front_radius": obs_front,
                        "radial_min": min_r,
                        "num_patches": n_pts,
                        "area_pixels": n_pts * 16,
                    })

                    # HTM initialization
                    raw_hist = compute_radial_histogram(patch_rad[c_mask], radial_bins)
                    htm_rec = HTMRecord(
                        pair_index=p_curr,
                        timestamp=ts_curr,
                        raw_radial_activity=raw_hist,
                        accepted_connected_patches=raw_hist,
                        assigned_track_components=raw_hist,
                        observed_radial_front_px=obs_front,
                        observed_radial_min_px=min_r,
                        temporal_continuity_score=1.0,
                    )
                    new_track.htm_history.append(htm_rec)

                    tracks[tid] = new_track
                    track_ownership_masks[step_idx][c_mask] = tid
                    patch_component_labels[step_idx][c_mask] = tid

                    track_state_rows.append({
                        "experiment": "Adaptive_PA_ROI_Tracking",
                        "track_id": tid,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "component_id": tid,
                        "component_uid": f"p{p_curr}_t{tid}",
                        "state": "OBSERVED",
                        "central_pa": orig_birth_cpa,
                        "tracked_cpa": orig_birth_cpa,
                        "roi_center_pa": init_roi[0],
                        "roi_half_width": init_roi[1],
                        "pa_min": float(arc_birth["arc_start"]),
                        "pa_max": float(arc_birth["arc_end"]),
                        "radial_front": obs_front,
                        "front_radius": obs_front,
                        "radial_min": min_r,
                        "area_pixels": n_pts * 16,
                        "num_patches": n_pts,
                        "birth_r_min_px": min_r,
                        "original_birth_pa": orig_birth_pa,
                        "original_birth_cpa": orig_birth_cpa,
                        "is_solar_root_anchor": True,
                        "outward_gate_evaluated": False,
                        "outward_gate_passed": True,
                        "radial_displacement": 0.0,
                    })

        # Convert state rows and conflicts to DataFrames
        df_track_states = pd.DataFrame(track_state_rows)
        df_conflicts = pd.DataFrame([vars(r) for r in all_conflict_records])
        df_rejections = pd.DataFrame(all_rejection_records)

        context.info(f"Adaptive Tracking complete. Total tracks created: {len(tracks)}, Total conflicts logged: {len(df_conflicts)}, Total outward rejections: {len(df_rejections)}.")

        # Save stage outputs
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_track_states.to_csv(stage_dir / "adaptive_track_states.csv", index=False)
        df_conflicts.to_csv(stage_dir / "patch_ownership_conflicts.csv", index=False)
        df_rejections.to_csv(stage_dir / "outward_gate_rejections.csv", index=False)

        # Context tables
        df_track_states.to_csv(context.tables_dir / "02_adaptive_track_states.csv", index=False)
        df_conflicts.to_csv(context.tables_dir / "03_patch_ownership_conflicts.csv", index=False)
        df_rejections.to_csv(context.tables_dir / "04_outward_gate_rejections.csv", index=False)

        # Save arrays
        np.save(context.arrays_dir / "adaptive_patch_component_labels.npy", patch_component_labels)
        np.save(context.arrays_dir / "adaptive_track_ownership_masks.npy", track_ownership_masks)

        anchor_tracks = sorted([t.track_id for t in tracks.values() if t.is_anchor])

        out = TrackingOutput(
            manifest_df=man_df,
            track_states_df=df_track_states,
            patch_component_labels=patch_component_labels,
            patch_activation_grid4=grid4_all,
            frame_cache=base_out.frame_cache,
            geometry=geom,
            total_tracks=len(tracks),
            anchor_tracks=anchor_tracks,
        )
        # Attach runtime metadata for Checkpoints
        out.adaptive_tracks = tracks
        out.conflict_records = all_conflict_records
        out.radial_bins = radial_bins
        out.ownership_masks = track_ownership_masks
        out.rejection_records = all_rejection_records
        return out
