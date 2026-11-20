"""SPT-2: Independent Seed-Origin Patch Ownership & Competing-Track Separation Stage.

The core extension beyond conventional connected-component tracking:
1. Detects when multiple independent seed-originated CMEs become connected by active patch bridges.
2. Performs marker-controlled geodesic and kinematic affinity region competition.
3. Assigns different portions of a single 8-connected T24 component to distinct persistent tracks.
4. Preserves spatial coherence and isolates ambiguous bridge patches.
5. Does NOT rely on A5b width normalization, A5b scoring, or nearest raw seed PA heuristics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import DetectionOutput, PipelineStage, TrackingOutput
from cme_pipeline.stages.tracking.spt_core import (
    ComponentGeometry,
    SPTLifecycleState,
    SPTTrack,
    SPTTrackHistoryItem,
    SeedProperties,
    detect_seeds_and_grow_components,
    predict_track_forward,
    update_track_kinematics,
)
from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg
from cme_pipeline.utils.io import compute_dict_hash

logger = logging.getLogger(__name__)


def arbitrate_contested_component(
    component: ComponentGeometry,
    candidate_tracks: List[SPTTrack],
    geom: GeometryContext,
    scores_grid: np.ndarray,
    *,
    bridge_margin_threshold: float = 0.05,
    mode: str = "marker_geodesic",
) -> Tuple[Dict[int, List[Tuple[int, int]]], List[Tuple[int, int]], List[Dict[str, Any]]]:
    """Separates competing tracks within a single connected active component.

    Returns:
        (track_to_assigned_patches, unresolved_bridge_patches, arbitration_decision_logs)
    """
    patch_indices = component.patch_indices
    n_patches = len(patch_indices)
    track_to_patches: Dict[int, List[Tuple[int, int]]] = {t.track_id: [] for t in candidate_tracks}
    unresolved_patches: List[Tuple[int, int]] = []
    logs: List[Dict[str, Any]] = []

    if len(candidate_tracks) <= 1:
        if candidate_tracks:
            track_to_patches[candidate_tracks[0].track_id] = list(patch_indices)
        return track_to_patches, unresolved_patches, logs

    patch_rad = np.asarray(geom.patch_rad, dtype=np.float32)
    patch_pa = np.asarray(geom.patch_pa, dtype=np.float32)

    # Build local 64x64 mask for this component
    comp_mask = np.zeros((64, 64), dtype=bool)
    for py, px in patch_indices:
        comp_mask[py, px] = True

    # Identify marker seeds for each candidate track within or adjacent to component
    # A marker is defined by prior owned patches that intersect or are adjacent to component,
    # or by the closest patches to the track's predicted front and CPA.
    track_markers: Dict[int, List[Tuple[int, int]]] = {}
    track_dists: Dict[int, np.ndarray] = {}

    for t in candidate_tracks:
        # Check direct spatial overlap first
        overlap_pts = [pt for pt in patch_indices if pt in t.owned_patches]
        if overlap_pts:
            track_markers[t.track_id] = overlap_pts
        else:
            # Fallback: find patches in component closest to track's predicted (CPA, r_front)
            best_pt = min(
                patch_indices,
                key=lambda pt: (
                    circular_diff_deg(patch_pa[pt[0], pt[1]], t.predicted_cpa) ** 2 +
                    ((patch_rad[pt[0], pt[1]] - t.predicted_r_front) / 4.0) ** 2
                )
            )
            track_markers[t.track_id] = [best_pt]

        # Compute geodesic / Euclidean distance from this track's markers within component
        marker_grid = np.zeros((64, 64), dtype=bool)
        for my, mx in track_markers[t.track_id]:
            marker_grid[my, mx] = True

        # Invert marker grid for distance transform
        edt = distance_transform_edt(~marker_grid)
        track_dists[t.track_id] = edt

    # Evaluate each patch in the component
    for py, px in patch_indices:
        p_pa = float(patch_pa[py, px])
        p_r = float(patch_rad[py, px])
        p_score = float(scores_grid[py, px])

        candidate_scores: List[Tuple[SPTTrack, float, float, float]] = []

        for t in candidate_tracks:
            geo_dist = float(track_dists[t.track_id][py, px])
            ang_diff = float(circular_diff_deg(p_pa, t.predicted_cpa))
            r_diff = abs(p_r - t.predicted_r_front)

            # Kinematic affinity: higher is better
            # Combines spatial marker proximity, predicted CPA alignment, and seed continuity
            seed_ang_diff = float(circular_diff_deg(p_pa, t.seed.seed_pa_deg))
            
            # Affinity formulation:
            # S = exp(- (geo_dist / 6.0)^2 - (ang_diff / 15.0)^2)
            spatial_term = np.exp(- (geo_dist / 8.0) ** 2)
            angular_term = np.exp(- (ang_diff / 18.0) ** 2)
            seed_bonus = 0.2 if seed_ang_diff <= 15.0 else 0.0

            total_affinity = spatial_term * angular_term + seed_bonus
            candidate_scores.append((t, total_affinity, geo_dist, ang_diff))

        # Sort candidates by total affinity descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_t, best_aff, best_geo, best_ang = candidate_scores[0]
        second_t, second_aff, second_geo, second_ang = candidate_scores[1]

        margin = best_aff - second_aff

        # Check if patch sits on an ambiguous artificial bridge
        if margin < bridge_margin_threshold and best_geo > 2.0 and second_geo > 2.0:
            unresolved_patches.append((py, px))
            winner_id = 0
            reason = "AMBIGUOUS_BRIDGE_PATCH"
        else:
            track_to_patches[best_t.track_id].append((py, px))
            winner_id = best_t.track_id
            reason = "COMPETITIVE_AFFINITY_WINNER"

        logs.append({
            "component_id": component.component_id,
            "patch_x": px,
            "patch_y": py,
            "patch_pa_deg": p_pa,
            "patch_r_px": p_r,
            "winner_track_id": winner_id,
            "best_track_id": best_t.track_id,
            "second_track_id": second_t.track_id,
            "best_affinity": best_aff,
            "second_affinity": second_aff,
            "affinity_margin": margin,
            "reason": reason,
        })

    return track_to_patches, unresolved_patches, logs


class SPT2PatchOwnershipTrackingStage(PipelineStage):
    """SPT-2 Independent Patch Ownership & Competing-Track Separation Stage."""
    name: str = "tracking"
    version: str = "1.0.0"

    def compute_fingerprint(
        self, inputs: DetectionOutput, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": "spt2",
            "parameters": config.tracking.parameters,
            "upstream_dataset": inputs.dataset_metadata.get("dataset_id", "unknown"),
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: DetectionOutput, config: PipelineConfig, context: PipelineContext
    ) -> TrackingOutput:
        logger.info("Running SPT2PatchOwnershipTrackingStage (SPT-2 Tracking)...")
        params = config.tracking.parameters or {}

        patch_size = int(params.get("patch_size", 4))
        max_cadence_gap_sec = float(params.get("max_cadence_gap_sec", 4500.0))
        r_min_px = float(params.get("seed_min_radius_px", 22.0))
        r_seed_max_px = float(params.get("seed_max_radius_px", 30.0))
        seed_score_cutoff = float(params.get("seed_score_cutoff", 0.0))
        outward_gate_enabled = bool(params.get("outward_gate_enabled", True))
        outward_tolerance_px = float(params.get("outward_gate_tolerance_px", 4.0))
        max_missed_frames = int(params.get("max_missed_frames", 3))
        min_confirmation_pairs = int(params.get("min_confirmation_pairs", 2))
        min_component_patches = int(params.get("min_component_patches", 1))
        bridge_threshold = float(params.get("bridge_margin_threshold", 0.05))
        separation_mode = str(params.get("separation_mode", "marker_geodesic"))

        manifest_df = inputs.manifest_df.copy()
        n_frames = len(manifest_df)
        geom = inputs.geometry
        grid4_all = inputs.patch_activation_grid4
        frame_cache = inputs.frame_cache

        scores_all = np.zeros((n_frames, 64, 64), dtype=np.float32)
        for f_idx in range(n_frames):
            fc = frame_cache[f_idx]
            z = fc.get("z_global")
            val = fc.get("val")
            if z is not None and val is not None:
                z_block = z.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3).mean(axis=(2, 3))
                val_block = val.reshape(64, 4, 64, 4).transpose(0, 2, 1, 3).mean(axis=(2, 3))
                usable_patches = val_block > 0.5
                scores_all[f_idx] = np.where(usable_patches, np.maximum(z_block, 0.0), 0.0)
            else:
                scores_all[f_idx] = grid4_all[f_idx].astype(np.float32)

        track_ownership_masks = np.zeros((n_frames, 64, 64), dtype=np.int32)
        active_tracks: List[SPTTrack] = []
        all_tracks: List[SPTTrack] = []
        next_track_id = 1

        history_rows: List[Dict[str, Any]] = []
        seed_birth_rows: List[Dict[str, Any]] = []
        all_arbitration_logs: List[Dict[str, Any]] = []

        for f_idx in range(n_frames):
            p_row = manifest_df.iloc[f_idx]
            p_num = int(p_row["pair_index"])
            timestamp_raw = p_row.get("end_time") or p_row.get("timestamp")
            if isinstance(timestamp_raw, str):
                t_curr = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
            else:
                t_curr = timestamp_raw

            is_large_gap = bool(p_row.get("is_large_gap", False))
            is_usable = bool(p_row.get("usable_detection", True))
            if "observation_status" in p_row:
                obs_status = str(p_row["observation_status"])
                if obs_status != "VALID":
                    is_usable = False

            if is_large_gap:
                logger.info(f"Cadence gap before Pair {p_num}. Terminating active tracks.")
                for t in active_tracks:
                    t.lifecycle_state = SPTLifecycleState.TERMINATED
                    t.termination_reason = "CADENCE_GAP"
                active_tracks.clear()

            if not is_usable:
                for t in active_tracks:
                    t.history.append(SPTTrackHistoryItem(
                        pair_index=p_num, timestamp=t_curr, state="UNAVAILABLE",
                        patch_indices=[], patch_count=0, cpa_deg=t.current_cpa,
                        angular_width_deg=t.current_width, pa_min_deg=t.pa_min,
                        pa_max_deg=t.pa_max, r_min_px=t.current_r_min,
                        r_max_px=t.current_r_front, r_mean_px=t.current_r_front,
                        observed_radial_velocity_px_s=0.0,
                        predicted_r_front_px=t.predicted_r_front,
                        predicted_cpa_deg=t.predicted_cpa,
                        associated_component_ids=[],
                    ))
                continue

            active_grid = grid4_all[f_idx]
            scores_grid = scores_all[f_idx]

            # 1. Detect seeds and grow full 8-connected T24 components
            components, seeds = detect_seeds_and_grow_components(
                active_grid, scores_grid, geom, p_num, t_curr,
                r_min_px=r_min_px, r_seed_max_px=r_seed_max_px,
                seed_score_cutoff=seed_score_cutoff,
                min_component_patches=min_component_patches,
            )

            # 2. Forward Predict Active Track States
            for track in active_tracks:
                pred_r, pred_sigma, pred_cpa = predict_track_forward(track, t_curr)
                track.predicted_r_front = pred_r
                track.predicted_r_sigma = pred_sigma
                track.predicted_cpa = pred_cpa

            # 3. Identify Candidate Tracks for each Component
            component_candidates: Dict[int, List[SPTTrack]] = {}
            for comp in components:
                cands: List[SPTTrack] = []
                for track in active_tracks:
                    ang_diff = circular_diff_deg(comp.cpa_deg, track.predicted_cpa)
                    max_allowed_ang = max(25.0, track.current_width / 2.0 + 15.0)
                    if ang_diff > max_allowed_ang:
                        continue

                    if outward_gate_enabled:
                        if comp.r_max_px < track.current_r_front - outward_tolerance_px:
                            continue

                    cands.append(track)
                component_candidates[comp.component_id] = cands

            # 4. Independent Patch Ownership & Contested Component Separation
            track_allocated_patches: Dict[int, List[Tuple[int, int]]] = {t.track_id: [] for t in active_tracks}
            track_associated_comps: Dict[int, List[int]] = {t.track_id: [] for t in active_tracks}
            claimed_component_ids: Set[int] = set()

            for comp in components:
                cands = component_candidates.get(comp.component_id, [])
                if not cands:
                    continue

                claimed_component_ids.add(comp.component_id)

                if len(cands) == 1:
                    # Uncontested component: assign entirely to candidate track
                    single_t = cands[0]
                    track_allocated_patches[single_t.track_id].extend(comp.patch_indices)
                    track_associated_comps[single_t.track_id].append(comp.component_id)
                else:
                    # CONTESTED COMPONENT: Multiple independent CMEs connected!
                    # Execute SPT-2 region competition and bridge separation
                    logger.info(
                        f"Pair {p_num}: Component {comp.component_id} ({comp.patch_count} patches) "
                        f"contested by {[t.track_id for t in cands]}. Executing SPT-2 separation."
                    )
                    t_patches, unres_pts, arb_logs = arbitrate_contested_component(
                        comp, cands, geom, scores_grid,
                        bridge_margin_threshold=bridge_threshold,
                        mode=separation_mode,
                    )
                    for log_item in arb_logs:
                        log_item["pair_index"] = p_num
                    all_arbitration_logs.extend(arb_logs)

                    for tid, pts in t_patches.items():
                        if pts:
                            track_allocated_patches[tid].extend(pts)
                            track_associated_comps[tid].append(comp.component_id)

            # 5. Update Active Tracks with Allocated Patches
            matched_track_ids: Set[int] = set()
            for track in list(active_tracks):
                owned_pts = track_allocated_patches.get(track.track_id, [])
                if owned_pts:
                    pts_y = np.array([p[0] for p in owned_pts])
                    pts_x = np.array([p[1] for p in owned_pts])
                    comp_r_vals = np.asarray(geom.patch_rad, dtype=np.float32)[pts_y, pts_x]
                    comp_pa_vals = np.asarray(geom.patch_pa, dtype=np.float32)[pts_y, pts_x]

                    new_r_front = float(np.max(comp_r_vals))
                    new_r_min = float(np.min(comp_r_vals))
                    new_r_mean = float(np.mean(comp_r_vals))
                    new_cpa = float(circular_mean_deg(comp_pa_vals))

                    pa_s = np.sort(np.mod(comp_pa_vals, 360.0))
                    gaps = np.append(np.diff(pa_s), (pa_s[0] + 360.0) - pa_s[-1])
                    mg_idx = int(np.argmax(gaps))
                    new_width = 360.0 - float(gaps[mg_idx]) if gaps[mg_idx] < 360.0 else 0.0
                    new_pa_min = float(pa_s[(mg_idx + 1) % len(pa_s)])
                    new_pa_max = float((new_pa_min + new_width) % 360.0)

                    update_track_kinematics(track, t_curr, new_r_front, new_cpa)

                    track.current_r_front = new_r_front
                    track.current_r_min = new_r_min
                    track.current_width = new_width
                    track.pa_min = new_pa_min
                    track.pa_max = new_pa_max
                    track.owned_patches = set(owned_pts)
                    track.last_observed_timestamp = t_curr
                    track.last_observed_pair_index = p_num
                    track.consecutive_observed_pairs += 1
                    track.consecutive_missed_pairs = 0
                    track.total_observed_pairs += 1
                    track.total_patches_accumulated += len(owned_pts)

                    if track.lifecycle_state == SPTLifecycleState.TENTATIVE:
                        if track.consecutive_observed_pairs >= min_confirmation_pairs:
                            track.lifecycle_state = SPTLifecycleState.CONFIRMED

                    track_ownership_masks[f_idx, pts_y, pts_x] = track.track_id
                    matched_track_ids.add(track.track_id)

                    track.history.append(SPTTrackHistoryItem(
                        pair_index=p_num, timestamp=t_curr, state="OBSERVED",
                        patch_indices=owned_pts, patch_count=len(owned_pts),
                        cpa_deg=new_cpa, angular_width_deg=new_width,
                        pa_min_deg=new_pa_min, pa_max_deg=new_pa_max,
                        r_min_px=new_r_min, r_max_px=new_r_front,
                        r_mean_px=new_r_mean,
                        observed_radial_velocity_px_s=track.radial_velocity_px_s,
                        predicted_r_front_px=track.predicted_r_front,
                        predicted_cpa_deg=track.predicted_cpa,
                        associated_component_ids=track_associated_comps[track.track_id],
                    ))
                else:
                    track.consecutive_missed_pairs += 1
                    track.consecutive_observed_pairs = 0
                    track.owned_patches.clear()

                    if track.consecutive_missed_pairs > max_missed_frames:
                        track.lifecycle_state = SPTLifecycleState.TERMINATED
                        track.termination_reason = "MAX_MISSED_EXCEEDED"
                        active_tracks.remove(track)
                    else:
                        track.lifecycle_state = SPTLifecycleState.TEMPORARILY_MISSING

                    track.history.append(SPTTrackHistoryItem(
                        pair_index=p_num, timestamp=t_curr, state="MISSED",
                        patch_indices=[], patch_count=0, cpa_deg=track.current_cpa,
                        angular_width_deg=track.current_width, pa_min_deg=track.pa_min,
                        pa_max_deg=track.pa_max, r_min_px=track.current_r_min,
                        r_max_px=track.current_r_front, r_mean_px=track.current_r_front,
                        observed_radial_velocity_px_s=0.0,
                        predicted_r_front_px=track.predicted_r_front,
                        predicted_cpa_deg=track.predicted_cpa,
                        associated_component_ids=[],
                    ))

            # 6. Track Birth from Unclaimed Seeded Components
            unclaimed_seeded = [
                c for c in components
                if c.component_id not in claimed_component_ids and c.has_seed and c.representative_seed is not None
            ]
            unclaimed_seeded.sort(key=lambda c: c.representative_seed.seed_score, reverse=True)

            for comp in unclaimed_seeded:
                seed = comp.representative_seed
                tid = next_track_id
                next_track_id += 1

                pts_y = np.array([p[0] for p in comp.patch_indices])
                pts_x = np.array([p[1] for p in comp.patch_indices])

                new_track = SPTTrack(
                    track_id=tid,
                    lifecycle_state=SPTLifecycleState.TENTATIVE,
                    birth_pair_index=p_num,
                    birth_timestamp=t_curr,
                    seed=seed,
                    current_cpa=comp.cpa_deg,
                    tracked_cpa=comp.cpa_deg,
                    predicted_cpa=comp.cpa_deg,
                    current_width=comp.angular_width_deg,
                    pa_min=comp.pa_min_deg,
                    pa_max=comp.pa_max_deg,
                    current_r_min=comp.r_min_px,
                    current_r_front=comp.r_max_px,
                    predicted_r_front=comp.r_max_px,
                    predicted_r_sigma=4.0,
                    radial_velocity_px_s=0.0035,
                    consecutive_observed_pairs=1,
                    consecutive_missed_pairs=0,
                    total_observed_pairs=1,
                    total_patches_accumulated=comp.patch_count,
                    last_observed_timestamp=t_curr,
                    last_observed_pair_index=p_num,
                    history=[SPTTrackHistoryItem(
                        pair_index=p_num, timestamp=t_curr, state="OBSERVED",
                        patch_indices=comp.patch_indices, patch_count=comp.patch_count,
                        cpa_deg=comp.cpa_deg, angular_width_deg=comp.angular_width_deg,
                        pa_min_deg=comp.pa_min_deg, pa_max_deg=comp.pa_max_deg,
                        r_min_px=comp.r_min_px, r_max_px=comp.r_max_px,
                        r_mean_px=comp.r_mean_px,
                        observed_radial_velocity_px_s=0.0,
                        predicted_r_front_px=comp.r_max_px,
                        predicted_cpa_deg=comp.cpa_deg,
                        associated_component_ids=[comp.component_id],
                    )],
                    owned_patches=set(comp.patch_indices),
                )
                active_tracks.append(new_track)
                all_tracks.append(new_track)
                track_ownership_masks[f_idx, pts_y, pts_x] = tid

                seed_birth_rows.append({
                    "pair_index": p_num,
                    "track_id": tid,
                    "seed_id": seed.seed_id,
                    "seed_score": seed.seed_score,
                    "seed_pa_deg": seed.seed_pa_deg,
                    "seed_radius_px": seed.seed_radius_px,
                    "component_id": comp.component_id,
                    "patch_count": comp.patch_count,
                })

        for t in all_tracks:
            for h in t.history:
                history_rows.append({
                    "experiment": "SPT2_Patch_Ownership",
                    "track_id": t.track_id,
                    "pair_index": h.pair_index,
                    "timestamp": h.timestamp.isoformat(),
                    "state": h.state,
                    "lifecycle_state": t.lifecycle_state.value,
                    "central_pa": h.cpa_deg,
                    "tracked_cpa": t.tracked_cpa,
                    "predicted_cpa": h.predicted_cpa_deg,
                    "angular_width": h.angular_width_deg,
                    "front_radius": h.r_max_px,
                    "predicted_front_radius": h.predicted_r_front_px,
                    "radial_velocity_px_s": h.observed_radial_velocity_px_s,
                    "num_patches": h.patch_count,
                    "pa_min": h.pa_min_deg,
                    "pa_max": h.pa_max_deg,
                    "birth_r_min_px": t.seed.seed_radius_px,
                    "seed_score": t.seed.seed_score,
                    "seed_pa": t.seed.seed_pa_deg,
                    "is_confirmed": t.lifecycle_state == SPTLifecycleState.CONFIRMED or t.total_observed_pairs >= min_confirmation_pairs,
                })

        df_track_states = pd.DataFrame(history_rows) if history_rows else pd.DataFrame()
        df_seed_birth = pd.DataFrame(seed_birth_rows) if seed_birth_rows else pd.DataFrame()
        df_arbitrations = pd.DataFrame(all_arbitration_logs) if all_arbitration_logs else pd.DataFrame()

        confirmed_tracks = [t for t in all_tracks if t.lifecycle_state == SPTLifecycleState.CONFIRMED or t.total_observed_pairs >= min_confirmation_pairs]
        logger.info(
            f"SPT-2 Tracking complete. Total tracks: {len(all_tracks)}, "
            f"Confirmed: {len(confirmed_tracks)}, Contested patch arbitrations: {len(df_arbitrations)}."
        )

        out = TrackingOutput(
            manifest_df=manifest_df,
            track_states_df=df_track_states,
            patch_component_labels=inputs.raw_component_labels,
            patch_activation_grid4=grid4_all,
            frame_cache=inputs.frame_cache,
            geometry=geom,
            total_tracks=len(all_tracks),
            track_ownership_masks=track_ownership_masks,
            tracking_method="spt2",
            tracking_parameters=params,
            dataset_metadata=inputs.dataset_metadata,
        )
        setattr(out, "seed_birth_df", df_seed_birth)
        setattr(out, "seed_arbitration_df", df_arbitrations)
        setattr(out, "all_spt_tracks", all_tracks)
        setattr(out, "confirmed_spt_tracks", confirmed_tracks)
        return out
