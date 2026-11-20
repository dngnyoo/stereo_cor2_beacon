"""SPT-1: Independent Seed-Origin Motion Prediction Tracking Stage.

Extends SPT-0 by:
1. Learning outward radial velocity and CPA trajectory from track history
2. Forward predicting expected leading front position and CPA with gap-scaled uncertainty
3. Associating candidate components against predicted track geometry
4. Preventing spurious associations and fragmentation across gaps

Does NOT import, call, or rely on A5b tracking or association logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

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


class SPT1MotionPredictionTrackingStage(PipelineStage):
    """SPT-1 Independent Motion Prediction Tracking Stage."""
    name: str = "tracking"
    version: str = "1.0.0"

    def compute_fingerprint(
        self, inputs: DetectionOutput, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": "spt1",
            "parameters": config.tracking.parameters,
            "upstream_dataset": inputs.dataset_metadata.get("dataset_id", "unknown"),
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: DetectionOutput, config: PipelineConfig, context: PipelineContext
    ) -> TrackingOutput:
        logger.info("Running SPT1MotionPredictionTrackingStage (SPT-1 Tracking)...")
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

        manifest_df = inputs.manifest_df.copy()
        n_frames = len(manifest_df)
        geom = inputs.geometry
        grid4_all = inputs.patch_activation_grid4
        frame_cache = inputs.frame_cache

        # Precompute continuous D4 detection scores
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

            # 2. Forward Predict Active Track States to Current Timestamp
            for track in active_tracks:
                pred_r, pred_sigma, pred_cpa = predict_track_forward(track, t_curr)
                track.predicted_r_front = pred_r
                track.predicted_r_sigma = pred_sigma
                track.predicted_cpa = pred_cpa

            # 3. Predict-to-Component Association Matching
            matched_track_ids: Set[int] = set()
            claimed_component_ids: Set[int] = set()
            component_to_tracks: Dict[int, List[Tuple[SPTTrack, float]]] = {}

            for track in active_tracks:
                for comp in components:
                    # Circular angular error against predicted CPA
                    ang_diff = circular_diff_deg(comp.cpa_deg, track.predicted_cpa)
                    max_allowed_ang = max(22.0, track.current_width / 2.0 + 12.0)
                    if ang_diff > max_allowed_ang:
                        continue

                    # Radial consistency check with outward propagation
                    if outward_gate_enabled:
                        if comp.r_max_px < track.current_r_front - outward_tolerance_px:
                            continue

                    # Normalized radial prediction discrepancy
                    r_discrepancy = abs(comp.r_max_px - track.predicted_r_front) / track.predicted_r_sigma

                    # Spatial continuity: overlap with prior owned patches
                    overlap_count = sum(1 for pt in comp.patch_indices if pt in track.owned_patches)

                    # Unified kinematic match score (lower is better)
                    # Combines angular alignment, radial prediction error, and spatial continuity bonus
                    match_score = (
                        (ang_diff / 15.0) ** 2 +
                        (min(r_discrepancy, 3.0)) ** 2 -
                        (2.0 if overlap_count > 0 else 0.0)
                    )
                    component_to_tracks.setdefault(comp.component_id, []).append((track, match_score))

            # Assign components to winning predicted track
            comp_dict = {c.component_id: c for c in components}
            track_claims: Dict[int, List[ComponentGeometry]] = {t.track_id: [] for t in active_tracks}

            for comp_id, cand_list in component_to_tracks.items():
                cand_list.sort(key=lambda x: x[1])
                best_track, _ = cand_list[0]
                track_claims[best_track.track_id].append(comp_dict[comp_id])
                claimed_component_ids.add(comp_id)

            # 4. Update Matched Active Tracks
            for track in list(active_tracks):
                claimed_comps = track_claims.get(track.track_id, [])
                if claimed_comps:
                    all_pts: List[Tuple[int, int]] = []
                    all_comp_ids: List[int] = []
                    for c in claimed_comps:
                        all_pts.extend(c.patch_indices)
                        all_comp_ids.append(c.component_id)

                    pts_y = np.array([p[0] for p in all_pts])
                    pts_x = np.array([p[1] for p in all_pts])
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
                    track.owned_patches = set(all_pts)
                    track.last_observed_timestamp = t_curr
                    track.last_observed_pair_index = p_num
                    track.consecutive_observed_pairs += 1
                    track.consecutive_missed_pairs = 0
                    track.total_observed_pairs += 1
                    track.total_patches_accumulated += len(all_pts)

                    if track.lifecycle_state == SPTLifecycleState.TENTATIVE:
                        if track.consecutive_observed_pairs >= min_confirmation_pairs:
                            track.lifecycle_state = SPTLifecycleState.CONFIRMED

                    track_ownership_masks[f_idx, pts_y, pts_x] = track.track_id
                    matched_track_ids.add(track.track_id)

                    track.history.append(SPTTrackHistoryItem(
                        pair_index=p_num, timestamp=t_curr, state="OBSERVED",
                        patch_indices=all_pts, patch_count=len(all_pts),
                        cpa_deg=new_cpa, angular_width_deg=new_width,
                        pa_min_deg=new_pa_min, pa_max_deg=new_pa_max,
                        r_min_px=new_r_min, r_max_px=new_r_front,
                        r_mean_px=new_r_mean,
                        observed_radial_velocity_px_s=track.radial_velocity_px_s,
                        predicted_r_front_px=track.predicted_r_front,
                        predicted_cpa_deg=track.predicted_cpa,
                        associated_component_ids=all_comp_ids,
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

            # 5. Track Birth from Unclaimed Seeded Components
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
                    "experiment": "SPT1_Motion_Prediction",
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

        confirmed_tracks = [t for t in all_tracks if t.lifecycle_state == SPTLifecycleState.CONFIRMED or t.total_observed_pairs >= min_confirmation_pairs]
        logger.info(
            f"SPT-1 Tracking complete. Total tracks: {len(all_tracks)}, "
            f"Confirmed: {len(confirmed_tracks)}, Seed births logged: {len(df_seed_birth)}."
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
            tracking_method="spt1",
            tracking_parameters=params,
            dataset_metadata=inputs.dataset_metadata,
        )
        setattr(out, "seed_birth_df", df_seed_birth)
        setattr(out, "all_spt_tracks", all_tracks)
        setattr(out, "confirmed_spt_tracks", confirmed_tracks)
        return out
