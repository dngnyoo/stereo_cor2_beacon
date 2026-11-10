"""Component-First Adaptive PA-ROI Tracking Stage for CME Pipeline (A3).

Implements Component-First Matching:
- Preserves 8-connected component integrity on full grid4 before patch assignment.
- Evaluates candidate track-component pairs based on:
    1. Temporal Overlap (direct overlap and 8-connected adjacency).
    2. Angular Proximity (circular distance to tracked_cpa).
    3. Kinematic Continuity (advancing leading edge, penalizing large inward retreat).
    4. Outward Front Position (prioritizing leading edge fragments).
- Resolves shared/bridging components via CPA proximity without fragmenting single-track features.
- Enforces strict 1:1 continuation per track (no multi-component retention).
- Falls back to PREDICTED_ONLY if no credible candidate exists or inward retreat exceeds tolerance.
- Births new solar root anchor tracks only from unassigned components with r_min <= 30.0 px.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_dilation

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import DetectionOutput, GeometryContext, PipelineStage, TrackingOutput
from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output
from cme_pipeline.stages.tracking.registry import resolve_and_validate_tracking_parameters
from cme_pipeline.stages.tracking.adaptive_roi_tracking import (
    AdaptiveTrack,
    HTMRecord,
    compute_radial_histogram,
)
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
class ComponentDecisionRecord:
    """Record of a component matching evaluation."""
    pair_index: int
    timestamp: str
    track_id: int
    candidate_comp_id: int
    num_patches: int
    front_radius: float
    radial_min: float
    cpa: float
    direct_overlap_patches: int
    adjacent_overlap_patches: int
    overlap_score: float
    cpa_diff_deg: float
    angular_score: float
    radial_displacement: float
    kinematic_score: float
    front_bonus: float
    total_match_score: float
    is_selected: bool
    rejection_reason: str


class ComponentFirstAdaptiveROITrackingStage(PipelineStage):
    """Component-First Adaptive PA-ROI Tracking Stage (A3)."""
    name: str = "tracking"
    version: str = "3.1.0"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        params = resolve_and_validate_tracking_parameters("a3", config.tracking)
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": "component_first",
            "date_str": config.date_str,
            "parameters": params,
        }
        return compute_dict_hash(desc)

    def run(self, inputs: Any, config: PipelineConfig, context: PipelineContext) -> TrackingOutput:
        effective_params = resolve_and_validate_tracking_parameters("a3", config.tracking)
        context.info("Running ComponentFirstAdaptiveROITrackingStage (A3 Component-First Tracking)...")

        # 1. Shared Frozen Preprocessing & Hierarchical Detection Inputs
        if isinstance(inputs, DetectionOutput):
            det_out = inputs
        else:
            det_out = load_frozen_detection_inputs(config, context)

        geom = det_out.geometry
        patch_pa = geom.patch_pa
        patch_rad = geom.patch_rad
        man_df = det_out.manifest_df
        grid4_all = det_out.patch_activation_grid4
        raw_comp_labels = det_out.raw_component_labels

        n_pairs = len(man_df)
        radial_bins = np.linspace(15.0, 130.0, 47)  # 2.5 px bins from occulter to FOV

        tracks: Dict[int, AdaptiveTrack] = {}
        next_track_id = 1

        all_decision_records: List[ComponentDecisionRecord] = []
        all_rejection_records: List[Dict[str, Any]] = []
        track_state_rows: List[Dict[str, Any]] = []

        # Track ownership is strictly Track ID; patch_component_labels retains true detection components
        patch_component_labels = raw_comp_labels.copy()
        track_ownership_masks = np.zeros((n_pairs, 64, 64), dtype=np.int32)

        outward_enabled = bool(effective_params["outward_gate_enabled"])
        outward_tol = float(effective_params["outward_gate_tolerance_px"])
        ema_alpha = float(effective_params["ema_alpha"])
        max_cadence_gap_sec = float(effective_params["max_cadence_gap_sec"])
        birth_gate_px = float(effective_params["birth_gate_threshold_px"])
        default_roi_half = float(effective_params["default_roi_half_width_deg"])
        roi_padding = float(effective_params["roi_padding_deg"])
        max_missed_frames = int(effective_params["max_missed_frames"])

        prev_time = None

        for step_idx in range(n_pairs):
            p_curr = int(man_df.iloc[step_idx]["pair_index"])
            ts_curr = datetime.fromisoformat(man_df.iloc[step_idx]["end_time"])
            cur_grid = grid4_all[step_idx]

            # Cadence gap check
            if prev_time is not None:
                dt_sec = (ts_curr - prev_time).total_seconds()
                if dt_sec > max_cadence_gap_sec:
                    context.info(f"Large cadence gap detected before Pair {p_curr} ({dt_sec/60:.1f} min). Terminating active tracks.")
                    for t in tracks.values():
                        if not t.terminated:
                            t.terminated = True
            prev_time = ts_curr

            # -------------------------------------------------------------
            # Step 1: Detect 8-Connected Components on Raw Full grid4
            # -------------------------------------------------------------
            lbl_img, n_comps = label(cur_grid, structure=STRUCT_8)
            active_tracks = [t for t in tracks.values() if not t.terminated]

            comp_masks = {}
            comp_info = {}
            for c_id in range(1, n_comps + 1):
                m = (lbl_img == c_id)
                cpas = patch_pa[m]
                crads = patch_rad[m]
                arc = minimal_circular_enclosing_arc(cpas)
                comp_masks[c_id] = m
                comp_info[c_id] = {
                    "mask": m,
                    "n_patches": int(m.sum()),
                    "r_front": float(np.max(crads)),
                    "r_min": float(np.min(crads)),
                    "cpa": float(circular_mean_deg(cpas)),
                    "span": float(arc["arc_span"]),
                    "arc": arc,
                }

            # -------------------------------------------------------------
            # Step 2: Provisional Candidate Identification & Match Scoring
            # -------------------------------------------------------------
            track_cands: Dict[int, List[Dict[str, Any]]] = {t.track_id: [] for t in active_tracks}
            comp_claims: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {c_id: [] for c_id in range(1, n_comps + 1)}

            for t in active_tracks:
                roi_center, roi_half = t.adaptive_active_roi
                last_obs = next((h for h in reversed(t.history) if h["state"] == "OBSERVED"), None)
                prev_mask = last_obs["mask"] if last_obs is not None else np.zeros((64, 64), bool)
                prev_front = last_obs["front_radius"] if last_obs is not None else 125.0
                prev_rmin = last_obs["radial_min"] if last_obs is not None else 15.0
                prev_dil = binary_dilation(prev_mask, STRUCT_8) if last_obs is not None else prev_mask
                prev_n = last_obs["num_patches"] if last_obs is not None else 1

                for c_id, cdata in comp_info.items():
                    m_c = cdata["mask"]
                    # Direct and 8-connected adjacent overlap
                    do_cnt = int(np.logical_and(prev_mask, m_c).sum())
                    adj_cnt = int(np.logical_and(prev_dil, m_c).sum())

                    # Angular proximity to active ROI
                    cpa_diff_roi = abs(circular_diff_deg(cdata["cpa"], roi_center))
                    in_roi = (cpa_diff_roi <= (roi_half + 5.0))

                    # Radial expansion corridor
                    in_rad = (cdata["r_min"] <= (prev_front + 30.0)) and (cdata["r_front"] >= (prev_rmin - 8.0))

                    # Eligible if has temporal overlap OR (in_roi and in_rad)
                    if do_cnt > 0 or adj_cnt > 0 or (in_roi and in_rad):
                        # Overlap score normalized by previous track size
                        ov = (do_cnt + 0.5 * (adj_cnt - do_cnt)) / max(1, prev_n)

                        # Angular score relative to tracked_cpa
                        cpa_track_diff = abs(circular_diff_deg(cdata["cpa"], t.tracked_cpa))
                        ang = max(0.0, 1.0 - cpa_track_diff / max(15.0, roi_half))

                        # Kinematic score: favor advancing front, penalize large inward retreat
                        rad_disp = cdata["r_front"] - prev_front
                        if rad_disp >= 0:
                            kin = 1.0
                        else:
                            kin = max(0.0, 1.0 - abs(rad_disp) / 15.0)

                        # Front position bonus
                        front_bonus = cdata["r_front"] / 100.0

                        if do_cnt > 0 or adj_cnt > 0:
                            total_score = 4.0 * ov + 2.0 * ang + 6.0 * kin + 3.0 * front_bonus
                        else:
                            total_score = 1.0 * ang + 3.0 * kin + 2.0 * front_bonus

                        cand_rec = {
                            "comp_id": c_id,
                            "do_cnt": do_cnt,
                            "adj_cnt": adj_cnt,
                            "overlap_score": ov,
                            "cpa_diff": cpa_track_diff,
                            "ang_score": ang,
                            "rad_disp": rad_disp,
                            "kin_score": kin,
                            "front_bonus": front_bonus,
                            "total_score": total_score,
                            "info": cdata,
                        }
                        track_cands[t.track_id].append(cand_rec)
                        comp_claims[c_id].append((t.track_id, cand_rec))

            # -------------------------------------------------------------
            # Step 3: Shared / Contested Component Resolution
            # -------------------------------------------------------------
            for c_id, claim_list in comp_claims.items():
                if len(claim_list) <= 1:
                    continue

                overlapping_claims = [cl for cl in claim_list if cl[1]["do_cnt"] > 0]
                if len(overlapping_claims) > 1:
                    # Drop candidates from non-overlapping tracks
                    non_overlapping_tids = [cl[0] for cl in claim_list if cl[1]["do_cnt"] == 0]
                    for tid in non_overlapping_tids:
                        track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]

                    # Multiple tracks have direct overlap! Partition patches among overlapping tracks
                    c_mask = comp_info[c_id]["mask"]
                    c_rcs = list(zip(*np.nonzero(c_mask)))
                    part_masks = {tid: np.zeros((64, 64), bool) for tid, _ in overlapping_claims}
                    for r, c in c_rcs:
                        ppa = patch_pa[r, c]
                        best_tid = min(
                            overlapping_claims,
                            key=lambda cl: abs(circular_diff_deg(ppa, tracks[cl[0]].tracked_cpa))
                        )[0]
                        part_masks[best_tid][r, c] = True

                    # Update candidate record for each partitioned track with its sub-component
                    for tid, sub_m in part_masks.items():
                        sub_lbl, n_sub = label(sub_m, structure=STRUCT_8)
                        if n_sub > 0:
                            last_obs = next((h for h in reversed(tracks[tid].history) if h["state"] == "OBSERVED"), None)
                            pm = last_obs["mask"] if last_obs is not None else np.zeros((64, 64), bool)
                            best_sub = max(range(1, n_sub + 1), key=lambda s: np.logical_and(pm, sub_lbl == s).sum())
                            sub_best_m = (sub_lbl == best_sub)
                            sub_pas = patch_pa[sub_best_m]
                            sub_rads = patch_rad[sub_best_m]
                            sub_arc = minimal_circular_enclosing_arc(sub_pas)

                            for cand in track_cands[tid]:
                                if cand["comp_id"] == c_id:
                                    cand["info"] = {
                                        "mask": sub_best_m,
                                        "n_patches": int(sub_best_m.sum()),
                                        "r_front": float(np.max(sub_rads)),
                                        "r_min": float(np.min(sub_rads)),
                                        "cpa": float(circular_mean_deg(sub_pas)),
                                        "span": float(sub_arc["arc_span"]),
                                        "arc": sub_arc,
                                    }
                                    cand["rad_disp"] = float(np.max(sub_rads)) - last_obs["front_radius"]
                                    cand["do_cnt"] = int(np.logical_and(pm, sub_best_m).sum())
                                    cand["adj_cnt"] = int(np.logical_and(binary_dilation(pm, STRUCT_8), sub_best_m).sum())
                        else:
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]
                elif len(overlapping_claims) == 1:
                    # Exactly one track has direct overlap; drop component from non-overlapping tracks
                    winner_tid = overlapping_claims[0][0]
                    for tid, cand in claim_list:
                        if tid != winner_tid:
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]
                else:
                    # No tracks have direct overlap!
                    # Pick the track with highest match score
                    winner_tid = max(claim_list, key=lambda cl: cl[1]["total_score"])[0]
                    for tid, cand in claim_list:
                        if tid != winner_tid:
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]

            # -------------------------------------------------------------
            # Step 4: Strict 1:1 Continuation Enforcement per Track
            # -------------------------------------------------------------
            claimed_patches_this_frame: Dict[Tuple[int, int], int] = {}

            # Sort active tracks by candidate match quality priority
            track_priority = []
            for t in active_tracks:
                cands = track_cands[t.track_id]
                best_score = max([c["total_score"] for c in cands]) if cands else -1.0
                track_priority.append((best_score, t))
            track_priority.sort(key=lambda x: x[0], reverse=True)

            for _, t in track_priority:
                cands = track_cands[t.track_id]
                last_obs = next((h for h in reversed(t.history) if h["state"] == "OBSERVED"), None)
                last_cpa = last_obs["central_pa"] if last_obs is not None else t.tracked_cpa
                last_front = last_obs["front_radius"] if last_obs is not None else 125.0
                last_rmin = last_obs["radial_min"] if last_obs is not None else 20.0

                if not cands:
                    t.missed += 1
                    if t.missed >= max_missed_frames:
                        t.terminated = True
                    track_state_rows.append({
                        "experiment": "Component_First_Adaptive_Tracking",
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
                        "radial_min": last_rmin,
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

                # Sort candidates by total matching score descending
                cands.sort(key=lambda x: x["total_score"], reverse=True)

                valid_cand = None
                for cd in cands:
                    if outward_enabled and cd["rad_disp"] < -outward_tol:
                        continue
                    # Remove already claimed patches if any
                    c_mask = cd["info"]["mask"]
                    rcs = list(zip(*np.nonzero(c_mask)))
                    overlap_claimed = any(rc in claimed_patches_this_frame for rc in rcs)
                    if overlap_claimed:
                        new_m = c_mask.copy()
                        for rc in rcs:
                            if rc in claimed_patches_this_frame:
                                new_m[rc] = False
                        n_rem = int(new_m.sum())
                        if n_rem == 0:
                            continue
                        sub_pas = patch_pa[new_m]
                        sub_rads = patch_rad[new_m]
                        cd["info"]["mask"] = new_m
                        cd["info"]["n_patches"] = n_rem
                        cd["info"]["r_front"] = float(np.max(sub_rads))
                        cd["info"]["r_min"] = float(np.min(sub_rads))
                        cd["info"]["cpa"] = float(circular_mean_deg(sub_pas))
                        cd["rad_disp"] = float(np.max(sub_rads)) - last_front
                    valid_cand = cd
                    break

                # Log decision records for all evaluated candidates
                for c_idx, cd in enumerate(cands):
                    is_sel = (valid_cand is not None and cd is valid_cand)
                    rej_reason = "NONE"
                    if cd is not valid_cand:
                        if outward_enabled and cd["rad_disp"] < -outward_tol:
                            rej_reason = "REJECTED_OUTWARD_GATE"
                        else:
                            rej_reason = "LOWER_SCORE_OR_CLAIMED"

                    all_decision_records.append(
                        ComponentDecisionRecord(
                            pair_index=p_curr,
                            timestamp=ts_curr.isoformat(),
                            track_id=t.track_id,
                            candidate_comp_id=cd["comp_id"],
                            num_patches=cd["info"]["n_patches"],
                            front_radius=cd["info"]["r_front"],
                            radial_min=cd["info"]["r_min"],
                            cpa=cd["info"]["cpa"],
                            direct_overlap_patches=cd["do_cnt"],
                            adjacent_overlap_patches=cd["adj_cnt"],
                            overlap_score=float(cd["overlap_score"]),
                            cpa_diff_deg=float(cd["cpa_diff"]),
                            angular_score=float(cd["ang_score"]),
                            radial_displacement=float(cd["rad_disp"]),
                            kinematic_score=float(cd["kin_score"]),
                            front_bonus=float(cd["front_bonus"]),
                            total_match_score=float(cd["total_score"]),
                            is_selected=is_sel,
                            rejection_reason=rej_reason,
                        )
                    )

                if valid_cand is None:
                    # Candidate rejected by outward motion gate or claimed
                    all_rejection_records.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "track_id": t.track_id,
                        "prev_front_radius": last_front,
                        "cand_front_radius": cands[0]["info"]["r_front"],
                        "radial_displacement": cands[0]["rad_disp"],
                        "tolerance_px": outward_tol,
                        "cand_num_patches": cands[0]["info"]["n_patches"],
                        "cand_cpa": cands[0]["info"]["cpa"],
                        "action": "REJECTED_OR_CLAIMED",
                    })

                    t.missed += 1
                    if t.missed >= max_missed_frames:
                        t.terminated = True

                    track_state_rows.append({
                        "experiment": "Component_First_Adaptive_Tracking",
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
                        "radial_min": last_rmin,
                        "area_pixels": 0,
                        "num_patches": 0,
                        "birth_r_min_px": t.birth_r_min_px,
                        "original_birth_pa": t.original_birth_pa,
                        "original_birth_cpa": t.original_birth_cpa,
                        "is_solar_root_anchor": t.is_anchor,
                        "outward_gate_evaluated": True,
                        "outward_gate_passed": False,
                        "radial_displacement": cands[0]["rad_disp"],
                    })
                    continue

                # Valid candidate accepted!
                best_cand = valid_cand
                t.missed = 0
                cinfo = best_cand["info"]
                t_patch_mask = cinfo["mask"]
                t_patch_coords = list(zip(*np.nonzero(t_patch_mask)))
                for rc in t_patch_coords:
                    assert rc not in claimed_patches_this_frame, f"Double claim at {rc} for track {t.track_id}!"
                    claimed_patches_this_frame[rc] = t.track_id

                obs_cpa = cinfo["cpa"]
                obs_span = cinfo["span"]
                obs_front = cinfo["r_front"]
                obs_min_rad = cinfo["r_min"]

                # Update adaptive active ROI
                new_roi_half = max(default_roi_half, obs_span / 2.0 + roi_padding)
                t.adaptive_active_roi = (obs_cpa, new_roi_half)

                # Update tracked CPA (EMA alpha)
                cpa_diff = circular_diff_deg(obs_cpa, t.tracked_cpa)
                t.tracked_cpa = float((t.tracked_cpa + ema_alpha * cpa_diff) % 360.0)
                t.observed_cpa = obs_cpa

                track_ownership_masks[step_idx][t_patch_mask] = t.track_id

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

                # HTM calculation
                roi_center, roi_half = t.adaptive_active_roi
                in_roi_mask = cur_grid & (np.abs(circular_diff_deg(patch_pa, roi_center)) <= roi_half)
                raw_hist = compute_radial_histogram(patch_rad[in_roi_mask], radial_bins)
                conn_hist = compute_radial_histogram(patch_rad[t_patch_mask], radial_bins)
                assigned_hist = conn_hist

                last_two = [h for h in t.history if h["state"] == "OBSERVED"]
                if len(last_two) >= 2:
                    m_prev = last_two[-2]["mask"]
                    m_curr = last_two[-1]["mask"]
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
                    "experiment": "Component_First_Adaptive_Tracking",
                    "track_id": t.track_id,
                    "pair_index": p_curr,
                    "timestamp": ts_curr.isoformat(),
                    "component_id": int(best_cand["comp_id"]),
                    "component_uid": f"p{p_curr}_c{int(best_cand['comp_id'])}",
                    "state": "OBSERVED",
                    "central_pa": obs_cpa,
                    "tracked_cpa": t.tracked_cpa,
                    "roi_center_pa": t.adaptive_active_roi[0],
                    "roi_half_width": t.adaptive_active_roi[1],
                    "pa_min": float(cinfo["arc"]["arc_start"]),
                    "pa_max": float(cinfo["arc"]["arc_end"]),
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
                    "radial_displacement": best_cand["rad_disp"],
                })

            # -------------------------------------------------------------
            # Step 5: Birth Initialization for Unassigned Patches
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

                if min_r <= birth_gate_px:
                    tid = next_track_id
                    next_track_id += 1
                    raw_birth_cid = int(lbl_img[c_mask][0])

                    idx_min = int(np.argmin(c_rads))
                    orig_birth_pa = float(c_pas[idx_min])
                    orig_birth_cpa = float(circular_mean_deg(c_pas))

                    arc_birth = minimal_circular_enclosing_arc(c_pas)
                    init_extent = float(arc_birth["arc_span"])
                    init_roi_half = max(default_roi_half, init_extent / 2.0 + roi_padding)
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

                    track_state_rows.append({
                        "experiment": "Component_First_Adaptive_Tracking",
                        "track_id": tid,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "component_id": raw_birth_cid,
                        "component_uid": f"p{p_curr}_c{raw_birth_cid}",
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

        df_track_states = pd.DataFrame(track_state_rows)
        df_decisions = pd.DataFrame([vars(r) for r in all_decision_records])
        df_rejections = pd.DataFrame(all_rejection_records)

        context.info(
            f"Component-First Tracking complete. Total tracks: {len(tracks)}, "
            f"Decisions logged: {len(df_decisions)}, Outward rejections: {len(df_rejections)}."
        )

        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_track_states.to_csv(stage_dir / "component_first_track_states.csv", index=False)
        df_decisions.to_csv(stage_dir / "component_matching_decisions.csv", index=False)
        df_rejections.to_csv(stage_dir / "outward_gate_rejections.csv", index=False)

        anchor_tracks = sorted([t.track_id for t in tracks.values() if t.is_anchor])

        out = TrackingOutput(
            manifest_df=man_df,
            track_states_df=df_track_states,
            patch_component_labels=patch_component_labels,
            patch_activation_grid4=grid4_all,
            frame_cache=det_out.frame_cache,
            geometry=geom,
            total_tracks=len(tracks),
            anchor_tracks=anchor_tracks,
            track_ownership_masks=track_ownership_masks,
            tracking_method="a3",
            tracking_parameters=effective_params,
            dataset_metadata=det_out.dataset_metadata,
        )
        out.adaptive_tracks = tracks
        out.decision_records = all_decision_records
        out.decisions_df = df_decisions
        out.radial_bins = radial_bins
        out.rejection_records = all_rejection_records
        return adapt_tracking_output(out, max_birth_radius_px=birth_gate_px)

