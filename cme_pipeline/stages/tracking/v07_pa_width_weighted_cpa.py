"""A5b: PA-Width-Normalized CPA Distance Adaptive PA-ROI Tracking Stage.

Refines contested-component patch ownership by normalizing the circular CPA distance
by each track's previously observed PA width:
    d_i^*(p) = d_circ(PA_p, CPA_i) / max(Delta PA_i, Delta PA_min)

Eliminates global capacity quotas and forced remote patch assignment.
Preserves all A4 multi-component, leading-front, adaptive ROI, and causal continuity rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
from cme_pipeline.stages.tracking.v02_adaptive_pa_roi import (
    AdaptiveTrack,
    HTMRecord,
    compute_radial_histogram,
)
from cme_pipeline.stages.tracking.v05_multi_component import (
    A4ComponentRecord,
    A4TrackHistory,
    MultiComponentDecisionRecord,
    asdict_rec,
    STRUCT_8,
    MAX_CADENCE_GAP_SEC,
    BIRTH_GATE_THRESHOLD_PX,
    ALPHA_CPA_EMA,
    DEFAULT_ROI_HALF_WIDTH_DEG,
    ROI_PADDING_DEG,
    MAX_MISSED_FRAMES,
)
from cme_pipeline.utils.geometry import (
    circular_diff_deg,
    circular_mean_deg,
    minimal_circular_enclosing_arc,
)
from cme_pipeline.utils.io import compute_dict_hash


class PAWidthWeightedCPAAdaptiveROITrackingStage(PipelineStage):
    """A5b PA-Width-Normalized CPA Distance Adaptive PA-ROI Tracking Stage."""
    name: str = "tracking"
    version: str = "5.1.0"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        params = resolve_and_validate_tracking_parameters("a5b", config.tracking)
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": "pa_width_weighted_cpa",
            "date_str": config.date_str,
            "parameters": params,
        }
        return compute_dict_hash(desc)

    def run(self, inputs: Any, config: PipelineConfig, context: PipelineContext) -> TrackingOutput:
        effective_params = resolve_and_validate_tracking_parameters("a5b", config.tracking)
        context.info("Running PAWidthWeightedCPAAdaptiveROITrackingStage (A5b Tracking)...")

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

        all_decision_records: List[MultiComponentDecisionRecord] = []
        track_state_rows: List[Dict[str, Any]] = []
        comp_state_rows: List[Dict[str, Any]] = []
        leading_front_rows: List[Dict[str, Any]] = []
        contested_allocation_rows: List[Dict[str, Any]] = []

        # Track ownership strictly Track ID; patch_component_labels preserves raw spatial components
        patch_component_labels = raw_comp_labels.copy()
        track_ownership_masks = np.zeros((n_pairs, 64, 64), dtype=np.int32)

        outward_enabled = bool(effective_params["outward_gate_enabled"])
        outward_tol = float(effective_params["outward_gate_tolerance_px"])
        min_pa_width_deg = float(effective_params["min_pa_width_deg"])
        ema_alpha = float(effective_params["ema_alpha"])
        max_cadence_gap_sec = float(effective_params["max_cadence_gap_sec"])
        birth_gate_px = float(effective_params["birth_gate_threshold_px"])
        default_roi_half = float(effective_params["default_roi_half_width_deg"])
        roi_padding = float(effective_params["roi_padding_deg"])
        max_missed_frames = int(effective_params["max_missed_frames"])

        prev_time = None

        for step_idx in range(n_pairs):
            # Unavailable C3/C0 observation is not a negative CME detection:
            # do not update tracks, increment missed, or advance last valid time.
            if ("usable_detection" in man_df.columns
                    and not bool(man_df.iloc[step_idx]["usable_detection"])):
                context.info(
                    f"Pair {int(man_df.iloc[step_idx]['pair_index'])}: "
                    f"{man_df.iloc[step_idx]['observation_status']} — "
                    "tracking observation unavailable; no missed-frame update."
                )
                continue
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

            comp_info: Dict[int, Dict[str, Any]] = {}
            for c_id in range(1, n_comps + 1):
                m = (lbl_img == c_id)
                cpas = patch_pa[m]
                crads = patch_rad[m]
                arc = minimal_circular_enclosing_arc(cpas)
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
                last_obs: Optional[A4TrackHistory] = next(
                    (h for h in reversed(t.history) if getattr(h, "state", "") == "OBSERVED"), None
                )
                prev_mask = last_obs.union_mask if last_obs is not None else np.zeros((64, 64), bool)
                prev_front = last_obs.leading_front_radius if last_obs is not None else 125.0
                prev_rmin = min([c.r_min for c in last_obs.components]) if (last_obs is not None and last_obs.components) else 15.0
                prev_dil = binary_dilation(prev_mask, STRUCT_8) if last_obs is not None else prev_mask
                prev_n = last_obs.total_patches if last_obs is not None else 1

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
                        ov = (do_cnt + 0.5 * (adj_cnt - do_cnt)) / max(1, prev_n)
                        cpa_track_diff = abs(circular_diff_deg(cdata["cpa"], t.tracked_cpa))
                        ang = max(0.0, 1.0 - cpa_track_diff / max(15.0, roi_half))

                        rad_disp = cdata["r_front"] - prev_front
                        kin = 1.0 if rad_disp >= 0 else max(0.0, 1.0 - abs(rad_disp) / 15.0)
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
            # Step 3: Shared / Contested Component Resolution (A5b PA-Width-Normalized)
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

                    c_mask = comp_info[c_id]["mask"]
                    c_rcs = list(zip(*np.nonzero(c_mask)))
                    comp_tids = [cl[0] for cl in overlapping_claims]

                    # 1. Compute previous observed PA width for each participating track
                    track_widths: Dict[int, float] = {}
                    track_cpas: Dict[int, float] = {}
                    for tid in comp_tids:
                        t = tracks[tid]
                        track_cpas[tid] = t.tracked_cpa
                        last_obs = next(
                            (h for h in reversed(t.history) if getattr(h, "state", "") == "OBSERVED"), None
                        )
                        if last_obs is not None and last_obs.union_mask.sum() > 0:
                            prev_pas = patch_pa[last_obs.union_mask]
                            prev_arc = minimal_circular_enclosing_arc(prev_pas)
                            w = max(float(prev_arc["arc_span"]), min_pa_width_deg)
                        else:
                            w = min_pa_width_deg
                        track_widths[tid] = w

                    # 2. Partition patches among overlapping tracks based on PA-width-normalized circular CPA distance
                    # d_i^*(p) = d_circ(PA_p, CPA_i) / max(Delta PA_i, Delta PA_min)
                    part_masks = {tid: np.zeros((64, 64), bool) for tid in comp_tids}
                    for r, c in c_rcs:
                        ppa = patch_pa[r, c]
                        best_tid = min(
                            comp_tids,
                            key=lambda tid: (
                                abs(circular_diff_deg(ppa, track_cpas[tid])) / track_widths[tid],
                                tid
                            )
                        )
                        part_masks[best_tid][r, c] = True

                    # 3. Log contested allocation event for auditing
                    for tid in comp_tids:
                        n_assigned = int(part_masks[tid].sum())
                        contested_allocation_rows.append({
                            "pair_index": p_curr,
                            "timestamp": ts_curr.isoformat(),
                            "comp_id": c_id,
                            "total_patches": len(c_rcs),
                            "track_id": tid,
                            "prev_pa_width": track_widths[tid],
                            "allocated_patches": n_assigned,
                            "tracked_cpa": track_cpas[tid],
                        })

                    # 4. Update candidate record for each partitioned track with its sub-component
                    for tid, sub_m in part_masks.items():
                        if sub_m.sum() == 0:
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]
                            continue

                        sub_lbl, n_sub = label(sub_m, structure=STRUCT_8)
                        if n_sub > 0:
                            last_obs = next(
                                (h for h in reversed(tracks[tid].history) if getattr(h, "state", "") == "OBSERVED"),
                                None
                            )
                            pm = last_obs.union_mask if last_obs is not None else np.zeros((64, 64), bool)
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
                                    cand["rad_disp"] = float(np.max(sub_rads)) - (last_obs.leading_front_radius if last_obs else 125.0)
                                    cand["do_cnt"] = int(np.logical_and(pm, sub_best_m).sum())
                                    cand["adj_cnt"] = int(np.logical_and(binary_dilation(pm, STRUCT_8), sub_best_m).sum())
                        else:
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]
                elif len(overlapping_claims) == 1:
                    winner_tid = overlapping_claims[0][0]
                    for tid, cand in claim_list:
                        if tid != winner_tid:
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]
                else:
                    # No tracks have direct overlap! Pick track with highest total score
                    winner_tid = max(claim_list, key=lambda cl: cl[1]["total_score"])[0]
                    for tid, cand in claim_list:
                        if tid != winner_tid:
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]

            # -------------------------------------------------------------
            # Step 4: Multi-Component Ownership Assignment
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
                last_obs: Optional[A4TrackHistory] = next(
                    (h for h in reversed(t.history) if getattr(h, "state", "") == "OBSERVED"), None
                )
                last_cpa = last_obs.central_pa if last_obs is not None else t.tracked_cpa
                last_front = last_obs.leading_front_radius if last_obs is not None else 125.0

                accepted_components: List[A4ComponentRecord] = []

                # Evaluate all candidate components for this track
                for cd in sorted(cands, key=lambda x: x["total_score"], reverse=True):
                    c_mask = cd["info"]["mask"]
                    rcs = list(zip(*np.nonzero(c_mask)))

                    # Check collision with already claimed patches
                    if any(rc in claimed_patches_this_frame for rc in rcs):
                        new_m = c_mask.copy()
                        for rc in rcs:
                            if rc in claimed_patches_this_frame:
                                new_m[rc] = False
                        n_rem = int(new_m.sum())
                        if n_rem == 0:
                            all_decision_records.append(
                                MultiComponentDecisionRecord(
                                    pair_index=p_curr, timestamp=ts_curr.isoformat(),
                                    track_id=t.track_id, candidate_comp_id=cd["comp_id"],
                                    num_patches=0, front_radius=cd["info"]["r_front"],
                                    radial_min=cd["info"]["r_min"], cpa=cd["info"]["cpa"],
                                    direct_overlap_patches=cd["do_cnt"], adjacent_overlap_patches=cd["adj_cnt"],
                                    overlap_score=float(cd["overlap_score"]), cpa_diff_deg=float(cd["cpa_diff"]),
                                    angular_score=float(cd["ang_score"]), radial_displacement=float(cd["rad_disp"]),
                                    kinematic_score=float(cd["kin_score"]), front_bonus=float(cd["front_bonus"]),
                                    total_match_score=float(cd["total_score"]), is_accepted=False,
                                    rejection_reason="ALL_PATCHES_CLAIMED",
                                )
                            )
                            continue

                        sub_pas = patch_pa[new_m]
                        sub_rads = patch_rad[new_m]
                        cd["info"]["mask"] = new_m
                        cd["info"]["n_patches"] = n_rem
                        cd["info"]["r_front"] = float(np.max(sub_rads))
                        cd["info"]["r_min"] = float(np.min(sub_rads))
                        cd["info"]["cpa"] = float(circular_mean_deg(sub_pas))
                        cd["rad_disp"] = float(np.max(sub_rads)) - last_front

                    # Credible continuity criteria:
                    # 1. Direct temporal overlap, OR
                    # 2. Eight-connected adjacency with angular proximity, OR
                    # 3. Within PA ROI and radial expansion corridor
                    has_overlap = (cd["do_cnt"] > 0 or cd["adj_cnt"] > 0)
                    in_corridor = (
                        (cd["cpa_diff"] <= (t.adaptive_active_roi[1] + 5.0)) and
                        (cd["info"]["r_front"] >= (last_front - 25.0) if last_obs else True)
                    )

                    if has_overlap or in_corridor:
                        comp_rec = A4ComponentRecord(
                            pair_index=p_curr,
                            track_id=t.track_id,
                            component_id=cd["comp_id"],
                            num_patches=cd["info"]["n_patches"],
                            cpa=cd["info"]["cpa"],
                            r_front=cd["info"]["r_front"],
                            r_min=cd["info"]["r_min"],
                            do_cnt=cd["do_cnt"],
                            adj_cnt=cd["adj_cnt"],
                            match_score=float(cd["total_score"]),
                            mask=cd["info"]["mask"],
                            is_leading_front=False,
                        )
                        accepted_components.append(comp_rec)
                        for rc in zip(*np.nonzero(cd["info"]["mask"])):
                            assert rc not in claimed_patches_this_frame, f"Double claim at {rc}!"
                            claimed_patches_this_frame[rc] = t.track_id

                        all_decision_records.append(
                            MultiComponentDecisionRecord(
                                pair_index=p_curr, timestamp=ts_curr.isoformat(),
                                track_id=t.track_id, candidate_comp_id=cd["comp_id"],
                                num_patches=comp_rec.num_patches, front_radius=comp_rec.r_front,
                                radial_min=comp_rec.r_min, cpa=comp_rec.cpa,
                                direct_overlap_patches=cd["do_cnt"], adjacent_overlap_patches=cd["adj_cnt"],
                                overlap_score=float(cd["overlap_score"]), cpa_diff_deg=float(cd["cpa_diff"]),
                                angular_score=float(cd["ang_score"]), radial_displacement=float(cd["rad_disp"]),
                                kinematic_score=float(cd["kin_score"]), front_bonus=float(cd["front_bonus"]),
                                total_match_score=float(cd["total_score"]), is_accepted=True,
                                rejection_reason="NONE",
                            )
                        )
                    else:
                        all_decision_records.append(
                            MultiComponentDecisionRecord(
                                pair_index=p_curr, timestamp=ts_curr.isoformat(),
                                track_id=t.track_id, candidate_comp_id=cd["comp_id"],
                                num_patches=cd["info"]["n_patches"], front_radius=cd["info"]["r_front"],
                                radial_min=cd["info"]["r_min"], cpa=cd["info"]["cpa"],
                                direct_overlap_patches=cd["do_cnt"], adjacent_overlap_patches=cd["adj_cnt"],
                                overlap_score=float(cd["overlap_score"]), cpa_diff_deg=float(cd["cpa_diff"]),
                                angular_score=float(cd["ang_score"]), radial_displacement=float(cd["rad_disp"]),
                                kinematic_score=float(cd["kin_score"]), front_bonus=float(cd["front_bonus"]),
                                total_match_score=float(cd["total_score"]), is_accepted=False,
                                rejection_reason="INSUFFICIENT_CONTINUITY",
                            )
                        )

                # Track-level continuation handling
                if not accepted_components:
                    t.missed += 1
                    if t.missed >= max_missed_frames:
                        t.terminated = True
                    hist_rec = A4TrackHistory(
                        pair_index=p_curr, timestamp=ts_curr, state="PREDICTED_ONLY",
                        union_mask=np.zeros((64, 64), bool), central_pa=last_cpa, tracked_cpa=t.tracked_cpa,
                        num_components=0, total_patches=0, leading_front_radius=last_front,
                        leading_front_state="PREDICTED", leading_front_comp_id=-1, components=[],
                    )
                    t.history.append(hist_rec)
                    track_state_rows.append({
                        "experiment": "A5b_PA_Width_Weighted_CPA",
                        "track_id": t.track_id,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "state": "PREDICTED_ONLY",
                        "central_pa": last_cpa,
                        "tracked_cpa": t.tracked_cpa,
                        "roi_center_pa": t.adaptive_active_roi[0],
                        "roi_half_width": t.adaptive_active_roi[1],
                        "front_radius": last_front,
                        "leading_front_state": "PREDICTED",
                        "leading_front_comp_id": -1,
                        "num_patches": 0,
                        "num_components": 0,
                        "radial_displacement": 0.0,
                    })
                    leading_front_rows.append({
                        "pair_index": p_curr, "timestamp": ts_curr.isoformat(),
                        "track_id": t.track_id, "leading_front_state": "PREDICTED",
                        "leading_front_radius": last_front, "responsible_comp_id": -1,
                        "radial_displacement": 0.0,
                    })
                    continue

                # Accepted components exist!
                t.missed = 0
                union_mask = np.zeros((64, 64), bool)
                for c in accepted_components:
                    union_mask |= c.mask

                all_cpas = patch_pa[union_mask]
                all_rads = patch_rad[union_mask]
                obs_cpa = float(circular_mean_deg(all_cpas))
                arc_union = minimal_circular_enclosing_arc(all_cpas)
                obs_span = float(arc_union["arc_span"])

                # Determine Leading Front State
                outer_comp = max(accepted_components, key=lambda c: c.r_front)
                outer_front_r = outer_comp.r_front
                rad_disp_front = outer_front_r - last_front if last_obs is not None else 0.0

                if (not outward_enabled) or (last_obs is None) or (rad_disp_front >= -outward_tol):
                    lf_state = "OBSERVED"
                    lf_radius = outer_front_r
                    lf_comp_id = outer_comp.component_id
                    outer_comp.is_leading_front = True
                else:
                    # Outer front missing/retreated, but inner root components observed!
                    lf_state = "PREDICTED"
                    lf_radius = last_front  # retain previous leading front radius
                    lf_comp_id = -1

                new_roi_half = max(default_roi_half, obs_span / 2.0 + roi_padding)
                t.adaptive_active_roi = (obs_cpa, new_roi_half)
                cpa_diff = circular_diff_deg(obs_cpa, t.tracked_cpa)
                t.tracked_cpa = float((t.tracked_cpa + ema_alpha * cpa_diff) % 360.0)
                t.observed_cpa = obs_cpa

                track_ownership_masks[step_idx][union_mask] = t.track_id

                hist_rec = A4TrackHistory(
                    pair_index=p_curr, timestamp=ts_curr, state="OBSERVED",
                    union_mask=union_mask, central_pa=obs_cpa, tracked_cpa=t.tracked_cpa,
                    num_components=len(accepted_components), total_patches=int(union_mask.sum()),
                    leading_front_radius=lf_radius, leading_front_state=lf_state,
                    leading_front_comp_id=lf_comp_id, components=accepted_components,
                )
                t.history.append(hist_rec)

                # HTM record
                roi_center, roi_half = t.adaptive_active_roi
                in_roi_mask = cur_grid & (np.abs(circular_diff_deg(patch_pa, roi_center)) <= roi_half)
                raw_hist = compute_radial_histogram(patch_rad[in_roi_mask], radial_bins)
                conn_hist = compute_radial_histogram(patch_rad[union_mask], radial_bins)

                htm_rec = HTMRecord(
                    pair_index=p_curr,
                    timestamp=ts_curr,
                    raw_radial_activity=raw_hist,
                    accepted_connected_patches=conn_hist,
                    assigned_track_components=conn_hist,
                    observed_radial_front_px=lf_radius,
                    observed_radial_min_px=float(np.min(all_rads)),
                    temporal_continuity_score=1.0,
                )
                t.htm_history.append(htm_rec)

                track_state_rows.append({
                    "experiment": "A5b_PA_Width_Weighted_CPA",
                    "track_id": t.track_id,
                    "pair_index": p_curr,
                    "timestamp": ts_curr.isoformat(),
                    "state": "OBSERVED",
                    "central_pa": obs_cpa,
                    "tracked_cpa": t.tracked_cpa,
                    "roi_center_pa": t.adaptive_active_roi[0],
                    "roi_half_width": t.adaptive_active_roi[1],
                    "front_radius": lf_radius,
                    "leading_front_state": lf_state,
                    "leading_front_comp_id": lf_comp_id,
                    "num_patches": int(union_mask.sum()),
                    "num_components": len(accepted_components),
                    "radial_displacement": rad_disp_front,
                })

                leading_front_rows.append({
                    "pair_index": p_curr,
                    "timestamp": ts_curr.isoformat(),
                    "track_id": t.track_id,
                    "leading_front_state": lf_state,
                    "leading_front_radius": lf_radius,
                    "responsible_comp_id": lf_comp_id,
                    "radial_displacement": rad_disp_front,
                })

                for c in accepted_components:
                    comp_state_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "track_id": t.track_id,
                        "component_id": c.component_id,
                        "num_patches": c.num_patches,
                        "cpa": c.cpa,
                        "r_min": c.r_min,
                        "r_front": c.r_front,
                        "do_cnt": c.do_cnt,
                        "adj_cnt": c.adj_cnt,
                        "match_score": c.match_score,
                        "is_leading_front": (c.component_id == lf_comp_id),
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
                        original_birth_pa=float(c_pas[int(np.argmin(c_rads))]),
                        original_birth_cpa=orig_birth_cpa,
                        initial_angular_extent=init_extent,
                        initial_birth_roi=init_roi,
                        adaptive_active_roi=init_roi,
                        tracked_cpa=orig_birth_cpa,
                        observed_cpa=orig_birth_cpa,
                    )
                    obs_front = float(np.max(c_rads))
                    n_pts = int(c_mask.sum())
                    c_rec = A4ComponentRecord(
                        pair_index=p_curr,
                        track_id=tid,
                        component_id=raw_birth_cid,
                        num_patches=n_pts,
                        cpa=orig_birth_cpa,
                        r_front=obs_front,
                        r_min=min_r,
                        do_cnt=0,
                        adj_cnt=0,
                        match_score=1.0,
                        mask=c_mask,
                        is_leading_front=True,
                    )
                    h_rec = A4TrackHistory(
                        pair_index=p_curr,
                        timestamp=ts_curr,
                        state="OBSERVED",
                        union_mask=c_mask,
                        central_pa=orig_birth_cpa,
                        tracked_cpa=orig_birth_cpa,
                        num_components=1,
                        total_patches=n_pts,
                        leading_front_radius=obs_front,
                        leading_front_state="OBSERVED",
                        leading_front_comp_id=raw_birth_cid,
                        components=[c_rec],
                    )
                    new_track.history.append(h_rec)
                    tracks[tid] = new_track

                    track_ownership_masks[step_idx][c_mask] = tid

                    track_state_rows.append({
                        "experiment": "A5b_PA_Width_Weighted_CPA",
                        "track_id": tid,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "state": "OBSERVED",
                        "central_pa": orig_birth_cpa,
                        "tracked_cpa": orig_birth_cpa,
                        "roi_center_pa": init_roi[0],
                        "roi_half_width": init_roi[1],
                        "front_radius": obs_front,
                        "leading_front_state": "OBSERVED",
                        "leading_front_comp_id": raw_birth_cid,
                        "num_patches": n_pts,
                        "num_components": 1,
                        "radial_displacement": 0.0,
                    })

                    leading_front_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "track_id": tid,
                        "leading_front_state": "OBSERVED",
                        "leading_front_radius": obs_front,
                        "responsible_comp_id": raw_birth_cid,
                        "radial_displacement": 0.0,
                    })

                    comp_state_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "track_id": tid,
                        "component_id": raw_birth_cid,
                        "num_patches": n_pts,
                        "cpa": orig_birth_cpa,
                        "r_min": min_r,
                        "r_front": obs_front,
                        "do_cnt": 0,
                        "adj_cnt": 0,
                        "match_score": 1.0,
                        "is_leading_front": True,
                    })

        context.info(
            f"A5b PA-Width-Weighted CPA Tracking complete. Total tracks: {len(tracks)}, "
            f"Decisions logged: {len(all_decision_records)}, Contested allocations: {len(contested_allocation_rows)}."
        )

        df_track_states = pd.DataFrame(track_state_rows)
        df_comp_states = pd.DataFrame(comp_state_rows)
        df_leading_front = pd.DataFrame(leading_front_rows)
        df_decisions = pd.DataFrame([asdict_rec(r) for r in all_decision_records])
        df_contested = pd.DataFrame(contested_allocation_rows)

        out = TrackingOutput(
            manifest_df=man_df,
            track_states_df=df_track_states,
            patch_component_labels=patch_component_labels,
            patch_activation_grid4=grid4_all,
            frame_cache=det_out.frame_cache,
            geometry=geom,
            total_tracks=len(tracks),
            anchor_tracks=[t.track_id for t in tracks.values() if t.is_anchor],
            birth_direction_references={},
            track_ownership_masks=track_ownership_masks,
            component_states_df=df_comp_states,
            leading_front_states_df=df_leading_front,
            tracking_method="a5b",
            tracking_parameters=effective_params,
            dataset_metadata=det_out.dataset_metadata,
        )
        setattr(out, "adaptive_tracks", tracks)
        setattr(out, "decision_records", all_decision_records)
        setattr(out, "decisions_df", df_decisions)
        setattr(out, "contested_allocation_df", df_contested)

        return adapt_tracking_output(out, max_birth_radius_px=birth_gate_px)

