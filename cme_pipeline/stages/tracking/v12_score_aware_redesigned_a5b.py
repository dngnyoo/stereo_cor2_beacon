"""A5b-NR2 Score-Aware Redesigned Normalized Multiplicative Tracking Stage.

Builds on A5b-NR1 by introducing score-aware arbitration for contested components.

In baseline A5b and A5b-NR1, contested components with overlapping tracks are
partitioned purely by angular PA-width Voronoi distance:
    d_i^*(p) = |circ_diff(PA_p, CPA_i)| / max(Delta PA_i, Delta PA_min)

In A5b-NR2, the effective distance is modulated by the track's candidate evidence score:
    d_eff(p, i) = |circ_diff(PA_p, CPA_i)| / (max(Delta PA_i, Delta PA_min) * (S_NR,i ** beta))

When competing tracks have equal evidence, this reduces exactly to baseline PA-width
partitioning. When one track has significantly stronger kinematic and spatial
consistency, its sphere of influence expands proportionally, resolving contested
boundary patches in favor of the physically superior match.

Deterministic fallback:
1. Minimum effective score-aware distance
2. Higher candidate score S_NR
3. Lower track_id
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict as asdict_rec
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import DetectionOutput, PipelineStage, TrackingOutput
from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs
from cme_pipeline.stages.tracking.a5b_nr_scoring import (
    circular_diff_deg,
    compute_angular_consistency,
    compute_detection_confidence,
    compute_kinematic_consistency,
    compute_nr_score,
    compute_spatial_continuity,
)
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
    STRUCT_8,
)
from cme_pipeline.stages.tracking.v06_pa_width_proportional import ContestedAllocationRecord
from cme_pipeline.utils.geometry import circular_mean_deg, minimal_circular_enclosing_arc
from cme_pipeline.utils.io import compute_dict_hash


class PAWidthWeightedCPAScoreAwareTrackingStage(PipelineStage):
    """A5b-NR2 Tracking Stage: Score-Aware Redesigned Normalized Multiplicative Tracking."""

    name: str = "tracking"
    version: str = "5.3.0-nr2"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        params = resolve_and_validate_tracking_parameters("a5b_nr2", config.tracking)
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": "a5b_nr2",
            "date_str": config.date_str,
            "parameters": params,
        }
        return compute_dict_hash(desc)

    def run(self, inputs: Any, config: PipelineConfig, context: PipelineContext) -> TrackingOutput:
        effective_params = resolve_and_validate_tracking_parameters("a5b_nr2", config.tracking)
        context.info("Running PAWidthWeightedCPAScoreAwareTrackingStage (A5b-NR2 Score-Aware Tracking)...")

        # 1. Shared Frozen Preprocessing & Hierarchical Detection Inputs
        if isinstance(inputs, DetectionOutput):
            det_out = inputs
        elif isinstance(inputs, dict) and "detection" in inputs:
            det_out = inputs["detection"]
        else:
            det_out = load_frozen_detection_inputs(config, context)

        geom = det_out.geometry
        patch_pa = geom.patch_pa
        patch_rad = geom.patch_rad
        man_df = det_out.manifest_df
        grid4_all = det_out.patch_activation_grid4
        raw_comp_labels = det_out.raw_component_labels

        n_pairs = len(man_df)
        radial_bins = np.linspace(15.0, 130.0, 47)

        tracks: Dict[int, AdaptiveTrack] = {}
        next_track_id = 1

        all_decision_records: List[MultiComponentDecisionRecord] = []
        normalized_score_rows: List[Dict[str, Any]] = []
        track_state_rows: List[Dict[str, Any]] = []
        comp_state_rows: List[Dict[str, Any]] = []
        leading_front_rows: List[Dict[str, Any]] = []
        contested_allocation_rows: List[Dict[str, Any]] = []

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
        terminate_on_qc = bool(effective_params["terminate_on_qc"])
        score_beta = float(effective_params.get("score_weight_beta", 1.0))

        prev_time = None

        for step_idx in range(n_pairs):
            if (
                "usable_detection" in man_df.columns
                and not bool(man_df.iloc[step_idx]["usable_detection"])
            ):
                status = str(man_df.iloc[step_idx]["observation_status"])
                if terminate_on_qc and status in ("SPATIAL_QC", "GLOBAL_ACTIVATION"):
                    terminated = 0
                    for track in tracks.values():
                        if not track.terminated:
                            track.terminated = True
                            terminated += 1
                    context.info(
                        f"Pair {int(man_df.iloc[step_idx]['pair_index'])}: {status}; "
                        f"terminated {terminated} active tracks at QC boundary."
                    )
                context.info(
                    f"Pair {int(man_df.iloc[step_idx]['pair_index'])}: "
                    f"{man_df.iloc[step_idx]['observation_status']} — "
                    "tracking observation unavailable; no missed-frame update."
                )
                continue

            p_curr = int(man_df.iloc[step_idx]["pair_index"])
            ts_curr = datetime.fromisoformat(str(man_df.iloc[step_idx]["end_time"]))
            cur_grid = grid4_all[step_idx]

            # Cadence gap check
            if prev_time is not None:
                dt_sec = (ts_curr - prev_time).total_seconds()
                if dt_sec > max_cadence_gap_sec:
                    context.info(
                        f"Large cadence gap detected before Pair {p_curr} ({dt_sec/60:.1f} min). "
                        "Terminating active tracks."
                    )
                    for t in tracks.values():
                        if not t.terminated:
                            t.terminated = True
            prev_time = ts_curr

            # -------------------------------------------------------------
            # Step 1: Detect 8-Connected Components on Raw Full grid4
            # -------------------------------------------------------------
            lbl_img, n_comps = label(cur_grid, structure=STRUCT_8)
            active_tracks = [t for t in tracks.values() if not t.terminated]

            # Precompute patch scores for frame detection confidence
            frame_patch_scores: Optional[np.ndarray] = None
            effective_threshold: Optional[float] = None
            if det_out.frame_cache and step_idx < len(det_out.frame_cache):
                f_cache = det_out.frame_cache[step_idx]
                z_glob = f_cache.get("z_global")
                v_mask = f_cache.get("val")
                r_th = f_cache.get("r_th")
                if z_glob is not None and v_mask is not None:
                    blocks = np.where(v_mask, np.abs(z_glob), np.nan).reshape(64, 4, 64, 4).transpose(0, 2, 1, 3).reshape(64, 64, 16)
                    frame_patch_scores = np.partition(blocks, 14, axis=2)[..., 14]
                if isinstance(r_th, dict):
                    effective_threshold = float(r_th.get("effective_threshold_p04", float("nan")))
                elif hasattr(r_th, "get"):
                    effective_threshold = float(r_th.get("effective_threshold_p04", float("nan")))

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

                # Collect observed history for causal velocity estimation
                obs_history: List[Tuple[datetime, float]] = [
                    (h.timestamp, h.leading_front_radius)
                    for h in t.history
                    if getattr(h, "state", "") == "OBSERVED"
                ]

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

                        if do_cnt > 0:
                            elig_cat = "DIRECT_OVERLAP"
                        elif adj_cnt > 0:
                            elig_cat = "ADJACENT_OVERLAP"
                        elif in_roi and in_rad:
                            elig_cat = "ROI_RADIAL_CORRIDOR"
                        else:
                            elig_cat = "INELIGIBLE"

                        if do_cnt > 0 or adj_cnt > 0:
                            a5b_score = 4.0 * ov + 2.0 * ang + 6.0 * kin + 3.0 * front_bonus
                        else:
                            a5b_score = 1.0 * ang + 3.0 * kin + 2.0 * front_bonus

                        # A5b-NR Evidence Components
                        c_val, c_diag = compute_spatial_continuity(prev_mask, m_c, prev_dilation=prev_dil)
                        a_val, a_diag = compute_angular_consistency(cdata["cpa"], t.tracked_cpa, roi_half)
                        k_val, k_diag = compute_kinematic_consistency(cdata["r_front"], prev_front, obs_history, ts_curr)
                        d_val, d_diag = compute_detection_confidence(m_c, frame_patch_scores, effective_threshold)

                        total_score = compute_nr_score(c_val, a_val, k_val, d_val)

                        cand_diag = {
                            "frame_index": step_idx,
                            "pair_index": p_curr,
                            "timestamp": ts_curr.isoformat(),
                            "track_id": t.track_id,
                            "component_id": c_id,
                            "direct_overlap_patches": do_cnt,
                            "adjacent_overlap_patches": adj_cnt,
                            "raw_overlap": float(ov),
                            "raw_angular": float(ang),
                            "raw_kinematic": float(kin),
                            "raw_front_bonus": float(front_bonus),
                            "C": float(c_val),
                            "A": float(a_val),
                            "K": float(k_val),
                            "D": float(d_val),
                            "has_spatial_contact": bool(do_cnt > 0 or adj_cnt > 0),
                            "candidate_score": float(total_score),
                            "normalized_geometric_score": float(total_score),
                            "raw_a5b_score": float(a5b_score),
                            "candidate_eligibility_category": elig_cat,
                            "candidate_rank": -1,
                            "selected_rejected_outcome": "REJECTED",
                            "final_ownership_outcome": "PENDING",
                            "rejection_reason": "PENDING",
                        }
                        normalized_score_rows.append(cand_diag)

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
                            "C": c_val,
                            "A": a_val,
                            "K": k_val,
                            "D": d_val,
                            "total_score": total_score,
                            "info": cdata,
                            "diag": cand_diag,
                            "is_partitioned": False,
                        }
                        track_cands[t.track_id].append(cand_rec)
                        comp_claims[c_id].append((t.track_id, cand_rec))

            # -------------------------------------------------------------
            # Step 3: Shared / Contested Component Resolution (Score-Aware Voronoi Partition)
            # -------------------------------------------------------------
            for c_id, claim_list in comp_claims.items():
                if len(claim_list) <= 1:
                    continue

                overlapping_claims = [cl for cl in claim_list if cl[1]["do_cnt"] > 0]
                if len(overlapping_claims) > 1:
                    # Drop candidates from non-overlapping tracks
                    non_overlapping_tids = [cl[0] for cl in claim_list if cl[1]["do_cnt"] == 0]
                    for tid in non_overlapping_tids:
                        for cd in track_cands[tid]:
                            if cd["comp_id"] == c_id:
                                cd["diag"]["selected_rejected_outcome"] = "REJECTED"
                                cd["diag"]["final_ownership_outcome"] = "LOST_CONTEST"
                                cd["diag"]["rejection_reason"] = "CONTESTED_LOST_TO_DIRECT_OVERLAP"
                        track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]

                    c_mask = comp_info[c_id]["mask"]
                    c_rcs = list(zip(*np.nonzero(c_mask)))
                    comp_tids = [cl[0] for cl in overlapping_claims]

                    # Compute previous observed PA width and candidate scores for each participating track
                    track_widths: Dict[int, float] = {}
                    track_cpas: Dict[int, float] = {}
                    track_scores: Dict[int, float] = {}
                    for tid in comp_tids:
                        t = tracks[tid]
                        track_cpas[tid] = t.tracked_cpa
                        cand_obj = next(cl[1] for cl in overlapping_claims if cl[0] == tid)
                        track_scores[tid] = float(cand_obj["total_score"])

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

                    # NR2 Score-Aware Voronoi partitioning:
                    # d_eff(p, i) = |circ_diff(PA_p, CPA_i)| / (w_i * (S_NR,i ** beta))
                    part_masks = {tid: np.zeros((64, 64), bool) for tid in comp_tids}
                    for r, c in c_rcs:
                        ppa = patch_pa[r, c]
                        best_tid = min(
                            comp_tids,
                            key=lambda tid: (
                                abs(circular_diff_deg(ppa, track_cpas[tid])) / (track_widths[tid] * (track_scores[tid] ** score_beta)),
                                -track_scores[tid],
                                tid,
                            ),
                        )
                        part_masks[best_tid][r, c] = True

                    # Log contested allocation event for auditing
                    for tid in comp_tids:
                        n_assigned = int(part_masks[tid].sum())
                        contested_allocation_rows.append({
                            "pair_index": p_curr,
                            "timestamp": ts_curr.isoformat(),
                            "comp_id": c_id,
                            "total_patches": len(c_rcs),
                            "track_id": tid,
                            "prev_pa_width": track_widths[tid],
                            "candidate_score": track_scores[tid],
                            "allocated_patches": n_assigned,
                            "tracked_cpa": track_cpas[tid],
                            "score_beta": score_beta,
                            "method": "A5b_NR2_Score_Aware_PA_Width_CPA",
                        })

                    # Construct sub-components for each competing track
                    for tid, sub_m in part_masks.items():
                        if sub_m.sum() == 0:
                            for cd in track_cands[tid]:
                                if cd["comp_id"] == c_id:
                                    cd["diag"]["selected_rejected_outcome"] = "REJECTED"
                                    cd["diag"]["final_ownership_outcome"] = "LOST_CONTEST"
                                    cd["diag"]["rejection_reason"] = "CONTESTED_ZERO_PATCHES"
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

                            # Re-evaluate evidence for partitioned sub-component
                            sub_cpa = float(circular_mean_deg(sub_pas))
                            sub_r_front = float(np.max(sub_rads))
                            obs_h = [
                                (h.timestamp, h.leading_front_radius)
                                for h in tracks[tid].history
                                if getattr(h, "state", "") == "OBSERVED"
                            ]
                            c_sub, _ = compute_spatial_continuity(pm, sub_best_m)
                            a_sub, _ = compute_angular_consistency(sub_cpa, tracks[tid].tracked_cpa, tracks[tid].adaptive_active_roi[1])
                            k_sub, _ = compute_kinematic_consistency(
                                sub_r_front,
                                last_obs.leading_front_radius if last_obs else 125.0,
                                obs_h,
                                ts_curr,
                            )
                            d_sub, _ = compute_detection_confidence(sub_best_m, frame_patch_scores, effective_threshold)
                            new_total = compute_nr_score(c_sub, a_sub, k_sub, d_sub)

                            for cand in track_cands[tid]:
                                if cand["comp_id"] == c_id:
                                    cand["is_partitioned"] = True
                                    cand["info"] = {
                                        "mask": sub_best_m,
                                        "n_patches": int(sub_best_m.sum()),
                                        "r_front": sub_r_front,
                                        "r_min": float(np.min(sub_rads)),
                                        "cpa": sub_cpa,
                                        "span": float(sub_arc["arc_span"]),
                                        "arc": sub_arc,
                                    }
                                    cand["rad_disp"] = sub_r_front - (last_obs.leading_front_radius if last_obs else 125.0)
                                    cand["do_cnt"] = int(np.logical_and(pm, sub_best_m).sum())
                                    cand["adj_cnt"] = int(np.logical_and(binary_dilation(pm, STRUCT_8), sub_best_m).sum())
                                    cand["C"] = c_sub
                                    cand["A"] = a_sub
                                    cand["K"] = k_sub
                                    cand["D"] = d_sub
                                    cand["total_score"] = new_total
                                    cand["diag"]["candidate_score"] = float(new_total)
                                    cand["diag"]["C"] = float(c_sub)
                                    cand["diag"]["A"] = float(a_sub)
                                    cand["diag"]["K"] = float(k_sub)
                                    cand["diag"]["D"] = float(d_sub)
                        else:
                            for cd in track_cands[tid]:
                                if cd["comp_id"] == c_id:
                                    cd["diag"]["selected_rejected_outcome"] = "REJECTED"
                                    cd["diag"]["final_ownership_outcome"] = "LOST_CONTEST"
                                    cd["diag"]["rejection_reason"] = "CONTESTED_ZERO_PATCHES"
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]
                elif len(overlapping_claims) == 1:
                    winner_tid = overlapping_claims[0][0]
                    for tid, cand in claim_list:
                        if tid != winner_tid:
                            cand["diag"]["selected_rejected_outcome"] = "REJECTED"
                            cand["diag"]["final_ownership_outcome"] = "LOST_CONTEST"
                            cand["diag"]["rejection_reason"] = "CONTESTED_LOST_TO_DIRECT_OVERLAP"
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]
                else:
                    winner_tid = max(claim_list, key=lambda cl: (cl[1]["total_score"], -cl[0]))[0]
                    for tid, cand in claim_list:
                        if tid != winner_tid:
                            cand["diag"]["selected_rejected_outcome"] = "REJECTED"
                            cand["diag"]["final_ownership_outcome"] = "LOST_CONTEST"
                            cand["diag"]["rejection_reason"] = "CONTESTED_LOST_TO_HIGHER_SCORE"
                            track_cands[tid] = [cd for cd in track_cands[tid] if cd["comp_id"] != c_id]

            # -------------------------------------------------------------
            # Step 4: Multi-Component Ownership Assignment
            # -------------------------------------------------------------
            claimed_patches_this_frame: Dict[Tuple[int, int], int] = {}

            # Assign candidate rank within each active track's candidates
            for t in active_tracks:
                track_all_cands = [r for r in normalized_score_rows if r["track_id"] == t.track_id and r["pair_index"] == p_curr]
                track_all_cands.sort(key=lambda r: (r["candidate_score"], -r["component_id"]), reverse=True)
                for rank, r in enumerate(track_all_cands, 1):
                    r["candidate_rank"] = rank

            # Sort active tracks by candidate match quality priority (deterministic tie-breaking on track_id)
            track_priority = []
            for t in active_tracks:
                cands = track_cands[t.track_id]
                best_score = max([c["total_score"] for c in cands]) if cands else -1.0
                track_priority.append((best_score, t))
            track_priority.sort(key=lambda x: (x[0], -x[1].track_id), reverse=True)

            for _, t in track_priority:
                cands = track_cands[t.track_id]
                last_obs = next(
                    (h for h in reversed(t.history) if getattr(h, "state", "") == "OBSERVED"), None
                )
                last_cpa = last_obs.central_pa if last_obs is not None else t.tracked_cpa
                last_front = last_obs.leading_front_radius if last_obs is not None else 125.0

                accepted_components: List[A4ComponentRecord] = []

                # Evaluate all candidate components for this track (deterministic tie-breaking on comp_id)
                for cd in sorted(cands, key=lambda x: (x["total_score"], -x["comp_id"]), reverse=True):
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
                            cd["diag"]["selected_rejected_outcome"] = "REJECTED"
                            cd["diag"]["final_ownership_outcome"] = "UNCLAIMED_COLLISION"
                            cd["diag"]["rejection_reason"] = "ALL_PATCHES_CLAIMED"
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
                        cd["diag"]["selected_rejected_outcome"] = "ACCEPTED"
                        cd["diag"]["final_ownership_outcome"] = "PARTITIONED" if cd.get("is_partitioned") else "EXCLUSIVE"
                        cd["diag"]["rejection_reason"] = "NONE"
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
                        cd["diag"]["selected_rejected_outcome"] = "REJECTED"
                        cd["diag"]["final_ownership_outcome"] = "REJECTED"
                        cd["diag"]["rejection_reason"] = "INSUFFICIENT_CONTINUITY"
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
                        "experiment": "A5b_NR2_Score_Aware_Normalized",
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
                    lf_state = "PREDICTED"
                    lf_radius = last_front
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
                    "experiment": "A5b_NR2_Score_Aware_Normalized",
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
                        "experiment": "A5b_NR2_Score_Aware_Normalized",
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

            # Resolve any remaining pending candidate statuses for this frame
            for r in normalized_score_rows:
                if r["pair_index"] == p_curr and r["final_ownership_outcome"] == "PENDING":
                    r["selected_rejected_outcome"] = "REJECTED"
                    r["final_ownership_outcome"] = "REJECTED"
                    r["rejection_reason"] = "UNCLAIMED"

        context.info(
            f"A5b-NR2 Score-Aware Tracking complete. Total tracks: {len(tracks)}, "
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
            tracking_method="a5b_nr2",
            tracking_parameters=effective_params,
            dataset_metadata=det_out.dataset_metadata,
        )
        setattr(out, "adaptive_tracks", tracks)
        setattr(out, "decision_records", all_decision_records)
        setattr(out, "decisions_df", df_decisions)
        setattr(out, "contested_allocation_df", df_contested)
        norm_diag_cols = [
            "frame_index", "pair_index", "timestamp", "track_id", "component_id",
            "direct_overlap_patches", "adjacent_overlap_patches",
            "raw_overlap", "raw_angular", "raw_kinematic", "raw_front_bonus",
            "C", "A", "K", "D", "has_spatial_contact",
            "candidate_score", "normalized_geometric_score", "raw_a5b_score",
            "candidate_eligibility_category", "candidate_rank",
            "selected_rejected_outcome", "final_ownership_outcome", "rejection_reason",
        ]
        setattr(out, "normalized_candidate_scores_df", pd.DataFrame(normalized_score_rows, columns=norm_diag_cols))

        return adapt_tracking_output(out, max_birth_radius_px=birth_gate_px)
