"""Adaptive PA-ROI 1:Many Family Association Stage.

Maintains track lineage and strict patch ownership:
- Independent Birth Seeds (r_min <= 30 px) form independent Family candidates (FAM_T{id})
- Child tracks inherit Family ONLY if temporal direct overlap (DO > 0.05) and CPA compatibility (Delta CPA <= 45 deg) are verified
- PA ROI overlap alone NEVER merges families
- Competing parents are NOT automatically merged
- Disjoint patch ownership is strictly preserved at the family level
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import (
    AssociationOutput,
    PipelineStage,
    RecoveryOutput,
    TrackingOutput,
)
from cme_pipeline.stages.tracking.adapter import get_track_observed_mask
from cme_pipeline.utils.geometry import circular_diff_deg
from cme_pipeline.utils.io import compute_dict_hash


class AdaptiveROIAssociationStage(PipelineStage):
    """Executes 1:Many Family Association preserving Adaptive Track Lineage."""
    name: str = "association"
    version: str = "3.0.0"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": "adaptive_roi_association",
            "do_threshold": config.association.do_threshold,
            "max_cpa_diff": 45.0,
            "date_str": config.date_str,
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> AssociationOutput:
        context.info("Running AdaptiveROIAssociationStage...")

        if isinstance(inputs, RecoveryOutput):
            tracking_output = inputs.tracking_output
            recovery_output = inputs
        elif isinstance(inputs, TrackingOutput):
            tracking_output = inputs
            recovery_output = None
        else:
            raise TypeError(f"Unexpected input type to association stage: {type(inputs)}")

        man_df = tracking_output.manifest_df
        n_pairs = len(man_df)
        df_states = tracking_output.track_states_df
        lbls_all = tracking_output.patch_component_labels
        frame_cache = tracking_output.frame_cache
        do_threshold = config.association.do_threshold

        obs_all = df_states[df_states["state"] == "OBSERVED"].copy()
        birth_df = obs_all.sort_values(["track_id", "pair_index"]).groupby("track_id").first().reset_index()
        track_birth_pair = {int(r["track_id"]): int(r["pair_index"]) for _, r in birth_df.iterrows()}
        track_birth_rmin = {int(r["track_id"]): float(r["birth_r_min_px"]) for _, r in birth_df.iterrows()}
        track_is_anchor = {int(r["track_id"]): bool(r["is_solar_root_anchor"]) for _, r in birth_df.iterrows()}
        track_birth_cpa = {int(r["track_id"]): float(r["original_birth_cpa"]) for _, r in birth_df.iterrows()}

        track_to_family: Dict[int, str] = {}
        family_members: Dict[str, List[int]] = {}
        family_anchors: Dict[str, List[int]] = {}
        candidate_edge_rows = []
        track_membership_rows = []

        for f_idx in range(n_pairs):
            p_curr = int(man_df.iloc[f_idx]["pair_index"])
            fc = frame_cache[f_idx]
            ts_curr = str(fc["timestamp"]).replace("T", " ")
            obs_curr = obs_all[obs_all["pair_index"] == p_curr].sort_values("track_id")

            has_predecessor = (f_idx > 0) and (not fc["is_large_gap"])

            if not has_predecessor:
                # Boundary / Initial Frame: spawn independent seed families for anchors
                for _, r in obs_curr.iterrows():
                    tid = int(r["track_id"])
                    fid = f"FAM_T{tid:02d}"
                    track_to_family[tid] = fid
                    family_members.setdefault(fid, []).append(tid)
                    is_anc = track_is_anchor.get(tid, False)
                    if is_anc:
                        family_anchors.setdefault(fid, []).append(tid)

                    track_membership_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": tid,
                        "family_id": fid,
                        "membership_status": "BOUNDARY_SEED",
                        "link_type": "INITIAL_FRAME_SEED" if f_idx == 0 else "POST_GAP_BOUNDARY_SEED",
                        "parent_track_id": np.nan,
                        "direct_overlap": 0.0,
                        "is_solar_root_anchor": is_anc,
                        "decision": "CONFIRMED_SEED",
                        "rationale": "Initial frame or post-gap boundary seed",
                    })
                continue

            # Frame f_idx > 0 with predecessor
            p_prev = int(man_df.iloc[f_idx - 1]["pair_index"])
            obs_prev = obs_all[obs_all["pair_index"] == p_prev]
            active_parent_tids = set(obs_prev["track_id"].astype(int))

            cur_lbls = lbls_all[f_idx]
            prev_lbls = lbls_all[f_idx - 1]

            for _, r in obs_curr.iterrows():
                tid = int(r["track_id"])
                is_anc = track_is_anchor.get(tid, False)

                # Continuing track from previous observation
                if track_birth_pair[tid] < p_curr:
                    fid = track_to_family.get(tid, f"FAM_T{tid:02d}")
                    track_membership_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": tid,
                        "family_id": fid,
                        "membership_status": "CONTINUING_MEMBER",
                        "link_type": "SELF_CONTINUATION",
                        "parent_track_id": tid,
                        "direct_overlap": 1.0,
                        "is_solar_root_anchor": is_anc,
                        "decision": "CONTINUING_OBSERVATION",
                        "rationale": "Track continuing across consecutive frames",
                    })
                    continue

                # Newly born track at p_curr
                c_mask = get_track_observed_mask(tracking_output, f_idx, tid, fallback_component_id=tid)
                c_size = int(c_mask.sum())
                if c_size == 0:
                    continue

                # Compute Direct Overlap with active parents
                parents_above_thresh = []
                for pid in active_parent_tids:
                    p_mask = get_track_observed_mask(tracking_output, f_idx - 1, pid, fallback_component_id=pid)
                    inter = int(np.logical_and(p_mask, c_mask).sum())
                    do_val = float(inter / c_size)

                    if do_val > 0.0:
                        candidate_edge_rows.append({
                            "pair_index": p_curr,
                            "timestamp": ts_curr,
                            "child_track_id": tid,
                            "parent_track_id": pid,
                            "intersection_patches": inter,
                            "child_size_patches": c_size,
                            "direct_overlap": do_val,
                            "passed_do_gate": bool(do_val > do_threshold),
                        })

                    if do_val > do_threshold:
                        # Check CPA compatibility (<= 45 deg)
                        p_row = obs_prev[obs_prev["track_id"] == pid].iloc[0]
                        p_cpa = float(p_row["tracked_cpa"])
                        c_cpa = track_birth_cpa[tid]
                        cpa_diff = abs(circular_diff_deg(c_cpa, p_cpa))

                        if cpa_diff <= 45.0:
                            parents_above_thresh.append((pid, do_val, cpa_diff))

                # Family Assignment Decision
                if len(parents_above_thresh) == 1:
                    pid, do_val, cpa_diff = parents_above_thresh[0]
                    parent_fid = track_to_family.get(pid, f"FAM_T{pid:02d}")
                    track_to_family[tid] = parent_fid
                    family_members.setdefault(parent_fid, []).append(tid)
                    if is_anc:
                        family_anchors.setdefault(parent_fid, []).append(tid)

                    track_membership_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": tid,
                        "family_id": parent_fid,
                        "membership_status": "PROVISIONAL_CHILD" if not is_anc else "ANCHOR_CHILD",
                        "link_type": "DIRECT_OVERLAP_CHILD",
                        "parent_track_id": pid,
                        "direct_overlap": do_val,
                        "is_solar_root_anchor": is_anc,
                        "decision": "CONFIRMED_PROVISIONAL_CHILD",
                        "rationale": f"Unique parent T{pid} with DO={do_val:.3f} and Delta CPA={cpa_diff:.1f} deg",
                    })

                elif len(parents_above_thresh) > 1:
                    # Competing parents: DO NOT auto-merge families
                    if is_anc:
                        # Independent anchor maintains independent family
                        fid = f"FAM_T{tid:02d}"
                        track_to_family[tid] = fid
                        family_members.setdefault(fid, []).append(tid)
                        family_anchors.setdefault(fid, []).append(tid)
                        decision = "CONFIRMED_INDEPENDENT_SEED"
                        rationale = f"Competing parents ({[p[0] for p in parents_above_thresh]}); anchor preserved as independent seed"
                    else:
                        fid = "UNRESOLVED_COMPETING"
                        decision = "UNRESOLVED_COMPETING"
                        rationale = f"Multiple competing parents ({[p[0] for p in parents_above_thresh]}); no auto-merger"

                    track_membership_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": tid,
                        "family_id": fid,
                        "membership_status": "COMPETING_PARENT_BRANCH",
                        "link_type": "MULTI_PARENT_COMPETITION",
                        "parent_track_id": parents_above_thresh[0][0],
                        "direct_overlap": parents_above_thresh[0][1],
                        "is_solar_root_anchor": is_anc,
                        "decision": decision,
                        "rationale": rationale,
                    })

                else:
                    # No parent with DO > threshold
                    if is_anc:
                        fid = f"FAM_T{tid:02d}"
                        track_to_family[tid] = fid
                        family_members.setdefault(fid, []).append(tid)
                        family_anchors.setdefault(fid, []).append(tid)
                        status = "SOLAR_ROOT_SEED"
                        decision = "CONFIRMED_SEED"
                        rationale = "No parent DO > 0.05; solar root anchor qualifies as independent family seed"
                    else:
                        fid = "UNASSOCIATED_CORRIDOR_ONLY"
                        status = "UNCONFIRMED_CORRIDOR_CANDIDATE"
                        decision = "UNCONFIRMED_CORRIDOR_CANDIDATE"
                        rationale = "No parent DO > 0.05 and r_min > 30 px"

                    track_membership_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": tid,
                        "family_id": fid,
                        "membership_status": status,
                        "link_type": "NO_DIRECT_OVERLAP",
                        "parent_track_id": np.nan,
                        "direct_overlap": 0.0,
                        "is_solar_root_anchor": is_anc,
                        "decision": decision,
                        "rationale": rationale,
                    })

        df_membership = pd.DataFrame(track_membership_rows)
        df_edges = pd.DataFrame(candidate_edge_rows)

        # Family Summary
        family_summary_rows = []
        for fid, members in family_members.items():
            anchors = family_anchors.get(fid, [])
            mem_obs = obs_all[obs_all["track_id"].isin(members)]
            first_p = int(mem_obs["pair_index"].min()) if not mem_obs.empty else -1
            last_p = int(mem_obs["pair_index"].max()) if not mem_obs.empty else -1
            n_active = mem_obs["pair_index"].nunique() if not mem_obs.empty else 0

            # Max detected patches across frames
            if not mem_obs.empty:
                framewise_patches = mem_obs.groupby("pair_index")["num_patches"].sum()
                max_p = int(framewise_patches.max())
                max_pair = int(framewise_patches.idxmax())
            else:
                max_p, max_pair = 0, -1

            family_summary_rows.append({
                "family_id": fid,
                "status": "CANDIDATE",
                "member_count": len(members),
                "anchor_count": len(anchors),
                "member_track_ids": str(sorted(members)),
                "anchor_track_ids": str(sorted(anchors)),
                "first_pair_index": first_p,
                "last_pair_index": last_p,
                "n_active_pairs": n_active,
                "max_detected_patches": max_p,
                "max_detected_pair": max_pair,
            })

        df_families = pd.DataFrame(family_summary_rows)

        context.info(f"Adaptive Association complete. Total families spawned: {len(df_families)}.")

        # Save stage tables
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_membership.to_csv(stage_dir / "track_family_membership.csv", index=False)
        df_edges.to_csv(stage_dir / "parent_child_edges.csv", index=False)
        df_families.to_csv(stage_dir / "family_summary.csv", index=False)

        # Save context tables
        df_membership.to_csv(context.tables_dir / "04_adaptive_track_states.csv", index=False)
        df_edges.to_csv(context.tables_dir / "05_adaptive_parent_child_edges.csv", index=False)
        df_families.to_csv(context.tables_dir / "06_adaptive_family_summary.csv", index=False)

        out = AssociationOutput(
            tracking_output=tracking_output,
            recovery_output=recovery_output,
            track_to_family=track_to_family,
            family_members=family_members,
            family_anchors=family_anchors,
            edges_df=df_edges,
            track_membership_df=df_membership,
            candidate_family_count=len(df_families),
        )
        out.family_summary_df = df_families
        return out
