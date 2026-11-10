"""Stage 32B-2 Rule B Pure Direct Overlap Family Association Stage."""

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
from cme_pipeline.utils.io import compute_dict_hash


class PureDOFamilyAssociationStage(PipelineStage):
    """Executes Stage 32B-2 Rule B Direct Overlap Family Association."""
    name: str = "association"
    version: str = "1.0.0"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": config.association.method,
            "do_threshold": config.association.do_threshold,
            "prevent_auto_merger": config.association.prevent_auto_merger,
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> AssociationOutput:
        context.info("Running PureDOFamilyAssociationStage (Stage 32B-2 Rule B)...")

        # Unwrap inputs
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
        df_states_e1 = tracking_output.track_states_df
        lbls_all = tracking_output.patch_component_labels
        frame_cache = tracking_output.frame_cache
        do_threshold = config.association.do_threshold

        obs_all = df_states_e1[df_states_e1["state"] == "OBSERVED"].copy()
        birth_df = obs_all.sort_values(["track_id", "pair_index"]).groupby("track_id").first().reset_index()
        track_birth_pair = {int(r["track_id"]): int(r["pair_index"]) for _, r in birth_df.iterrows()}
        track_birth_rmin = {int(r["track_id"]): float(r["birth_r_min_px"]) for _, r in birth_df.iterrows()}
        track_is_anchor = {tid: (rmin <= config.birth_gate.max_birth_radius_px) for tid, rmin in track_birth_rmin.items()}

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
                boundary_reason = "Initial frame of full sequence" if f_idx == 0 else f"Cadence gap ({fc['gap_from_prev_min']:.1f} min > 75 min)"
                for _, r in obs_curr.iterrows():
                    tid = int(r["track_id"])
                    fid = f"FAM_T{tid:02d}"
                    track_to_family[tid] = fid
                    family_members.setdefault(fid, []).append(tid)
                    is_anc = track_is_anchor[tid]
                    if is_anc:
                        family_anchors.setdefault(fid, []).append(tid)

                    comp_val = (
                        int(r["component_id"])
                        if "component_id" in r and pd.notnull(r["component_id"])
                        else int(r.get("leading_front_comp_id", -1))
                    )
                    track_membership_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": comp_val,
                        "family_id": fid,
                        "membership_status": "BOUNDARY_SEED",
                        "link_type": "INITIAL_FRAME_SEED" if f_idx == 0 else "POST_GAP_BOUNDARY_SEED",
                        "parent_track_id": np.nan,
                        "direct_overlap": 0.0,
                        "is_solar_root_anchor": is_anc,
                        "decision": "CONFIRMED_SEED",
                        "rationale": boundary_reason,
                    })
                continue

            p_prev = int(man_df.iloc[f_idx - 1]["pair_index"])
            f_prev = f_idx - 1
            obs_prev = obs_all[obs_all["pair_index"] == p_prev].sort_values("track_id")

            parent_info: Dict[int, Dict[str, Any]] = {}
            for _, p_row in obs_prev.iterrows():
                p_tid = int(p_row["track_id"])
                p_cid = (
                    int(p_row["component_id"])
                    if "component_id" in p_row and pd.notnull(p_row["component_id"])
                    else int(p_row.get("leading_front_comp_id", -1))
                )
                m_p = get_track_observed_mask(tracking_output, f_prev, p_tid, fallback_component_id=p_cid)
                parent_info[p_tid] = {
                    "mask": m_p,
                    "area": int(m_p.sum()),
                    "family_id": track_to_family[p_tid],
                    "is_anchor": track_is_anchor[p_tid],
                }

            for _, c_row in obs_curr.iterrows():
                c_tid = int(c_row["track_id"])
                c_cid = (
                    int(c_row["component_id"])
                    if "component_id" in c_row and pd.notnull(c_row["component_id"])
                    else int(c_row.get("leading_front_comp_id", -1))
                )
                m_c = get_track_observed_mask(tracking_output, f_idx, c_tid, fallback_component_id=c_cid)
                c_area = int(np.sum(m_c))
                is_newborn = (track_birth_pair[c_tid] == p_curr)

                curr_parent_evals = []
                for p_tid, p_data in parent_info.items():
                    inter = int(np.sum(p_data["mask"] & m_c))
                    do = float(inter / c_area) if c_area > 0 else 0.0
                    eval_item = {
                        "parent_tid": p_tid,
                        "parent_cid": p_data["mask"].sum(),
                        "parent_family": p_data["family_id"],
                        "parent_is_anchor": p_data["is_anchor"],
                        "inter_patches": inter,
                        "direct_overlap": do,
                        "is_overlapping": (do > do_threshold),
                    }
                    curr_parent_evals.append(eval_item)

                    if eval_item["is_overlapping"]:
                        candidate_edge_rows.append({
                            "pair_index": p_curr,
                            "parent_pair_index": p_prev,
                            "parent_track_id": p_tid,
                            "child_track_id": c_tid,
                            "child_is_newborn": is_newborn,
                            "parent_family_id": p_data["family_id"],
                            "direct_overlap": do,
                            "decision": "RECORDED_EDGE",
                        })

                if not is_newborn:
                    fid = track_to_family[c_tid]
                    track_membership_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": c_tid,
                        "component_id": c_cid,
                        "family_id": fid,
                        "membership_status": "CORE_CONTINUATION" if track_is_anchor[c_tid] else "CHILD_CONTINUATION",
                        "link_type": "CONFIRMED_TRACK_CONTINUITY",
                        "parent_track_id": c_tid,
                        "direct_overlap": 1.0,
                        "is_solar_root_anchor": track_is_anchor[c_tid],
                        "decision": "MAINTAINED_EXISTING_MEMBERSHIP",
                        "rationale": f"Continuous observation of Track {c_tid} from Pair {p_prev}.",
                    })
                else:
                    overlapping_parents = [pe for pe in curr_parent_evals if pe["is_overlapping"]]
                    n_ov = len(overlapping_parents)

                    if n_ov == 1:
                        sole_p = overlapping_parents[0]
                        fid = sole_p["parent_family"]
                        track_to_family[c_tid] = fid
                        family_members.setdefault(fid, []).append(c_tid)
                        if track_is_anchor[c_tid]:
                            family_anchors.setdefault(fid, []).append(c_tid)

                        track_membership_rows.append({
                            "pair_index": p_curr,
                            "timestamp": ts_curr,
                            "track_id": c_tid,
                            "component_id": c_cid,
                            "family_id": fid,
                            "membership_status": "PROVISIONAL_CHILD",
                            "link_type": "SOLE_PARENT_DIRECT_OVERLAP",
                            "parent_track_id": sole_p["parent_tid"],
                            "direct_overlap": sole_p["direct_overlap"],
                            "is_solar_root_anchor": track_is_anchor[c_tid],
                            "decision": "CONFIRMED_PROVISIONAL_CHILD",
                            "rationale": f"Inherited {fid} from Parent {sole_p['parent_tid']} (DO={sole_p['direct_overlap']:.3f}).",
                        })
                    elif n_ov > 1:
                        if track_is_anchor[c_tid]:
                            fid = f"FAM_T{c_tid:02d}"
                            track_to_family[c_tid] = fid
                            family_members.setdefault(fid, []).append(c_tid)
                            family_anchors.setdefault(fid, []).append(c_tid)
                            decision = "CONFIRMED_INDEPENDENT_SEED"
                            rationale = f"Anchor Track {c_tid} has {n_ov} competing parents; preserved as independent seed {fid}."
                        else:
                            fid = f"UNRESOLVED_T{c_tid:02d}"
                            track_to_family[c_tid] = fid
                            family_members.setdefault(fid, []).append(c_tid)
                            decision = "UNRESOLVED_COMPETING"
                            rationale = f"Non-anchor Track {c_tid} has {n_ov} competing parents; auto-merger prevented."

                        track_membership_rows.append({
                            "pair_index": p_curr,
                            "timestamp": ts_curr,
                            "track_id": c_tid,
                            "component_id": c_cid,
                            "family_id": fid,
                            "membership_status": "COMPETING_PARENT_BRANCH",
                            "link_type": "COMPETING_PARENTS",
                            "parent_track_id": np.nan,
                            "direct_overlap": max(pe["direct_overlap"] for pe in overlapping_parents),
                            "is_solar_root_anchor": track_is_anchor[c_tid],
                            "decision": decision,
                            "rationale": rationale,
                        })
                    else:
                        fid = f"FAM_T{c_tid:02d}"
                        track_to_family[c_tid] = fid
                        family_members.setdefault(fid, []).append(c_tid)
                        if track_is_anchor[c_tid]:
                            family_anchors.setdefault(fid, []).append(c_tid)

                        track_membership_rows.append({
                            "pair_index": p_curr,
                            "timestamp": ts_curr,
                            "track_id": c_tid,
                            "component_id": c_cid,
                            "family_id": fid,
                            "membership_status": "INDEPENDENT_SEED",
                            "link_type": "NO_PARENT_OVERLAP",
                            "parent_track_id": np.nan,
                            "direct_overlap": 0.0,
                            "is_solar_root_anchor": track_is_anchor[c_tid],
                            "decision": "CONFIRMED_SEED",
                            "rationale": f"No overlapping parent found (DO <= {do_threshold}).",
                        })

        df_edges = pd.DataFrame(candidate_edge_rows)
        df_track_states = pd.DataFrame(track_membership_rows)

        # Save stage outputs
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_edges.to_csv(stage_dir / "parent_child_edges.csv", index=False)
        df_track_states.to_csv(stage_dir / "track_membership_states.csv", index=False)
        df_edges.to_csv(context.tables_dir / "03_full_day_parent_child_edges.csv", index=False)
        df_track_states.to_csv(context.tables_dir / "02_full_day_e1_track_states.csv", index=False)

        candidate_families = [f for f in family_members.keys() if not f.startswith("UNRESOLVED")]
        context.info(f"Association finished: {len(candidate_families)} candidate families, {len(df_edges)} edges.")

        return AssociationOutput(
            tracking_output=tracking_output,
            recovery_output=recovery_output,
            track_to_family=track_to_family,
            family_members=family_members,
            family_anchors=family_anchors,
            edges_df=df_edges,
            track_membership_df=df_track_states,
            candidate_family_count=len(candidate_families),
        )

