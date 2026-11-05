"""Birth-Anchored 1:Many Family Association Stage (Variants B & C).

Maintains CME family directional identity and prevents false mergers across
independent solar root anchors (r_min <= 30 px) with angular deviation > 45 deg.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.direction import (
    create_birth_direction_reference,
    update_birth_direction_reference,
)
from cme_pipeline.core.stage import (
    AssociationOutput,
    BirthDirectionReference,
    PipelineStage,
    RecoveryOutput,
    TrackingOutput,
)
from cme_pipeline.utils.geometry import circular_diff_deg
from cme_pipeline.utils.io import compute_dict_hash


class BirthAnchoredAssociationStage(PipelineStage):
    """Executes Birth-Anchored Direct Overlap Family Association with Directional Integrity."""
    name: str = "association"
    version: str = "2.0.0"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": config.association.method,
            "do_threshold": config.association.do_threshold,
            "direction_threshold_deg": config.association.direction_threshold_deg,
            "enforce_anchor_separation": config.association.enforce_anchor_separation,
            "upstream_tracks": inputs.total_tracks if hasattr(inputs, "total_tracks") else 0,
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> AssociationOutput:
        context.info("Running BirthAnchoredAssociationStage (Variants B & C)...")

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
        df_states = tracking_output.track_states_df
        lbls_all = tracking_output.patch_component_labels
        frame_cache = tracking_output.frame_cache
        do_threshold = config.association.do_threshold
        dir_threshold = config.association.direction_threshold_deg
        enforce_separation = config.association.enforce_anchor_separation

        obs_all = df_states[df_states["state"] == "OBSERVED"].copy()

        # Build or retrieve BirthDirectionReference dictionary
        birth_refs: Dict[int, BirthDirectionReference] = dict(tracking_output.birth_direction_references)

        if not birth_refs:
            # Dynamically construct 2-frame causal references if upstream did not provide them (e.g. Variant B)
            context.info("Constructing two-frame causal BirthDirectionReferences from observed track states...")
            birth_df = obs_all.sort_values(["track_id", "pair_index"]).groupby("track_id").first().reset_index()

            for _, r in birth_df.iterrows():
                tid = int(r["track_id"])
                p_birth = int(r["pair_index"])
                ts_birth = datetime.fromisoformat(str(r["timestamp"]))
                b_rmin = float(r["birth_r_min_px"]) if "birth_r_min_px" in r else float(r["front_radius"])
                cpa_1 = float(r["central_pa"])
                ang_w = float(r["pa_max"] - r["pa_min"]) % 360.0

                ref = create_birth_direction_reference(
                    track_id=tid,
                    birth_pair_id=p_birth,
                    birth_time=ts_birth,
                    birth_r_min_px=b_rmin,
                    central_pa=cpa_1,
                    angular_extent=ang_w,
                    anchor_threshold_px=config.birth_gate.max_birth_radius_px,
                )

                # Check second consecutive observation
                t_obs = obs_all[obs_all["track_id"] == tid].sort_values("pair_index")
                if len(t_obs) >= 2:
                    r2 = t_obs.iloc[1]
                    p2 = int(r2["pair_index"])
                    ts2 = datetime.fromisoformat(str(r2["timestamp"]))
                    cpa_2 = float(r2["central_pa"])
                    is_consec = (p2 == p_birth + 1)
                    ref = update_birth_direction_reference(
                        ref=ref,
                        current_pair_id=p2,
                        current_time=ts2,
                        current_cpa=cpa_2,
                        is_consecutive=is_consec,
                        max_lock_deviation_deg=config.tracking.max_direction_deviation_deg,
                    )

                birth_refs[tid] = ref

        track_to_family: Dict[int, str] = {}
        family_members: Dict[str, List[int]] = {}
        family_anchors: Dict[str, List[int]] = {}
        family_reference_cpas: Dict[str, float] = {}
        candidate_edge_rows = []
        track_membership_rows = []

        # Map each track's birth pair index
        birth_pairs = {
            tid: int(obs_all[obs_all["track_id"] == tid]["pair_index"].min())
            for tid in obs_all["track_id"].unique()
        }

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

                    ref = birth_refs.get(tid)
                    is_anc = ref.is_anchor if ref else False
                    if is_anc:
                        family_anchors.setdefault(fid, []).append(tid)
                        family_reference_cpas[fid] = ref.reference_cpa

                    track_membership_rows.append({
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": int(r["component_id"]),
                        "family_id": fid,
                        "membership_status": "BOUNDARY_SEED",
                        "link_type": "INITIAL_FRAME_SEED" if f_idx == 0 else "POST_GAP_BOUNDARY_SEED",
                        "parent_track_id": np.nan,
                        "direct_overlap": 0.0,
                        "is_solar_root_anchor": is_anc,
                        "delta_cpa_from_family": 0.0,
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
                p_cid = int(p_row["component_id"])
                m_p = (lbls_all[f_prev] == p_cid)
                p_ref = birth_refs.get(p_tid)
                parent_info[p_tid] = {
                    "mask": m_p,
                    "area": int(m_p.sum()),
                    "family_id": track_to_family[p_tid],
                    "is_anchor": p_ref.is_anchor if p_ref else False,
                    "ref_cpa": p_ref.reference_cpa if p_ref else float(p_row["central_pa"]),
                }

            for _, c_row in obs_curr.iterrows():
                c_tid = int(c_row["track_id"])
                c_cid = int(c_row["component_id"])
                m_c = (lbls_all[f_idx] == c_cid)
                c_area = int(np.sum(m_c))
                is_newborn = (birth_pairs[c_tid] == p_curr)
                c_ref = birth_refs.get(c_tid)
                c_is_anchor = c_ref.is_anchor if c_ref else False
                c_direction = c_ref.reference_cpa if (c_ref and c_ref.reference_cpa is not None) else float(c_row["central_pa"])

                curr_parent_evals = []
                for p_tid, p_data in parent_info.items():
                    inter = int(np.sum(p_data["mask"] & m_c))
                    do = float(inter / c_area) if c_area > 0 else 0.0
                    eval_item = {
                        "parent_tid": p_tid,
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
                        "membership_status": "CORE_CONTINUATION" if c_is_anchor else "CHILD_CONTINUATION",
                        "link_type": "CONFIRMED_TRACK_CONTINUITY",
                        "parent_track_id": c_tid,
                        "direct_overlap": 1.0,
                        "is_solar_root_anchor": c_is_anchor,
                        "delta_cpa_from_family": 0.0,
                        "decision": "MAINTAINED_EXISTING_MEMBERSHIP",
                        "rationale": f"Continuous observation of Track {c_tid} from Pair {p_prev}.",
                    })
                else:
                    overlapping_parents = [pe for pe in curr_parent_evals if pe["is_overlapping"]]
                    n_ov = len(overlapping_parents)

                    if n_ov == 1:
                        sole_p = overlapping_parents[0]
                        parent_fid = sole_p["parent_family"]
                        fam_ref = family_reference_cpas.get(parent_fid, parent_info[sole_p["parent_tid"]]["ref_cpa"])
                        delta_cpa = abs(float(circular_diff_deg(c_direction, fam_ref)))

                        # Directional Anchor Separation Condition
                        should_separate = enforce_separation and c_is_anchor and (delta_cpa > dir_threshold)

                        if should_separate:
                            # Independent Anchor Separation
                            fid = f"FAM_T{c_tid:02d}"
                            track_to_family[c_tid] = fid
                            family_members.setdefault(fid, []).append(c_tid)
                            family_anchors.setdefault(fid, []).append(c_tid)
                            family_reference_cpas[fid] = c_direction

                            decision = "SEPARATED_INDEPENDENT_FAMILY"
                            rationale = (
                                f"Anchor Track {c_tid} has delta_cpa={delta_cpa:.1f}deg > {dir_threshold}deg "
                                f"from Family {parent_fid} (ref={fam_ref:.1f}deg); separated as independent CME seed."
                            )
                            status = "INDEPENDENT_ANCHOR_SEED"
                        else:
                            # Inherit parent family
                            fid = parent_fid
                            track_to_family[c_tid] = fid
                            family_members.setdefault(fid, []).append(c_tid)
                            if c_is_anchor:
                                family_anchors.setdefault(fid, []).append(c_tid)

                            decision = "CONFIRMED_PROVISIONAL_CHILD"
                            status = "PROVISIONAL_CHILD" if not c_is_anchor else "ANCHOR_CHILD"
                            rationale = (
                                f"Inherited {fid} from Parent {sole_p['parent_tid']} "
                                f"(DO={sole_p['direct_overlap']:.3f}, delta_cpa={delta_cpa:.1f}deg <= {dir_threshold}deg)."
                            )

                        track_membership_rows.append({
                            "pair_index": p_curr,
                            "timestamp": ts_curr,
                            "track_id": c_tid,
                            "component_id": c_cid,
                            "family_id": fid,
                            "membership_status": status,
                            "link_type": "SOLE_PARENT_DIRECT_OVERLAP",
                            "parent_track_id": sole_p["parent_tid"],
                            "direct_overlap": sole_p["direct_overlap"],
                            "is_solar_root_anchor": c_is_anchor,
                            "delta_cpa_from_family": delta_cpa,
                            "decision": decision,
                            "rationale": rationale,
                        })

                    elif n_ov > 1:
                        # Competing parents -> prevent auto merger
                        if c_is_anchor:
                            fid = f"FAM_T{c_tid:02d}"
                            track_to_family[c_tid] = fid
                            family_members.setdefault(fid, []).append(c_tid)
                            family_anchors.setdefault(fid, []).append(c_tid)
                            family_reference_cpas[fid] = c_direction
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
                            "is_solar_root_anchor": c_is_anchor,
                            "delta_cpa_from_family": np.nan,
                            "decision": decision,
                            "rationale": rationale,
                        })

                    else:
                        # No overlapping parent -> independent seed
                        fid = f"FAM_T{c_tid:02d}"
                        track_to_family[c_tid] = fid
                        family_members.setdefault(fid, []).append(c_tid)
                        if c_is_anchor:
                            family_anchors.setdefault(fid, []).append(c_tid)
                            family_reference_cpas[fid] = c_direction

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
                            "is_solar_root_anchor": c_is_anchor,
                            "delta_cpa_from_family": 0.0,
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
        separated_count = sum(1 for r in track_membership_rows if r["decision"] == "SEPARATED_INDEPENDENT_FAMILY")
        context.info(
            f"BirthAnchoredAssociation complete: {len(candidate_families)} candidate families, "
            f"{separated_count} independent anchor separations."
        )

        return AssociationOutput(
            tracking_output=tracking_output,
            recovery_output=recovery_output,
            track_to_family=track_to_family,
            family_members=family_members,
            family_anchors=family_anchors,
            edges_df=df_edges,
            track_membership_df=df_track_states,
            candidate_family_count=len(candidate_families),
            family_reference_cpas=family_reference_cpas,
            birth_direction_references=birth_refs,
        )

