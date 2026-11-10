"""Identity (1:1 Track-to-Family) Association Stage.

Used primarily with multi-component trackers (A4, A5b) where spatial fragments
and disconnected CME substructures are already tracked under a single persistent
Track ID, eliminating the need for heuristic parent-child Track merging (Rule B).
Also works cleanly with any single-component tracker (E1, A3) for ablations.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
from cme_pipeline.utils.io import compute_dict_hash


class IdentityAssociationStage(PipelineStage):
    """Maps each Track 1:1 to an independent candidate Family (`FAM_T{tid:02d}`)."""
    name: str = "association"
    version: str = "1.0.0"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": "identity",
            "max_birth_radius_px": config.birth_gate.max_birth_radius_px,
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> AssociationOutput:
        context.info("Running IdentityAssociationStage (1:1 Track-to-Family Direct Mapping)...")

        if isinstance(inputs, RecoveryOutput):
            tracking_output = inputs.tracking_output
            recovery_output = inputs
        elif isinstance(inputs, TrackingOutput):
            tracking_output = inputs
            recovery_output = None
        else:
            raise TypeError(f"Unexpected input type to IdentityAssociationStage: {type(inputs)}")

        man_df = tracking_output.manifest_df
        df_states = tracking_output.track_states_df
        frame_cache = tracking_output.frame_cache

        obs_all = df_states[df_states["state"] == "OBSERVED"].copy()
        birth_df = obs_all.sort_values(["track_id", "pair_index"]).groupby("track_id").first().reset_index()

        # Resolve birth_r_min_px cleanly (supporting adapted and direct outputs)
        tracks_dict = getattr(tracking_output, "adaptive_tracks", {})
        track_birth_rmin: Dict[int, float] = {}
        for _, r in birth_df.iterrows():
            tid = int(r["track_id"])
            if "birth_r_min_px" in birth_df.columns and pd.notnull(r["birth_r_min_px"]):
                track_birth_rmin[tid] = float(r["birth_r_min_px"])
            elif tid in tracks_dict:
                track_birth_rmin[tid] = float(tracks_dict[tid].birth_r_min_px)
            else:
                track_birth_rmin[tid] = float("inf")

        track_birth_pair = {int(r["track_id"]): int(r["pair_index"]) for _, r in birth_df.iterrows()}
        track_is_anchor = {
            tid: (rmin <= config.birth_gate.max_birth_radius_px)
            for tid, rmin in track_birth_rmin.items()
        }

        track_to_family: Dict[int, str] = {}
        family_members: Dict[str, List[int]] = {}
        family_anchors: Dict[str, List[int]] = {}

        for tid in sorted(track_birth_rmin.keys()):
            fid = f"FAM_T{tid:02d}"
            track_to_family[tid] = fid
            family_members[fid] = [tid]
            if track_is_anchor[tid]:
                family_anchors[fid] = [tid]

        track_membership_rows = []
        for f_idx in range(len(man_df)):
            p_curr = int(man_df.iloc[f_idx]["pair_index"])
            ts_curr = str(frame_cache[f_idx]["timestamp"]).replace("T", " ")
            obs_curr = obs_all[obs_all["pair_index"] == p_curr].sort_values("track_id")

            for _, r in obs_curr.iterrows():
                tid = int(r["track_id"])
                fid = track_to_family[tid]
                is_birth = (track_birth_pair.get(tid) == p_curr)
                comp_val = r["component_id"] if "component_id" in r and pd.notnull(r["component_id"]) else r.get("leading_front_comp_id", -1)

                track_membership_rows.append({
                    "pair_index": p_curr,
                    "timestamp": ts_curr,
                    "track_id": tid,
                    "component_id": int(comp_val),
                    "family_id": fid,
                    "membership_status": "IDENTITY_SEED" if is_birth else "CONTINUING_MEMBER",
                    "link_type": "DIRECT_IDENTITY_MAPPING",
                    "parent_track_id": np.nan if is_birth else tid,
                    "direct_overlap": 1.0 if not is_birth else 0.0,
                    "is_solar_root_anchor": track_is_anchor[tid],
                    "decision": "CONFIRMED_IDENTITY_SEED" if is_birth else "CONTINUING_MEMBER",
                    "rationale": "Direct 1:1 Identity Association preserving multi-component track boundary.",
                })

        df_edges = pd.DataFrame(columns=[
            "pair_index", "parent_pair_index", "parent_track_id", "child_track_id",
            "child_is_newborn", "parent_family_id", "direct_overlap", "decision",
        ])
        df_track_membership = pd.DataFrame(track_membership_rows)

        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_edges.to_csv(stage_dir / "parent_child_edges.csv", index=False)
        df_track_membership.to_csv(stage_dir / "track_membership_states.csv", index=False)
        df_edges.to_csv(context.tables_dir / "03_full_day_parent_child_edges.csv", index=False)
        df_track_membership.to_csv(context.tables_dir / "02_full_day_e1_track_states.csv", index=False)

        context.info(f"Identity Association finished: {len(family_members)} 1:1 candidate families.")

        return AssociationOutput(
            tracking_output=tracking_output,
            recovery_output=recovery_output,
            track_to_family=track_to_family,
            family_members=family_members,
            family_anchors=family_anchors,
            edges_df=df_edges,
            track_membership_df=df_track_membership,
            candidate_family_count=len(family_members),
        )

