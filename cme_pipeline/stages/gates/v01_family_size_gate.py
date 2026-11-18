"""Pure Family-level Size/Persistence Gate: no Recovery execution."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import BirthGateOutput, PipelineStage, SizeGateOutput
from cme_pipeline.stages.tracking.adapter import get_track_family_observed_mask
from cme_pipeline.utils.io import compute_dict_hash


class FamilySizeGateStage(PipelineStage):
    """Admits Families based strictly on observed size and persistence."""
    name: str = "size_gate"
    version: str = "2.1.0"

    def compute_fingerprint(
        self, inputs: BirthGateOutput, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "min_detected_patches": config.size_gate.min_detected_patches,
            "min_observed_pairs": config.size_gate.min_observed_pairs,
            "size_qualified_persistence": config.size_gate.size_qualified_persistence,
            "upstream_birth_gated_count": len(inputs.birth_gated_family_ids),
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: BirthGateOutput, config: PipelineConfig, context: PipelineContext
    ) -> SizeGateOutput:
        context.info("Running FamilySizeGateStage...")

        assoc_output = inputs.association_output
        track_output = assoc_output.tracking_output

        man_df = track_output.manifest_df
        n_pairs = len(man_df)
        obs_all = track_output.track_states_df[track_output.track_states_df["state"] == "OBSERVED"].copy()

        min_det_threshold = config.size_gate.min_detected_patches
        min_obs_threshold = config.size_gate.min_observed_pairs
        qualified_mode = bool(config.size_gate.size_qualified_persistence)
        if min_det_threshold < 1 or min_obs_threshold < 1:
            raise ValueError("Gate thresholds must be positive")
        family_summary_rows = []
        admitted_fams: List[str] = []
        admitted_track_ids: List[int] = []

        for fid, members in assoc_output.family_members.items():
            if fid.startswith("UNRESOLVED"):
                continue

            m_sorted = sorted(members)
            anchors = assoc_output.family_anchors.get(fid, [])
            has_anchor = fid in inputs.birth_gated_family_ids

            # Compute max detected patch area in any single frame
            max_det = 0
            max_det_pair = None
            active_pairs = []
            qualified_pairs = []

            for f_idx in range(n_pairs):
                p_num = int(man_df.iloc[f_idx]["pair_index"])
                sub_obs = obs_all[(obs_all["pair_index"] == p_num) & (obs_all["track_id"].isin(m_sorted))]
                if len(sub_obs) > 0:
                    active_pairs.append(p_num)
                    cids = sub_obs["component_id"].unique() if "component_id" in sub_obs.columns else None
                    det_union = get_track_family_observed_mask(
                        track_output, f_idx, m_sorted, fallback_component_ids=cids
                    )
                    det_cnt = int(det_union.sum())
                    if det_cnt >= min_det_threshold:
                        qualified_pairs.append(p_num)
                    if det_cnt > max_det:
                        max_det = det_cnt
                        max_det_pair = p_num

            n_observed = len(set(active_pairs))
            n_qualified = len(set(qualified_pairs))
            is_admitted = (
                has_anchor and (
                    n_qualified >= min_obs_threshold if qualified_mode
                    else (max_det >= min_det_threshold and n_observed >= min_obs_threshold)
                )
            )
            status = "ADMITTED" if is_admitted else "REJECTED"
            rejection_reason = "NONE"
            if not has_anchor:
                rejection_reason = inputs.rejected_family_reasons.get(fid, "NO_SOLAR_ROOT_ANCHOR")
            elif qualified_mode and n_qualified < min_obs_threshold:
                rejection_reason = (
                    f"SIZE_QUALIFIED_DURATION_BELOW_{min_obs_threshold}_PAIRS "
                    f"(size>={min_det_threshold}, n={n_qualified})"
                )
            elif max_det < min_det_threshold:
                rejection_reason = f"SIZE_BELOW_{min_det_threshold}_PATCHES (max={max_det})"
            elif n_observed < min_obs_threshold:
                rejection_reason = (
                    f"DURATION_BELOW_{min_obs_threshold}_OBSERVED_PAIRS (n={n_observed})"
                )

            if is_admitted:
                admitted_fams.append(fid)
                admitted_track_ids.extend(m_sorted)

            family_summary_rows.append({
                "family_id": fid,
                "status": status,
                "rejection_reason": rejection_reason,
                "member_count": len(m_sorted),
                "anchor_count": len(anchors),
                "member_track_ids": str(m_sorted),
                "anchor_track_ids": str(sorted(anchors)),
                "first_pair_index": min(active_pairs) if active_pairs else np.nan,
                "last_pair_index": max(active_pairs) if active_pairs else np.nan,
                "n_active_pairs": n_observed,
                "n_size_qualified_pairs": n_qualified,
                "size_qualified_pairs": str(sorted(set(qualified_pairs))),
                "size_qualified_persistence_enabled": qualified_mode,
                "min_required_observed_pairs": min_obs_threshold,
                "min_required_detected_patches": min_det_threshold,
                "max_detected_patches": max_det,
                "max_detected_pair": max_det_pair,
            })

        admitted_track_ids = sorted(list(set(admitted_track_ids)))
        df_fam_summary = pd.DataFrame(family_summary_rows).sort_values("max_detected_patches", ascending=False)
        df_size_decisions = df_fam_summary[
            ["family_id", "status", "rejection_reason", "member_count", "anchor_count",
             "n_active_pairs", "n_size_qualified_pairs", "max_detected_patches"]
        ].copy()

        context.info(
            f"Size Gate Result: {len(admitted_fams)} admitted families, "
            f"{len(admitted_track_ids)} admitted tracks."
        )

        # Save stage tables and arrays
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_fam_summary.to_csv(stage_dir / "family_summary.csv", index=False)
        df_size_decisions.to_csv(stage_dir / "size_gate_decisions.csv", index=False)

        # Standard outputs
        df_fam_summary.to_csv(context.tables_dir / "04_full_day_family_summary.csv", index=False)
        df_size_decisions.to_csv(context.tables_dir / "05_full_day_size_gate_decisions.csv", index=False)

        return SizeGateOutput(
            birth_gate_output=inputs,
            admitted_family_ids=admitted_fams,
            admitted_track_ids=admitted_track_ids,
            family_summary_df=df_fam_summary,
            size_gate_decisions_df=df_size_decisions,
            min_detected_patches=min_det_threshold,
        )

