"""Family-level Size Gate Stage (min_detected_patches >= 5) and Recovery Union."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import BirthGateOutput, PipelineStage, SizeGateOutput
from cme_pipeline.stages.tracking.adapter import get_track_family_observed_mask
from cme_pipeline.utils.io import compute_dict_hash


class FamilySizeGateStage(PipelineStage):
    """Evaluates Family-level Size Gate (>= 5 detected patches) and generates Recovery Masks."""
    name: str = "size_gate"
    version: str = "1.0.0"

    def compute_fingerprint(
        self, inputs: BirthGateOutput, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "min_detected_patches": config.size_gate.min_detected_patches,
            "upstream_birth_gated_count": len(inputs.birth_gated_family_ids),
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: BirthGateOutput, config: PipelineConfig, context: PipelineContext
    ) -> SizeGateOutput:
        context.info("Running FamilySizeGateStage...")

        assoc_output = inputs.association_output
        track_output = assoc_output.tracking_output
        recovery_output = assoc_output.recovery_output

        man_df = track_output.manifest_df
        n_pairs = len(man_df)
        df_states_e1 = track_output.track_states_df
        lbls_all = track_output.patch_component_labels
        obs_all = df_states_e1[df_states_e1["state"] == "OBSERVED"].copy()

        min_det_threshold = config.size_gate.min_detected_patches
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
                    if det_cnt > max_det:
                        max_det = det_cnt
                        max_det_pair = p_num

            is_admitted = has_anchor and (max_det >= min_det_threshold)
            status = "ADMITTED" if is_admitted else "REJECTED"
            rejection_reason = "NONE"
            if not has_anchor:
                rejection_reason = inputs.rejected_family_reasons.get(fid, "NO_SOLAR_ROOT_ANCHOR")
            elif max_det < min_det_threshold:
                rejection_reason = f"SIZE_BELOW_{min_det_threshold}_PATCHES (max={max_det})"

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
                "n_active_pairs": len(active_pairs),
                "max_detected_patches": max_det,
                "max_detected_pair": max_det_pair,
            })

        admitted_track_ids = sorted(list(set(admitted_track_ids)))
        df_fam_summary = pd.DataFrame(family_summary_rows).sort_values("max_detected_patches", ascending=False)
        df_size_decisions = df_fam_summary[
            ["family_id", "status", "rejection_reason", "member_count", "anchor_count", "max_detected_patches"]
        ].copy()

        context.info(
            f"Size Gate Result: {len(admitted_fams)} admitted families, "
            f"{len(admitted_track_ids)} admitted tracks."
        )

        # Execute recovery for admitted tracks using recovery operator
        context.info("Executing recovery for admitted tracks...")
        if recovery_output is not None and recovery_output.recovery_func is not None:
            recovery_runner = recovery_output.recovery_func
        else:
            from cme_pipeline.stages.recovery.v01_signed_pa_recovery import run_dynamic_patch_recovery_for_tracks
            own_masks = getattr(track_output, "track_ownership_masks", None)
            if own_masks is None:
                own_masks = getattr(track_output, "ownership_masks", None)
            def recovery_runner(tids: List[int]) -> np.ndarray:
                return run_dynamic_patch_recovery_for_tracks(
                    df_states=track_output.track_states_df,
                    patch_labels_all=track_output.patch_component_labels,
                    grid4_all=track_output.patch_activation_grid4,
                    frame_cache=track_output.frame_cache,
                    pa_raster=track_output.geometry.pa_raster,
                    radius_raster=track_output.geometry.radius_raster,
                    patch_pa=track_output.geometry.patch_pa,
                    patch_rad=track_output.geometry.patch_rad,
                    target_track_ids=tids,
                    occulter_inner_radius_px=track_output.geometry.occulter_inner_radius_px,
                    radial_window_px=config.recovery.radial_window_px,
                    pa_margin_deg=config.recovery.pa_margin_deg,
                    track_ownership_all=own_masks,
                )

        rec_union = recovery_runner(admitted_track_ids)
        tot_rec_patches = int(rec_union.sum())
        context.info(f"Total cumulative recovered patches: {tot_rec_patches}")

        fam_rec_dict: Dict[str, np.ndarray] = {}
        for fid in admitted_fams:
            f_members = [
                tid for tid, f_name in assoc_output.track_to_family.items()
                if f_name == fid and tid in admitted_track_ids
            ]
            fam_rec_dict["rec_" + fid] = recovery_runner(f_members)

        # Save stage tables and arrays
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_fam_summary.to_csv(stage_dir / "family_summary.csv", index=False)
        df_size_decisions.to_csv(stage_dir / "size_gate_decisions.csv", index=False)
        np.save(stage_dir / "recovered_patch_masks.npy", rec_union)
        np.savez_compressed(stage_dir / "family_recovered_patch_masks.npz", **fam_rec_dict)

        # Standard outputs
        df_fam_summary.to_csv(context.tables_dir / "04_full_day_family_summary.csv", index=False)
        df_size_decisions.to_csv(context.tables_dir / "05_full_day_size_gate_decisions.csv", index=False)
        np.save(context.arrays_dir / "recovered_patch_masks_pipeline_b_full_day.npy", rec_union)
        np.savez_compressed(context.arrays_dir / "family_recovered_patch_masks.npz", **fam_rec_dict)

        return SizeGateOutput(
            birth_gate_output=inputs,
            admitted_family_ids=admitted_fams,
            admitted_track_ids=admitted_track_ids,
            family_summary_df=df_fam_summary,
            size_gate_decisions_df=df_size_decisions,
            recovered_patch_masks=rec_union,
            family_recovered_patch_masks=fam_rec_dict,
            min_detected_patches=min_det_threshold,
            total_recovered_patches=tot_rec_patches,
        )

