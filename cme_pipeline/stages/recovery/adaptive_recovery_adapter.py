"""Adapter connecting Adaptive PA-ROI Tracking & Association to Signed Dynamic PA Recovery.

Applies canonical Experiment 12 v2 Signed Dynamic PA Recovery to Adaptive Track Families:
- Pure post-tracking/post-association execution
- Strict separation between original observed patches and newly recovered patches
- Independent per-family recovery without retroactive track identity alteration
- Family-level recovery overlap logging
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import AssociationOutput, GeometryContext, TrackingOutput
from cme_pipeline.stages.recovery.signed_pa_recovery import run_dynamic_patch_recovery_for_tracks


def run_adaptive_family_recovery(
    association_output: AssociationOutput,
    config: PipelineConfig,
    context: PipelineContext,
    target_pair_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Runs Exp 12 Signed Dynamic Patch Recovery for all candidate families in association_output."""
    tracking_output = association_output.tracking_output
    geom = tracking_output.geometry
    man_df = tracking_output.manifest_df
    df_states = tracking_output.track_states_df
    lbls_all = tracking_output.patch_component_labels
    grid4_all = tracking_output.patch_activation_grid4
    frame_cache = tracking_output.frame_cache

    family_members = association_output.family_members
    family_anchors = association_output.family_anchors

    n_pairs = len(man_df)
    if target_pair_indices is not None:
        valid_f_indices = [i for i, p in enumerate(man_df["pair_index"]) if p in target_pair_indices]
    else:
        valid_f_indices = list(range(n_pairs))

    family_recovered_masks: Dict[str, np.ndarray] = {}
    family_original_masks: Dict[str, np.ndarray] = {}
    family_newly_recovered_masks: Dict[str, np.ndarray] = {}

    recovery_growth_rows = []

    # Prepare states DF with required columns for recovery
    df_rec_states = df_states.copy()
    if "front_radius" not in df_rec_states.columns and "radial_front" in df_rec_states.columns:
        df_rec_states["front_radius"] = df_rec_states["radial_front"]

    context.info(f"Running Signed Dynamic PA Recovery across {len(family_members)} candidate families...")

    # For each family, run recovery on its member tracks
    for fid, members in family_members.items():
        if not members:
            continue

        # Run canonical Exp 12 recovery
        rec_mask_3d = run_dynamic_patch_recovery_for_tracks(
            df_states=df_rec_states,
            patch_labels_all=lbls_all,
            grid4_all=grid4_all,
            frame_cache=frame_cache,
            pa_raster=geom.pa_raster,
            radius_raster=geom.radius_raster,
            patch_pa=geom.patch_pa,
            patch_rad=geom.patch_rad,
            target_track_ids=members,
            occulter_inner_radius_px=geom.occulter_inner_radius_px,
            radial_window_px=config.recovery.radial_window_px,
            pa_margin_deg=config.recovery.pa_margin_deg,
        )

        # Build original observed patch mask for this family
        orig_mask_3d = np.zeros((n_pairs, 64, 64), dtype=bool)
        for f_idx in range(n_pairs):
            orig_mask_3d[f_idx] = np.isin(lbls_all[f_idx], members)

        newly_rec_3d = rec_mask_3d & ~orig_mask_3d

        family_recovered_masks[fid] = rec_mask_3d
        family_original_masks[fid] = orig_mask_3d
        family_newly_recovered_masks[fid] = newly_rec_3d

        n_orig = int(orig_mask_3d[valid_f_indices].sum())
        n_rec = int(rec_mask_3d[valid_f_indices].sum())
        n_new = int(newly_rec_3d[valid_f_indices].sum())
        growth_pct = ((n_rec - n_orig) / max(1, n_orig)) * 100.0

        recovery_growth_rows.append({
            "family_id": fid,
            "member_track_count": len(members),
            "anchor_track_count": len(family_anchors.get(fid, [])),
            "original_observed_patches": n_orig,
            "recovered_total_patches": n_rec,
            "newly_recovered_patches": n_new,
            "patch_growth_pct": growth_pct,
        })

    df_growth = pd.DataFrame(recovery_growth_rows)

    # -------------------------------------------------------------
    # Check for recovery overlap between different families
    # -------------------------------------------------------------
    overlap_rows = []
    fam_list = sorted(family_recovered_masks.keys())
    for i in range(len(fam_list)):
        for j in range(i + 1, len(fam_list)):
            f_a = fam_list[i]
            f_b = fam_list[j]
            mask_a = family_recovered_masks[f_a][valid_f_indices]
            mask_b = family_recovered_masks[f_b][valid_f_indices]
            overlap_count = int(np.logical_and(mask_a, mask_b).sum())

            if overlap_count > 0:
                overlap_rows.append({
                    "family_a": f_a,
                    "family_b": f_b,
                    "overlapping_recovered_patches": overlap_count,
                    "status": "DISJOINT_PRESERVED_NO_MERGE",
                })

    df_overlap = pd.DataFrame(overlap_rows) if overlap_rows else pd.DataFrame(
        columns=["family_a", "family_b", "overlapping_recovered_patches", "status"]
    )

    return {
        "family_recovered_masks": family_recovered_masks,
        "family_original_masks": family_original_masks,
        "family_newly_recovered_masks": family_newly_recovered_masks,
        "recovery_growth_df": df_growth,
        "recovery_overlap_df": df_overlap,
        "valid_f_indices": valid_f_indices,
    }

