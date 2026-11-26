"""Downstream Adapter for Independent Seed-Origin Propagation Tracking (SPT).

This adapter fulfills input contracts for downstream CME pipeline stages:
- Stage 04: Association (Identity Association)
- Stage 05: Gates (Family Birth Radius Gate & Consecutive Size-Qualified Persistence Gate)
- Stage 06: Recovery (Signed Dynamic PA Recovery applied after Gates)
- Stage 07: Segmentation (Island-First Pixel Segmentation)
- Stage 08: Catalog Kinematics (Independent Event Kinematics & POS Velocity Estimation)

Guarantees:
1. Does not modify or rely upon A5b baseline stages.
2. Standardizes track ownership masks, radial bounds, and geometry (rsun_px, occulter_r_min).
3. Preserves multi-track separation within bridged components through downstream stages.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import (
    TrackingOutput,
    AssociationOutput,
    BirthGateOutput,
    SizeGateOutput,
    SegmentationOutput,
)
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage
from cme_pipeline.stages.recovery.v02_apply_after_gates import ApplyRecoveryAfterGatesStage
from cme_pipeline.stages.segmentation.v01_island_first_segmentation import IslandFirstSegmentationStage
from cme_pipeline.stages.segmentation.cme_catalog import build_cme_catalog
from cme_pipeline.utils.geometry import circular_mean_deg, minimal_circular_enclosing_arc


# Verified physical solar radius in 256x256 binned pixels
VERIFIED_RSUN_PX = {
    20101212: 8.163,
    20240503: 8.525,
    20240510: 8.525,
}

VERIFIED_OCCULTER_INNER_PX = {
    20101212: 16.326,
    20240503: 17.050,
    20240510: 17.050,
}


@dataclass
class SPTDownstreamPipelineResult:
    """Consolidated downstream pipeline products for an SPT run."""
    tracking_output: TrackingOutput
    association_output: AssociationOutput
    birth_gate_output: BirthGateOutput
    size_gate_output: SizeGateOutput
    recovery_output: SizeGateOutput
    segmentation_output: SegmentationOutput
    event_catalog_df: pd.DataFrame
    frame_measurements_df: pd.DataFrame
    admitted_track_ids: List[int]
    admitted_family_ids: List[str]


def adapt_spt_tracking_output(
    raw_output: TrackingOutput,
    date_str: int | str,
    max_birth_radius_px: float = 30.0,
) -> TrackingOutput:
    """Prepares an SPT TrackingOutput to satisfy downstream stage input contracts.

    Ensures all metadata fields (birth_r_min_px, pa_min, pa_max, rsun_px) are
    strictly populated without altering the underlying SPT ownership masks.
    """
    adapted = copy.copy(raw_output)
    geom = copy.copy(raw_output.geometry)

    # 1. Verify solar radius and occulter geometry
    date_int = int(str(date_str).replace("-", "")[:8])
    if not np.isfinite(getattr(geom, "rsun_px", np.nan)) or geom.rsun_px <= 0:
        geom.rsun_px = VERIFIED_RSUN_PX.get(date_int, 8.525)
    if not np.isfinite(getattr(geom, "occulter_inner_radius_px", np.nan)) or geom.occulter_inner_radius_px <= 0:
        geom.occulter_inner_radius_px = VERIFIED_OCCULTER_INNER_PX.get(date_int, geom.rsun_px * 2.0)
    adapted.geometry = geom

    man_df = raw_output.manifest_df.reset_index(drop=True)
    pair_to_fidx = {int(r["pair_index"]): idx for idx, r in man_df.iterrows()}
    patch_pa = geom.patch_pa
    patch_rad = geom.patch_rad

    # 2. Track ownership masks validation
    own_masks = getattr(raw_output, "track_ownership_masks", None)
    if own_masks is None:
        raise ValueError("SPT TrackingOutput must contain mutually exclusive track_ownership_masks.")
    adapted.track_ownership_masks = own_masks

    # 3. Standardize track_states_df columns
    df = raw_output.track_states_df.copy()
    if df.empty:
        adapted.track_states_df = df
        return adapted

    # Harmonize radial front / front_radius
    if "front_radius" not in df.columns and "radial_front" in df.columns:
        df["front_radius"] = df["radial_front"]
    elif "radial_front" not in df.columns and "front_radius" in df.columns:
        df["radial_front"] = df["front_radius"]

    # Harmonize central_pa / tracked_cpa
    if "tracked_cpa" not in df.columns and "central_pa" in df.columns:
        df["tracked_cpa"] = df["central_pa"]
    elif "central_pa" not in df.columns and "tracked_cpa" in df.columns:
        df["central_pa"] = df["tracked_cpa"]

    # birth_r_min_px
    if "birth_r_min_px" not in df.columns:
        birth_r_map: Dict[int, float] = {}
        obs_sorted = df[df["state"] == "OBSERVED"].sort_values(["track_id", "pair_index"])
        for tid, grp in obs_sorted.groupby("track_id"):
            first_p = int(grp.iloc[0]["pair_index"])
            f_idx = pair_to_fidx.get(first_p)
            if f_idx is not None:
                m = (own_masks[f_idx] == int(tid))
                birth_r_map[int(tid)] = float(np.min(patch_rad[m])) if m.any() else np.nan
        df["birth_r_min_px"] = df["track_id"].map(birth_r_map)

    # is_solar_root_anchor
    if "is_solar_root_anchor" not in df.columns:
        df["is_solar_root_anchor"] = df["birth_r_min_px"] <= max_birth_radius_px

    # original_birth_cpa
    if "original_birth_cpa" not in df.columns:
        birth_cpa_map: Dict[int, float] = {}
        obs_sorted = df[df["state"] == "OBSERVED"].sort_values(["track_id", "pair_index"])
        for tid, grp in obs_sorted.groupby("track_id"):
            birth_cpa_map[int(tid)] = float(grp.iloc[0]["central_pa"])
        df["original_birth_cpa"] = df["track_id"].map(birth_cpa_map)

    # leading_front_state
    if "leading_front_state" not in df.columns:
        df["leading_front_state"] = np.where(df["state"] == "OBSERVED", "OBSERVED", "PREDICTED")

    # leading_front_comp_id
    if "leading_front_comp_id" not in df.columns:
        df["leading_front_comp_id"] = df.get("component_id", -1)

    # num_components
    if "num_components" not in df.columns:
        df["num_components"] = np.where(df["state"] == "OBSERVED", 1, 0)

    # Compute pa_min and pa_max from owned patches if not already present
    if "pa_min" not in df.columns or "pa_max" not in df.columns:
        pa_mins, pa_maxs = [], []
        for _, row in df.iterrows():
            if row["state"] != "OBSERVED":
                pa_mins.append(np.nan)
                pa_maxs.append(np.nan)
                continue
            p_idx = int(row["pair_index"])
            tid = int(row["track_id"])
            f_idx = pair_to_fidx.get(p_idx)
            if f_idx is None:
                pa_mins.append(np.nan)
                pa_maxs.append(np.nan)
                continue
            cur_mask = (own_masks[f_idx] == tid)
            if cur_mask.any():
                arc = minimal_circular_enclosing_arc(patch_pa[cur_mask], margin_deg=0.0)
                pa_mins.append(float(arc["arc_start"]))
                pa_maxs.append(float(arc["arc_end"]))
            else:
                pa_mins.append(np.nan)
                pa_maxs.append(np.nan)
        df["pa_min"] = pa_mins
        df["pa_max"] = pa_maxs

    adapted.track_states_df = df
    return adapted


def run_spt_downstream_pipeline(
    tracking_output: TrackingOutput,
    config: PipelineConfig,
    context: PipelineContext,
) -> SPTDownstreamPipelineResult:
    """Executes Stages 04 through 08 cleanly using adapted SPT outputs."""
    date_val = getattr(config, "date_str", 20101212)
    max_birth = getattr(config.birth_gate, "max_birth_radius_px", 30.0)

    # 1. Adapt TrackingOutput
    adapted_tracking = adapt_spt_tracking_output(tracking_output, date_val, max_birth_radius_px=max_birth)

    # 2. Stage 04: Identity Association
    context.info("SPT Downstream: Running IdentityAssociationStage...")
    assoc_out = IdentityAssociationStage().run(adapted_tracking, config, context)

    # 3. Stage 05: Gates (Birth Radius & Consecutive Size-Qualified Persistence)
    context.info("SPT Downstream: Running FamilyBirthRadiusGateStage...")
    birth_out = FamilyBirthRadiusGateStage().run(assoc_out, config, context)

    context.info("SPT Downstream: Running FamilySizeGateStage...")
    size_out = FamilySizeGateStage().run(birth_out, config, context)

    admitted_fids = sorted(size_out.admitted_family_ids)
    admitted_tids = sorted(size_out.admitted_track_ids)
    context.info(f"SPT Downstream: Gates admitted {len(admitted_fids)} families / {len(admitted_tids)} tracks.")

    # 4. Stage 06: Signed Dynamic PA Recovery
    context.info("SPT Downstream: Running ApplyRecoveryAfterGatesStage...")
    recovery_stage = ApplyRecoveryAfterGatesStage()
    recovery_out = recovery_stage.run(size_out, config, context)
    context.info(f"SPT Downstream: Recovery complete ({recovery_out.total_recovered_patches} recovered patches).")

    # 5. Stage 07: Island-First Pixel Segmentation
    context.info("SPT Downstream: Running IslandFirstSegmentationStage...")
    seg_stage = IslandFirstSegmentationStage()
    seg_out = seg_stage.run(recovery_out, config, context)
    context.info(f"SPT Downstream: Segmentation complete ({seg_out.total_segmented_pixels} cumulative active pixels).")

    # 6. Stage 08: Independent CME Catalog Kinematics
    context.info("SPT Downstream: Building CME Catalog & Kinematics...")
    cat_cfg = getattr(config, "catalog", None)
    solar_r_km = getattr(cat_cfg, "solar_radius_km", 695700.0) if cat_cfg is not None else 695700.0
    event_df, frame_df = build_cme_catalog(seg_out, solar_radius_km=solar_r_km)
    context.info(f"SPT Downstream: Generated {len(event_df)} catalog events.")

    return SPTDownstreamPipelineResult(
        tracking_output=adapted_tracking,
        association_output=assoc_out,
        birth_gate_output=birth_out,
        size_gate_output=size_out,
        recovery_output=recovery_out,
        segmentation_output=seg_out,
        event_catalog_df=event_df,
        frame_measurements_df=frame_df,
        admitted_track_ids=admitted_tids,
        admitted_family_ids=admitted_fids,
    )
