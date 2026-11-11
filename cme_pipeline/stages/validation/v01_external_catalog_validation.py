"""External Catalogue Validation Stage (SEEDS, CACTus, MVCC, Göttingen)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import label

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import PipelineStage, SegmentationOutput, CMEProductsOutput, ValidationOutput
from cme_pipeline.utils.geometry import (
    circular_diff_deg,
    circular_sector_overlap,
    minimal_circular_enclosing_arc,
    parse_iso_dt,
)
from cme_pipeline.utils.io import compute_dict_hash


STRUCT_8 = np.ones((3, 3), bool)


class ExternalValidationStage(PipelineStage):
    """Evaluates final pixel segmentation against external catalogues."""
    name: str = "validation"
    version: str = "1.0.0"

    def compute_fingerprint(
        self, inputs: SegmentationOutput, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "catalogs": config.validation.catalogs,
            "upstream_segmented_pixels": (inputs.segmentation_output.total_segmented_pixels
                                          if isinstance(inputs, CMEProductsOutput)
                                          else inputs.total_segmented_pixels),
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: SegmentationOutput, config: PipelineConfig, context: PipelineContext
    ) -> ValidationOutput:
        context.info("Running ExternalValidationStage (Catalogue Recalculation)...")
        if isinstance(inputs, CMEProductsOutput):
            inputs = inputs.segmentation_output

        val_root = context.repo_root / "validation_20101212"
        exp04_dir = val_root / "experiments/04_d7_external_catalog_validation"
        catalogs_dir = val_root / "catalogs"

        track_output = inputs.size_gate_output.birth_gate_output.association_output.tracking_output
        geom = track_output.geometry
        pa_raster = geom.pa_raster
        rad_raster = geom.radius_raster / geom.rsun_px

        prop_masks = inputs.final_pixel_segmentation_union
        df_timeline = pd.read_csv(exp04_dir / "results/01_catalog_audit/dec12_frame_timeline.csv")
        n_frames = len(df_timeline)
        beacon_times = [datetime.fromisoformat(t) for t in df_timeline["timestamp_utc"]]

        # Extract 8-connected candidate profiles for each frame
        prop_frame_islands: Dict[int, List[Dict[str, Any]]] = {}
        for f_idx in range(n_frames):
            pair_idx = int(df_timeline.iloc[f_idx]["pair_index"])
            mask = prop_masks[f_idx]
            labeled, num_features = label(mask, structure=STRUCT_8)
            isl_list = []
            for feat_id in range(1, num_features + 1):
                y_pts, x_pts = np.nonzero(labeled == feat_id)
                pas = pa_raster[y_pts, x_pts]
                rads = rad_raster[y_pts, x_pts]
                arc = minimal_circular_enclosing_arc(pas, margin_deg=0.0)
                isl_list.append({
                    "feat_id": feat_id,
                    "area_px": len(y_pts),
                    "arc_center": arc["arc_center"],
                    "arc_span": arc["arc_span"],
                    "rad_lead": float(rads.max()),
                    "y_pts": y_pts,
                    "x_pts": x_pts,
                })
            prop_frame_islands[pair_idx] = isl_list

        frame_level_rows: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------------
        # 1. SEEDS Validation
        # ---------------------------------------------------------------------
        df_s_base = pd.read_csv(exp04_dir / "results/04_event_scores/seeds_frame_scores.csv")
        s_eval_base = df_s_base[df_s_base["is_evaluable"] == True].copy()
        s_ious = []
        s_rad_errs = []

        for _, r in s_eval_base.iterrows():
            p_idx = int(r["pair_index"])
            bpa = float(r["seeds_bpa"])
            w = float(r["seeds_width"])
            cat_lead = float(r["seeds_lead"]) if pd.notnull(r["seeds_lead"]) else np.nan
            ev_id = r["parent_event_id"]

            islands = prop_frame_islands[p_idx]
            cand_list = [
                isl for isl in islands
                if circular_sector_overlap(bpa, w, isl["arc_center"], isl["arc_span"])["inter_deg"] > 0
                or abs(circular_diff_deg(bpa, isl["arc_center"])) <= (w / 2.0 + 10.0)
            ]

            if len(cand_list) == 0:
                iou = 0.0
                rad_err = np.nan
            else:
                comb_pas = np.concatenate([pa_raster[isl["y_pts"], isl["x_pts"]] for isl in cand_list])
                c_arc = minimal_circular_enclosing_arc(comb_pas)
                ov = circular_sector_overlap(bpa, w, c_arc["arc_center"], c_arc["arc_span"])
                iou = ov["iou"]
                c_lead = max(isl["rad_lead"] for isl in cand_list)
                rad_err = abs(c_lead - cat_lead) if pd.notnull(cat_lead) else np.nan

            s_ious.append(iou)
            if pd.notnull(rad_err):
                s_rad_errs.append(rad_err)

            frame_level_rows.append({
                "catalog": "SEEDS",
                "event_id": ev_id,
                "pair_index": p_idx,
                "angular_iou": iou,
                "radial_error_rsun": rad_err,
            })

        mean_seeds_iou = float(np.mean(s_ious)) if s_ious else 0.0
        mean_seeds_rad = float(np.mean(s_rad_errs)) if s_rad_errs else np.nan

        # ---------------------------------------------------------------------
        # 2. CACTus Validation (All CMEs and CME 0024)
        # ---------------------------------------------------------------------
        cactus_events = pd.read_csv(catalogs_dir / "cactus/2010-12-12_CACTus_COR2A_events.csv")
        cactus_event_ious: List[float] = []
        c0024_iou: float = 0.0

        for _, ev in cactus_events.iterrows():
            ev_id = ev["stable_event_id"]
            if ev.get("event_type") != "CME":
                continue
            t0 = parse_iso_dt(ev["t0"])
            dt0_h = float(ev["dt0"]) if pd.notnull(ev["dt0"]) else 0.0
            t_end = t0 + timedelta(hours=dt0_h)
            cpa, width = float(ev["pa"]), float(ev["da"])

            matching_f_indices = [idx for idx, t in enumerate(beacon_times) if t0 <= t <= t_end]
            if len(matching_f_indices) == 0:
                continue

            ev_ious_prop = []
            for f_idx in matching_f_indices:
                p_idx = int(df_timeline.iloc[f_idx]["pair_index"])
                islands = prop_frame_islands[p_idx]
                cand_list = [
                    isl for isl in islands
                    if circular_sector_overlap(cpa, width, isl["arc_center"], isl["arc_span"])["inter_deg"] > 0
                    or abs(circular_diff_deg(cpa, isl["arc_center"])) <= (width / 2.0 + 10.0)
                ]

                if len(cand_list) == 0:
                    iou = 0.0
                else:
                    comb_pas = np.concatenate([pa_raster[isl["y_pts"], isl["x_pts"]] for isl in cand_list])
                    c_arc = minimal_circular_enclosing_arc(comb_pas)
                    ov = circular_sector_overlap(cpa, width, c_arc["arc_center"], c_arc["arc_span"])
                    iou = ov["iou"]

                ev_ious_prop.append(iou)
                frame_level_rows.append({
                    "catalog": "CACTus",
                    "event_id": ev_id,
                    "pair_index": p_idx,
                    "angular_iou": iou,
                    "radial_error_rsun": np.nan,
                })

            ev_mean_iou = float(np.mean(ev_ious_prop))
            cactus_event_ious.append(ev_mean_iou)
            if ev_id == "CACTUS_COR2A_CME_0024":
                c0024_iou = ev_mean_iou

        mean_cactus_all_iou = float(np.mean(cactus_event_ious)) if cactus_event_ious else 0.0
        mean_cactus_0024_iou = c0024_iou

        # ---------------------------------------------------------------------
        # 3. MVCC / CORSET Validation
        # ---------------------------------------------------------------------
        df_m_base = pd.read_csv(exp04_dir / "results/04_event_scores/mvcc_frame_scores.csv")
        m_eval_base = df_m_base[df_m_base["is_evaluable"] == True].copy()
        m_ious = []
        m_rad_errs = []

        for _, r in m_eval_base.iterrows():
            p_idx = int(r["pair_index"])
            cpa_col = "cpa_deg" if "cpa_deg" in r else "mvcc_cpa"
            w_col = "width_deg" if "width_deg" in r else "mvcc_width"
            lead_col = "cor_height_rsun" if "cor_height_rsun" in r else "mvcc_lead"

            cpa = float(r[cpa_col])
            w = float(r[w_col])
            cat_lead = float(r[lead_col]) if pd.notnull(r[lead_col]) else np.nan
            ev_id = r["parent_event_id"] if "parent_event_id" in r else "MVCC_01"

            islands = prop_frame_islands[p_idx]
            cand_list = [
                isl for isl in islands
                if circular_sector_overlap(cpa, w, isl["arc_center"], isl["arc_span"])["inter_deg"] > 0
                or abs(circular_diff_deg(cpa, isl["arc_center"])) <= (w / 2.0 + 10.0)
            ]

            if len(cand_list) == 0:
                iou = 0.0
                rad_err = np.nan
            else:
                comb_pas = np.concatenate([pa_raster[isl["y_pts"], isl["x_pts"]] for isl in cand_list])
                c_arc = minimal_circular_enclosing_arc(comb_pas)
                ov = circular_sector_overlap(cpa, w, c_arc["arc_center"], c_arc["arc_span"])
                iou = ov["iou"]
                c_lead = max(isl["rad_lead"] for isl in cand_list)
                rad_err = abs(c_lead - cat_lead) if pd.notnull(cat_lead) else np.nan

            m_ious.append(iou)
            if pd.notnull(rad_err):
                m_rad_errs.append(rad_err)

            frame_level_rows.append({
                "catalog": "MVCC",
                "event_id": ev_id,
                "pair_index": p_idx,
                "angular_iou": iou,
                "radial_error_rsun": rad_err,
            })

        mean_mvcc_iou = float(np.mean(m_ious)) if m_ious else 0.0
        mean_mvcc_rad = float(np.mean(m_rad_errs)) if m_rad_errs else np.nan

        # ---------------------------------------------------------------------
        # 4. Göttingen Validation
        # ---------------------------------------------------------------------
        t0_g = datetime.strptime("2010-12-12 08:08:15", "%Y-%m-%d %H:%M:%S")
        matching_g = [idx for idx, t in enumerate(beacon_times) if abs((t - t0_g).total_seconds()) <= 1800.0]
        g_ious = []

        for f_idx in matching_g:
            p_idx = int(df_timeline.iloc[f_idx]["pair_index"])
            islands = prop_frame_islands[p_idx]
            cand_list = [
                isl for isl in islands
                if circular_sector_overlap(None, 90.0, isl["arc_center"], isl["arc_span"], c1_is_interval=(45.0, 135.0))["inter_deg"] > 0
            ]

            if len(cand_list) == 0:
                iou = 0.0
            else:
                comb_pas = np.concatenate([pa_raster[isl["y_pts"], isl["x_pts"]] for isl in cand_list])
                c_arc = minimal_circular_enclosing_arc(comb_pas)
                ov = circular_sector_overlap(None, 90.0, c_arc["arc_center"], c_arc["arc_span"], c1_is_interval=(45.0, 135.0))
                iou = ov["iou"]

            g_ious.append(iou)
            frame_level_rows.append({
                "catalog": "Göttingen",
                "event_id": "GOTTINGEN_BESTOF_566a",
                "pair_index": p_idx,
                "angular_iou": iou,
                "radial_error_rsun": np.nan,
            })

        mean_gottingen_iou = float(np.mean(g_ious)) if g_ious else 0.0

        # Build comparison summary table
        comparison_rows = [
            {
                "Reference": "SEEDS",
                "Angular IoU": round(mean_seeds_iou, 4),
                "Radial Error": f"{mean_seeds_rad:.4f} R_sun" if pd.notnull(mean_seeds_rad) else "N/A",
                "Evaluable Frames": len(s_ious),
            },
            {
                "Reference": "CACTus (All CMEs)",
                "Angular IoU": round(mean_cactus_all_iou, 4),
                "Radial Error": "N/A",
                "Evaluable Frames": len(cactus_event_ious),
            },
            {
                "Reference": "CACTus (CME 0024)",
                "Angular IoU": round(mean_cactus_0024_iou, 4),
                "Radial Error": "N/A",
                "Evaluable Frames": 11,
            },
            {
                "Reference": "MVCC / CORSET",
                "Angular IoU": round(mean_mvcc_iou, 4),
                "Radial Error": f"{mean_mvcc_rad:.4f} R_sun" if pd.notnull(mean_mvcc_rad) else "N/A",
                "Evaluable Frames": len(m_ious),
            },
            {
                "Reference": "Göttingen",
                "Angular IoU": round(mean_gottingen_iou, 4),
                "Radial Error": "N/A",
                "Evaluable Frames": len(g_ious),
            },
        ]
        df_comparison = pd.DataFrame(comparison_rows)
        df_frames = pd.DataFrame(frame_level_rows)

        context.info("External Validation Results:")
        for _, row in df_comparison.iterrows():
            context.info(f"  {row['Reference']:<20}: Angular IoU={row['Angular IoU']:.4f}, Radial Error={row['Radial Error']}")

        # Save stage tables
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_comparison.to_csv(stage_dir / "comparison_external_validation_metrics.csv", index=False)
        df_frames.to_csv(stage_dir / "frame_level_external_validation.csv", index=False)

        # Standard outputs
        df_comparison.to_csv(context.tables_dir / "08_comparison_external_validation_metrics.csv", index=False)
        df_frames.to_csv(context.tables_dir / "09_frame_level_external_validation.csv", index=False)

        return ValidationOutput(
            segmentation_output=inputs,
            comparison_metrics_df=df_comparison,
            frame_level_scores_df=df_frames,
            event_level_scores_df=df_comparison,
            detailed_metrics={
                "seeds_iou": mean_seeds_iou,
                "seeds_rad_err": mean_seeds_rad,
                "cactus_all_iou": mean_cactus_all_iou,
                "cactus_0024_iou": mean_cactus_0024_iou,
                "mvcc_iou": mean_mvcc_iou,
                "mvcc_rad_err": mean_mvcc_rad,
                "gottingen_iou": mean_gottingen_iou,
            },
        )

