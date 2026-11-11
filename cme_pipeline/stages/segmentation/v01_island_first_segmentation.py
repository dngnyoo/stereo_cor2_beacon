"""Stage 26 Island-First Final Pixel Segmentation Stage."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, label

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import PipelineStage, SegmentationOutput, SizeGateOutput
from cme_pipeline.utils.io import compute_dict_hash


STRUCT_8 = np.ones((3, 3), bool)


class IslandFirstSegmentationStage(PipelineStage):
    """Executes Stage 26 Island-First Sub-Patch Pixel Segmentation."""
    name: str = "segmentation"
    version: str = "1.0.0"

    def compute_fingerprint(
        self, inputs: SizeGateOutput, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": config.segmentation.method,
            "min_island_area_px": config.segmentation.min_island_area_px,
            "max_hole_fill_px": config.segmentation.max_hole_fill_px,
            "occulter_r_min_px": config.segmentation.occulter_r_min_px,
            "fov_r_max_px": config.segmentation.fov_r_max_px,
            "upstream_recovered_patches": inputs.total_recovered_patches,
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: SizeGateOutput, config: PipelineConfig, context: PipelineContext
    ) -> SegmentationOutput:
        context.info("Running IslandFirstSegmentationStage (Stage 26 Algorithm)...")

        birth_output = inputs.birth_gate_output
        assoc_output = birth_output.association_output
        track_output = assoc_output.tracking_output

        man_df = track_output.manifest_df
        n_pairs = len(man_df)
        frame_cache = track_output.frame_cache
        geom = track_output.geometry

        admitted_fams = inputs.admitted_family_ids
        rec_union = inputs.recovered_patch_masks
        fam_rec_dict = inputs.family_recovered_patch_masks

        min_island_area = config.segmentation.min_island_area_px
        max_hole_fill = config.segmentation.max_hole_fill_px
        occulter_r_min = config.segmentation.occulter_r_min_px
        fov_r_max = config.segmentation.fov_r_max_px

        final_pixel_union = np.zeros((n_pairs, 256, 256), dtype=bool)
        family_pixel_masks: Dict[str, np.ndarray] = {
            fid: np.zeros((n_pairs, 256, 256), dtype=bool) for fid in admitted_fams
        }
        pixel_summary_rows = []

        for f_idx in range(n_pairs):
            fc = frame_cache[f_idx]
            p_num = fc["pair_idx"]
            ts_str = str(fc["timestamp"]).replace("T", " ")
            usable = fc["usable"]
            z_glob = fc["z_global"]

            # Upscale 4x4 patches to 256x256 pixel ROI
            rec_m = rec_union[f_idx]
            roi_px = np.kron(rec_m, np.ones((4, 4), bool)) & usable

            # Occulter and outer FOV boundaries
            r_dist = geom.radius_raster
            valid_fov = (r_dist >= occulter_r_min) & (r_dist <= fov_r_max)
            roi_eff = roi_px & valid_fov

            # Z_global > 0 condition
            raw_core = roi_eff & (z_glob > 0.0)

            # 8-connectivity island labeling
            labeled_islands, n_isl = label(raw_core, structure=STRUCT_8)
            retained_islands = np.zeros_like(raw_core, dtype=bool)

            for i_id in range(1, n_isl + 1):
                isl_m = (labeled_islands == i_id)
                if isl_m.sum() >= min_island_area:
                    retained_islands |= isl_m

            # Conservative morphological closure & hole filling <= 16 px strictly within ROI
            filled_core = binary_fill_holes(retained_islands)
            holes = filled_core & (~retained_islands)
            hole_lbls, n_h = label(holes, structure=STRUCT_8)
            for h_id in range(1, n_h + 1):
                h_m = (hole_lbls == h_id)
                if h_m.sum() <= max_hole_fill:
                    retained_islands |= h_m

            final_f_mask = retained_islands & roi_eff
            final_pixel_union[f_idx] = final_f_mask

            # Assign pixels to families based on family recovery ROI
            for fid in admitted_fams:
                f_rec = fam_rec_dict["rec_" + fid][f_idx]
                f_roi = np.kron(f_rec, np.ones((4, 4), bool)) & roi_eff
                family_pixel_masks[fid][f_idx] = final_f_mask & f_roi

            pixel_summary_rows.append({
                "f_idx": f_idx,
                "pair_index": p_num,
                "timestamp": ts_str,
                "recovery_patches": int(rec_m.sum()),
                "recovery_roi_pixels": int(roi_eff.sum()),
                "segmented_pixels": int(final_f_mask.sum()),
                "fill_fraction": float(final_f_mask.sum() / roi_eff.sum()) if roi_eff.sum() > 0 else 0.0,
                "n_islands": int(n_isl),
                "is_large_gap": fc["is_large_gap"],
            })

        tot_segmented_px = int(final_pixel_union.sum())
        context.info(f"Final pixel segmentation complete: {tot_segmented_px} cumulative active pixels.")

        df_pixel_summary = pd.DataFrame(pixel_summary_rows)

        # Family-level pixel statistics
        fam_stat_rows = []
        for fid in admitted_fams:
            f_mask_3d = family_pixel_masks[fid]
            tot_px = int(f_mask_3d.sum())
            max_px = int(np.max([f_mask_3d[f].sum() for f in range(n_pairs)]))
            act_pairs = [int(man_df.iloc[f]["pair_index"]) for f in range(n_pairs) if f_mask_3d[f].sum() > 0]
            fam_stat_rows.append({
                "family_id": fid,
                "total_segmented_pixels": tot_px,
                "max_segmented_pixels": max_px,
                "n_active_pairs": len(act_pairs),
                "first_pair": min(act_pairs) if act_pairs else np.nan,
                "last_pair": max(act_pairs) if act_pairs else np.nan,
            })

        df_fam_stats = pd.DataFrame(fam_stat_rows).sort_values("total_segmented_pixels", ascending=False)

        # Save stage tables and arrays
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_pixel_summary.to_csv(stage_dir / "framewise_pixel_summary.csv", index=False)
        df_fam_stats.to_csv(stage_dir / "family_pixel_statistics.csv", index=False)
        np.save(stage_dir / "final_pixel_segmentation_union.npy", final_pixel_union)
        np.savez_compressed(
            stage_dir / "family_pixel_masks.npz",
            **{"mask_" + k: v for k, v in family_pixel_masks.items()}
        )

        # Standard outputs
        df_pixel_summary.to_csv(context.tables_dir / "06_full_day_framewise_pixel_summary.csv", index=False)
        df_fam_stats.to_csv(context.tables_dir / "07_full_day_family_pixel_statistics.csv", index=False)
        np.save(context.arrays_dir / "final_pixel_segmentation_union.npy", final_pixel_union)
        np.savez_compressed(
            context.arrays_dir / "family_pixel_masks.npz",
            **{"mask_" + k: v for k, v in family_pixel_masks.items()}
        )

        output = SegmentationOutput(
            size_gate_output=inputs,
            final_pixel_segmentation_union=final_pixel_union,
            family_pixel_masks=family_pixel_masks,
            framewise_pixel_summary_df=df_pixel_summary,
            family_pixel_statistics_df=df_fam_stats,
            total_segmented_pixels=tot_segmented_px,
        )

        return output
