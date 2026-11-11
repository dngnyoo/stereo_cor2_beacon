"""Signed Dynamic PA Patch Recovery Stage (Exp 12 v2 / Exp 13 algorithm)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import binary_propagation

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import PipelineStage, RecoveryOutput, TrackingOutput
from cme_pipeline.utils.geometry import circular_diff_deg, minimal_circular_enclosing_arc
from cme_pipeline.utils.io import compute_dict_hash


STRUCT_8 = np.ones((3, 3), bool)


def run_dynamic_patch_recovery_for_tracks(
    df_states: pd.DataFrame,
    patch_labels_all: np.ndarray,
    grid4_all: np.ndarray,
    frame_cache: List[Dict[str, Any]],
    pa_raster: np.ndarray,
    radius_raster: np.ndarray,
    patch_pa: np.ndarray,
    patch_rad: np.ndarray,
    target_track_ids: List[int],
    occulter_inner_radius_px: float = 16.326,
    radial_window_px: float = 13.35,
    pa_margin_deg: float = 5.0,
    track_ownership_all: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Canonical Exp 12 v2 / Exp 13 Signed Dynamic Patch Recovery for target tracks."""
    n_pairs = len(frame_cache)
    rec_masks = np.zeros((n_pairs, 64, 64), dtype=bool)
    dynamic_patch_history: Dict[int, Dict[str, Any]] = {}

    sub_states = df_states[
        df_states["track_id"].isin(target_track_ids) & (df_states["state"] == "OBSERVED")
    ].copy()

    for f_idx, fc in enumerate(frame_cache):
        p_idx = fc["pair_idx"]
        usable = fc["usable"]
        smoothed = fc["smoothed"]
        frac_val = fc["frac_val"]
        t24_p04 = float(fc["r_th"]["T24_p04"])

        cur_states = sub_states[sub_states["pair_index"] == p_idx]
        cur_labels = patch_labels_all[f_idx]
        cur_own = track_ownership_all[f_idx] if track_ownership_all is not None else None

        for tid in sorted(cur_states["track_id"].unique()):
            s_row = cur_states[cur_states["track_id"] == tid].iloc[0]
            if cur_own is not None:
                seed_patches = (cur_own == int(tid))
            else:
                cid = int(s_row["component_id"])
                seed_patches = (cur_labels == cid)
            if seed_patches.sum() == 0:
                continue

            # Dynamic PA corridor
            if tid in dynamic_patch_history:
                prev_info = dynamic_patch_history[tid]
                pa_s = (prev_info["pa_min"] - pa_margin_deg) % 360.0
                pa_e = (prev_info["pa_max"] + pa_margin_deg) % 360.0
            else:
                pa_s = (float(s_row["pa_min"]) - pa_margin_deg) % 360.0
                pa_e = (float(s_row["pa_max"]) + pa_margin_deg) % 360.0

            span = (pa_e - pa_s) % 360.0
            if span <= 0.0:
                span = 360.0
            c_pa = (pa_s + span / 2.0) % 360.0

            # Domain A Local Background
            sec_mask = (np.abs(circular_diff_deg(pa_raster, c_pa)) <= (span / 2.0)) & usable
            sec_vals = smoothed[sec_mask & np.isfinite(smoothed)]
            if len(sec_vals) > 1:
                l_med = float(np.median(sec_vals))
                l_mad = float(np.median(np.abs(sec_vals - l_med)))
                l_rob = float(1.4826 * l_mad) if l_mad > 0 else float(np.std(sec_vals, ddof=1))
            else:
                l_med, l_rob = 0.0, 1.0

            z_local = np.zeros_like(smoothed)
            z_local[sec_mask] = (smoothed[sec_mask] - l_med) / max(l_rob, 1e-4)

            # Radial gate (Option B)
            rf_col = "radial_front" if "radial_front" in s_row else "front_radius"
            curr_rf = float(s_row[rf_col])
            prev_s = sub_states[(sub_states["track_id"] == tid) & (sub_states["pair_index"] < p_idx)].sort_values("pair_index")
            if len(prev_s) >= 2:
                p1 = prev_s.iloc[-1]
                p0 = prev_s.iloc[-2]
                dt_obs = float(p1["pair_index"] - p0["pair_index"]) * 0.5
                v_obs = (float(p1[rf_col]) - float(p0[rf_col])) / max(dt_obs, 0.1)
                dt_curr = float(p_idx - p1["pair_index"]) * 0.5
                exp_rf = float(p1[rf_col]) + max(0.0, v_obs) * dt_curr
            elif len(prev_s) == 1:
                exp_rf = float(prev_s.iloc[-1][rf_col])
            else:
                exp_rf = curr_rf

            r_gate_max = exp_rf + radial_window_px
            r_gate_min = occulter_inner_radius_px

            p_in_gate = (
                (patch_rad >= r_gate_min) &
                (patch_rad <= r_gate_max) &
                (np.abs(circular_diff_deg(patch_pa, c_pa)) <= (span / 2.0))
            )

            # Signed condition: Z_global > 0
            z_loc_blocks = np.where(usable, np.abs(z_local), np.nan).reshape(64, 4, 64, 4).transpose(0, 2, 1, 3)
            with np.errstate(all="ignore"):
                p_act_loc = np.nanpercentile(z_loc_blocks.reshape(64, 64, 16), 95.0, axis=2)

            cand_patches = p_in_gate & (p_act_loc > t24_p04) & (frac_val >= 0.5)

            # Signed verification: block positive Z_global
            z_glob_blocks = fc["z_global"].reshape(64, 4, 64, 4).transpose(0, 2, 1, 3)
            with np.errstate(all="ignore"):
                p_glob_mean = np.nanmean(z_glob_blocks.reshape(64, 64, 16), axis=2)
            cand_patches = cand_patches & (p_glob_mean > 0.0)

            # Morphological propagation from seed
            rec_track = binary_propagation(seed_patches, mask=cand_patches | seed_patches, structure=STRUCT_8)
            rec_masks[f_idx] |= rec_track

            # Update dynamic history
            rec_sup = np.kron(rec_track, np.ones((4, 4), bool)) & usable
            if rec_sup.sum() > 0:
                y_pts, x_pts = np.nonzero(rec_sup)
                pas = pa_raster[y_pts, x_pts]
                arc = minimal_circular_enclosing_arc(pas, margin_deg=0.0)
                dynamic_patch_history[tid] = {
                    "pa_min": float(arc["arc_start"]),
                    "pa_max": float(arc["arc_end"]),
                    "c_pa": float(arc["arc_center"]),
                    "span": float(arc["arc_span"]),
                }

    return rec_masks


class DynamicPARecoveryRunner:
    """Callable recovery operator picklable by standard pickle."""

    def __init__(self, tracking_output: TrackingOutput, config: PipelineConfig):
        self.track_states_df = tracking_output.track_states_df
        self.patch_component_labels = tracking_output.patch_component_labels
        self.track_ownership_masks = getattr(tracking_output, "track_ownership_masks", None)
        if self.track_ownership_masks is None:
            self.track_ownership_masks = getattr(tracking_output, "ownership_masks", None)
        self.patch_activation_grid4 = tracking_output.patch_activation_grid4
        self.frame_cache = tracking_output.frame_cache
        self.geometry = tracking_output.geometry
        self.config = config

    def __call__(self, target_tids: List[int]) -> np.ndarray:
        return run_dynamic_patch_recovery_for_tracks(
            df_states=self.track_states_df,
            patch_labels_all=self.patch_component_labels,
            grid4_all=self.patch_activation_grid4,
            frame_cache=self.frame_cache,
            pa_raster=self.geometry.pa_raster,
            radius_raster=self.geometry.radius_raster,
            patch_pa=self.geometry.patch_pa,
            patch_rad=self.geometry.patch_rad,
            target_track_ids=target_tids,
            occulter_inner_radius_px=self.geometry.occulter_inner_radius_px,
            radial_window_px=self.config.recovery.radial_window_px,
            pa_margin_deg=self.config.recovery.pa_margin_deg,
            track_ownership_all=getattr(self, "track_ownership_masks", None),
        )


class SignedPARecoveryStage(PipelineStage):
    """Executes Signed Dynamic PA Patch Recovery."""
    name: str = "recovery"
    version: str = "1.0.0"

    def compute_fingerprint(self, inputs: TrackingOutput, config: PipelineConfig, context: PipelineContext) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": config.recovery.method,
            "radial_window_px": config.recovery.radial_window_px,
            "pa_margin_deg": config.recovery.pa_margin_deg,
            "gaussian_sigma": config.recovery.gaussian_sigma,
            "upstream_manifest_len": len(inputs.manifest_df),
            "upstream_tracking_method": getattr(inputs, "tracking_method", None),
            "upstream_total_tracks": inputs.total_tracks,
        }
        return compute_dict_hash(desc)

    def run(self, inputs: TrackingOutput, config: PipelineConfig, context: PipelineContext) -> RecoveryOutput:
        context.info("Running SignedPARecoveryStage...")

        runner = DynamicPARecoveryRunner(inputs, config)
        context.info("SignedPARecoveryStage initialized recovery runner.")

        return RecoveryOutput(
            tracking_output=inputs,
            recovery_func=runner,
            target_track_ids=None,
            recovered_patch_masks=None,
            total_recovered_patches=0,
        )
