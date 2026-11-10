"""Shared Tracking Compatibility Adapter & Downstream Interface Utilities.

Standardizes outputs across all tracking algorithms (E1, A3, A4, A5b, future A6+)
so that downstream stages (Rule B / Identity Association, Birth/Size Gates,
Signed Dynamic PA Recovery, Island-First Pixel Segmentation) operate uniformly.

Key Guarantees:
1. Track IDs (`track_ownership_masks`) and Component IDs (`patch_component_labels`)
   remain strictly separate and are NEVER conflated.
2. Predicted-only track states (`state == 'PREDICTED_ONLY'`) and decoupled predicted
   leading fronts never leak synthetic coordinates as observed patches.
3. Downstream metadata (`birth_r_min_px`, `pa_min`, `pa_max`, `radial_front`,
   `component_states_df`, `leading_front_states_df`) is consistently available.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from cme_pipeline.core.stage import TrackingOutput
from cme_pipeline.utils.geometry import circular_mean_deg, minimal_circular_enclosing_arc


def derive_ownership_masks_from_components(
    patch_component_labels: np.ndarray,
    track_states_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> np.ndarray:
    """Derives mutually exclusive `(N, 64, 64)` track ownership masks for 1:1 trackers (e.g. E1)."""
    own_masks = np.zeros_like(patch_component_labels, dtype=np.int32)
    if track_states_df.empty:
        return own_masks

    pair_to_fidx = {int(r["pair_index"]): idx for idx, r in manifest_df.reset_index(drop=True).iterrows()}
    obs_df = track_states_df[track_states_df["state"] == "OBSERVED"]

    for _, row in obs_df.iterrows():
        p_idx = int(row["pair_index"])
        f_idx = pair_to_fidx.get(p_idx)
        if f_idx is None:
            continue
        tid = int(row["track_id"])
        cid = int(row["component_id"])
        mask = (patch_component_labels[f_idx] == cid)
        if mask.any():
            existing = own_masks[f_idx][mask]
            if np.any((existing != 0) & (existing != tid)):
                raise ValueError(
                    f"Track ownership collision at pair {p_idx} between track {tid} and {np.unique(existing[existing != 0])}"
                )
            own_masks[f_idx][mask] = tid

    return own_masks


def get_track_observed_mask(
    tracking_output: TrackingOutput,
    f_idx: int,
    track_id: int,
    fallback_component_id: Optional[int] = None,
) -> np.ndarray:
    """Canonical helper returning all active patches owned by `track_id` in frame `f_idx`.

    Always uses `track_ownership_masks` (which cleanly handles single-component,
    multi-component, and split-contested-component tracks) when available.
    Falls back to `patch_component_labels == fallback_component_id` only for
    legacy unadapted E1 pickles.
    """
    own_masks = getattr(tracking_output, "track_ownership_masks", None)
    if own_masks is None:
        own_masks = getattr(tracking_output, "ownership_masks", None)
    if own_masks is not None:
        return (own_masks[f_idx] == int(track_id))

    if fallback_component_id is not None and int(fallback_component_id) >= 0:
        return (tracking_output.patch_component_labels[f_idx] == int(fallback_component_id))
    return np.zeros(tracking_output.patch_component_labels.shape[1:], dtype=bool)


def get_track_family_observed_mask(
    tracking_output: TrackingOutput,
    f_idx: int,
    track_ids: Sequence[int],
    fallback_component_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Canonical helper returning observed patches owned by any track in `track_ids`."""
    own_masks = getattr(tracking_output, "track_ownership_masks", None)
    if own_masks is None:
        own_masks = getattr(tracking_output, "ownership_masks", None)
    if own_masks is not None:
        return np.isin(own_masks[f_idx], list(track_ids))

    if fallback_component_ids is not None:
        return np.isin(tracking_output.patch_component_labels[f_idx], list(fallback_component_ids))
    return np.zeros(tracking_output.patch_component_labels.shape[1:], dtype=bool)


def adapt_tracking_output(
    raw_output: TrackingOutput,
    max_birth_radius_px: float = 30.0,
) -> TrackingOutput:
    """Standardizes a TrackingOutput instance to fulfill full downstream contracts cleanly.

    Works uniformly for E1, A3, A4, A5b, and future methods without modifying arrays
    or conflating Component IDs with Track IDs.
    """
    adapted = copy.copy(raw_output)
    man_df = raw_output.manifest_df.reset_index(drop=True)
    geom = raw_output.geometry
    patch_pa = geom.patch_pa
    patch_rad = geom.patch_rad
    pair_to_fidx = {int(r["pair_index"]): idx for idx, r in man_df.iterrows()}

    # 1. Ensure track_ownership_masks is populated
    own_masks = getattr(raw_output, "track_ownership_masks", None)
    if own_masks is None:
        own_masks = getattr(raw_output, "ownership_masks", None)
    if own_masks is None:
        own_masks = derive_ownership_masks_from_components(
            raw_output.patch_component_labels,
            raw_output.track_states_df,
            man_df,
        )
    adapted.track_ownership_masks = own_masks

    # 2. Standardize track_states_df columns cleanly without conflating IDs
    df = raw_output.track_states_df.copy()
    if df.empty:
        adapted.track_states_df = df
        return adapted

    tracks_dict = getattr(raw_output, "adaptive_tracks", {})

    # birth_r_min_px
    if "birth_r_min_px" not in df.columns:
        if tracks_dict:
            birth_r_map = {int(tid): float(t.birth_r_min_px) for tid, t in tracks_dict.items()}
            df["birth_r_min_px"] = df["track_id"].map(birth_r_map)
        else:
            # Compute from first observed mask of each track
            birth_r_map: Dict[int, float] = {}
            obs_sorted = df[df["state"] == "OBSERVED"].sort_values(["track_id", "pair_index"])
            for tid, grp in obs_sorted.groupby("track_id"):
                first_p = int(grp.iloc[0]["pair_index"])
                f_idx = pair_to_fidx[first_p]
                m = (own_masks[f_idx] == int(tid))
                birth_r_map[int(tid)] = float(np.min(patch_rad[m])) if m.any() else np.nan
            df["birth_r_min_px"] = df["track_id"].map(birth_r_map)

    # is_solar_root_anchor
    if "is_solar_root_anchor" not in df.columns:
        if tracks_dict:
            anc_map = {int(tid): bool(t.is_anchor) for tid, t in tracks_dict.items()}
            df["is_solar_root_anchor"] = df["track_id"].map(anc_map)
        else:
            df["is_solar_root_anchor"] = df["birth_r_min_px"] <= max_birth_radius_px

    # original_birth_cpa / original_birth_pa
    if "original_birth_cpa" not in df.columns:
        if tracks_dict:
            df["original_birth_cpa"] = df["track_id"].map(
                {int(tid): float(t.original_birth_cpa) for tid, t in tracks_dict.items()}
            )
        else:
            birth_cpa_map: Dict[int, float] = {}
            obs_sorted = df[df["state"] == "OBSERVED"].sort_values(["track_id", "pair_index"])
            for tid, grp in obs_sorted.groupby("track_id"):
                birth_cpa_map[int(tid)] = float(grp.iloc[0]["central_pa"])
            df["original_birth_cpa"] = df["track_id"].map(birth_cpa_map)

    if "original_birth_pa" not in df.columns and tracks_dict:
        df["original_birth_pa"] = df["track_id"].map(
            {int(tid): float(t.original_birth_pa) for tid, t in tracks_dict.items()}
        )

    # tracked_cpa
    if "tracked_cpa" not in df.columns:
        df["tracked_cpa"] = df["central_pa"]

    # radial_front <-> front_radius harmonization
    if "radial_front" not in df.columns and "front_radius" in df.columns:
        df["radial_front"] = df["front_radius"]
    elif "front_radius" not in df.columns and "radial_front" in df.columns:
        df["front_radius"] = df["radial_front"]

    # leading_front_state & leading_front_comp_id
    if "leading_front_state" not in df.columns:
        df["leading_front_state"] = np.where(df["state"] == "OBSERVED", "OBSERVED", "PREDICTED")
    if "leading_front_comp_id" not in df.columns:
        if "component_id" in df.columns:
            df["leading_front_comp_id"] = df["component_id"].fillna(-1).astype(int)
        else:
            df["leading_front_comp_id"] = -1

    # num_components
    if "num_components" not in df.columns:
        df["num_components"] = np.where(df["state"] == "OBSERVED", 1, 0)

    # pa_min and pa_max strictly computed from owned patches for OBSERVED states
    if "pa_min" not in df.columns or "pa_max" not in df.columns:
        pa_mins: List[float] = []
        pa_maxs: List[float] = []
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

    # 3. Populate component_states_df and leading_front_states_df if missing (E1 / A3)
    if getattr(raw_output, "component_states_df", None) is None:
        comp_rows = []
        lbls_all = raw_output.patch_component_labels
        for _, row in df[df["state"] == "OBSERVED"].iterrows():
            p_idx = int(row["pair_index"])
            tid = int(row["track_id"])
            f_idx = pair_to_fidx.get(p_idx)
            if f_idx is None:
                continue
            cur_mask = (own_masks[f_idx] == tid)
            if not cur_mask.any():
                continue
            cids_in_track = np.unique(lbls_all[f_idx][cur_mask])
            cids_in_track = [int(c) for c in cids_in_track if c > 0]
            lf_cid = int(row.get("leading_front_comp_id", -1))
            for cid in cids_in_track:
                cm = cur_mask & (lbls_all[f_idx] == cid)
                rads = patch_rad[cm]
                pas = patch_pa[cm]
                comp_rows.append({
                    "pair_index": p_idx,
                    "timestamp": str(row["timestamp"]),
                    "track_id": tid,
                    "component_id": cid,
                    "num_patches": int(cm.sum()),
                    "cpa": float(circular_mean_deg(pas)),
                    "r_min": float(np.min(rads)),
                    "r_front": float(np.max(rads)),
                    "is_leading_front": (cid == lf_cid) or (len(cids_in_track) == 1),
                })
        adapted.component_states_df = pd.DataFrame(comp_rows)

    if getattr(raw_output, "leading_front_states_df", None) is None:
        lf_rows = []
        for _, row in df.iterrows():
            lf_rows.append({
                "pair_index": int(row["pair_index"]),
                "timestamp": str(row["timestamp"]),
                "track_id": int(row["track_id"]),
                "leading_front_state": str(row["leading_front_state"]),
                "leading_front_radius": float(row["front_radius"]),
                "responsible_comp_id": int(row["leading_front_comp_id"]),
                "radial_displacement": float(row.get("radial_displacement", 0.0)),
            })
        adapted.leading_front_states_df = pd.DataFrame(lf_rows)

    return adapted

