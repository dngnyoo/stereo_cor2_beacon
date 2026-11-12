"""Birth-Anchored Directional Tracking Stage (Variant C).

Maintains CME directional identity causally locked from the first two consecutive
observations near the solar surface (r_min <= 30 px).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.direction import (
    compute_solar_root_pa,
    create_birth_direction_reference,
    update_birth_direction_reference,
)
from cme_pipeline.core.stage import (
    BirthDirectionReference,
    PipelineStage,
    TrackingOutput,
)
from cme_pipeline.stages.tracking.v01_baseline_tracking import E1TrackingStage
from cme_pipeline.utils.geometry import (
    circular_diff_deg,
    circular_mean_deg,
    circular_sector_overlap,
)
from cme_pipeline.utils.io import compute_dict_hash


# Canonical E1 constants
MAX_CADENCE_GAP_SEC = 4500.0  # 75 min
BIRTH_GATE_THRESHOLD_PX = 30.0
PA_ASSOC_MAX_DEG = 30.0
RADIAL_ASSOC_MAX_PX = 25.0
ASSOCIATION_COST_MAX = 5.0
DIRECTION_THRESHOLD_DEG = 30.0


def interval_overlap_fraction(center_a: float, offsets_a: np.ndarray,
                              center_b: float, offsets_b: np.ndarray) -> float:
    """Fraction of circular interval overlap on S^1."""
    if len(offsets_a) == 0 or len(offsets_b) == 0:
        return 0.0
    min_a, max_a = float(np.min(offsets_a)), float(np.max(offsets_a))
    min_b, max_b = float(np.min(offsets_b)), float(np.max(offsets_b))
    shift = float(circular_diff_deg(center_b, center_a))
    min_b_shifted, max_b_shifted = min_b + shift, max_b + shift

    inter_min = max(min_a, min_b_shifted)
    inter_max = min(max_a, max_b_shifted)
    if inter_max <= inter_min:
        return 0.0
    overlap_len = inter_max - inter_min
    union_len = max(max_a, max_b_shifted) - min(min_a, min_b_shifted)
    return float(overlap_len / union_len) if union_len > 0 else 0.0


def predict_radial_front(history: List[Dict[str, Any]], target_time: datetime) -> Optional[float]:
    """E1 two-observation radial velocity extrapolation."""
    evidence = [h for h in history if h.get("state") != "PREDICTED_ONLY"]
    if len(evidence) < 2:
        return None
    a, b = evidence[-2], evidence[-1]
    dt_hist = (b["timestamp"] - a["timestamp"]).total_seconds()
    dt_future = (target_time - b["timestamp"]).total_seconds()
    if dt_hist <= 0 or dt_future <= 0:
        return None
    velocity = (b["front_radius"] - a["front_radius"]) / dt_hist
    outward = max(0.0, velocity)
    return float(b["front_radius"] + outward * dt_future)


@dataclass
class DirectionalTrack:
    track_id: int
    birth_r_min: float
    birth_proximity: str
    birth_ref: BirthDirectionReference
    history: List[Dict[str, Any]] = field(default_factory=list)
    missed: int = 0
    terminated: bool = False


class BirthAnchoredTrackingStage(PipelineStage):
    """Executes Birth-Anchored Directional Tracking with two-frame causal direction locking."""
    name: str = "tracking"
    version: str = "2.0.0"

    def compute_fingerprint(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "method": config.tracking.method,
            "direction_threshold_deg": config.tracking.direction_threshold_deg,
            "max_lock_deviation_deg": config.tracking.max_direction_deviation_deg,
            "date_str": config.date_str,
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: Any, config: PipelineConfig, context: PipelineContext
    ) -> TrackingOutput:
        context.info("Running BirthAnchoredTrackingStage (Variant C)...")

        # Reuse E1TrackingStage for WCS, frame cache, geometry, and component rasters
        base_stage = E1TrackingStage()
        base_out = base_stage.run(inputs, config, context)

        val_root = context.repo_root / "validation/frozen/20101212"
        exp11_dir = val_root / "experiments/11_d7_patch_first_tracking_feasibility"
        df_geom = pd.read_csv(exp11_dir / "results/tables/01_patch_component_geometries.csv")

        man_df = base_out.manifest_df
        n_pairs = len(man_df)
        dir_threshold = config.tracking.direction_threshold_deg
        max_lock_dev = config.tracking.max_direction_deviation_deg

        # Group components by pair_index
        pair_components: Dict[int, List[Dict[str, Any]]] = {}
        for p_idx in man_df["pair_index"].unique():
            sub = df_geom[df_geom["pair_index"] == p_idx].copy()
            pair_components[p_idx] = sub.to_dict("records")

        tracks: Dict[int, DirectionalTrack] = {}
        next_track_id = 1
        track_state_rows = []
        candidate_eval_rows = []

        for step_idx in range(n_pairs):
            p_curr = int(man_df.iloc[step_idx]["pair_index"])
            ts_curr = datetime.fromisoformat(man_df.iloc[step_idx]["end_time"])
            comps_curr = pair_components.get(p_curr, [])

            # Frame 0: Spawn initial tracks with PROVISIONAL references
            if step_idx == 0:
                for c in comps_curr:
                    tid = next_track_id
                    next_track_id += 1
                    b_rmin = float(c["r_min_px"])
                    b_prox = "NEAR" if b_rmin <= BIRTH_GATE_THRESHOLD_PX else "FAR"

                    b_ref = create_birth_direction_reference(
                        track_id=tid,
                        birth_pair_id=p_curr,
                        birth_time=ts_curr,
                        birth_r_min_px=b_rmin,
                        central_pa=float(c["central_pa_deg"]),
                        angular_extent=float(c["pa_span_deg"]),
                        birth_pa=float(c["central_pa_deg"]),
                        anchor_threshold_px=BIRTH_GATE_THRESHOLD_PX,
                    )

                    t = DirectionalTrack(
                        track_id=tid,
                        birth_r_min=b_rmin,
                        birth_proximity=b_prox,
                        birth_ref=b_ref,
                    )
                    st = {
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": int(c["component_id"]),
                        "component_uid": c["component_uid"],
                        "state": "OBSERVED",
                        "central_pa": float(c["central_pa_deg"]),
                        "pa_min": float(c["pa_min_deg"]),
                        "pa_max": float(c["pa_max_deg"]),
                        "angular_width": float(c["pa_span_deg"]),
                        "front_radius": float(c["r_front_px"]),
                        "area_pixels": int(c["area_pixels"]),
                        "num_patches": int(c["num_patches"]),
                        "centroid_x": float(c["centroid_x"]),
                        "centroid_y": float(c["centroid_y"]),
                        "association_cost": 0.0,
                        "reference_cpa": b_ref.reference_cpa,
                        "reference_status": b_ref.status,
                        "delta_reference_pa": 0.0,
                    }
                    t.history.append(st)
                    tracks[tid] = t

                    track_state_rows.append({
                        "experiment": "Birth_Anchored_Track",
                        "track_id": tid,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "component_id": int(c["component_id"]),
                        "component_uid": c["component_uid"],
                        "state": "OBSERVED",
                        "central_pa": float(c["central_pa_deg"]),
                        "pa_min": float(c["pa_min_deg"]),
                        "pa_max": float(c["pa_max_deg"]),
                        "radial_front": float(c["r_front_px"]),
                        "front_radius": float(c["r_front_px"]),
                        "area_pixels": int(c["area_pixels"]),
                        "num_patches": int(c["num_patches"]),
                        "centroid_x": float(c["centroid_x"]),
                        "centroid_y": float(c["centroid_y"]),
                        "association_cost": 0.0,
                        "birth_r_min_px": b_rmin,
                        "birth_proximity": b_prox,
                        "reference_cpa": b_ref.reference_cpa,
                        "reference_status": b_ref.status,
                        "delta_reference_pa": 0.0,
                    })
                continue

            # Transitions: Frame step_idx - 1 -> step_idx
            p_prev = int(man_df.iloc[step_idx - 1]["pair_index"])
            ts_prev = datetime.fromisoformat(man_df.iloc[step_idx - 1]["end_time"])
            dt_sec = (ts_curr - ts_prev).total_seconds()
            is_consecutive = dt_sec <= MAX_CADENCE_GAP_SEC

            # Gap handling (> 75 min)
            if not is_consecutive:
                for t in tracks.values():
                    if not t.terminated:
                        t.terminated = True

            active_tracks = [t for t in tracks.values() if not t.terminated]
            pair_evals = []

            for t in active_tracks:
                last_obs = next((h for h in reversed(t.history) if h["state"] != "PREDICTED_ONLY"), None)
                if last_obs is None:
                    continue

                observed_in_prev_frame = (t.history[-1]["state"] == "OBSERVED")
                is_near = (t.birth_r_min <= BIRTH_GATE_THRESHOLD_PX)

                if observed_in_prev_frame and is_near:
                    priority_group = 1
                elif observed_in_prev_frame and not is_near:
                    priority_group = 2
                elif (not observed_in_prev_frame) and is_near:
                    priority_group = 3
                else:
                    priority_group = 4

                pred_front = predict_radial_front(t.history, ts_curr)
                expected_r = pred_front if pred_front is not None else float(last_obs["front_radius"])

                ref = t.birth_ref
                ref_pa = ref.reference_cpa if ref.reference_cpa is not None else float(last_obs["central_pa"])

                for c in comps_curr:
                    cand_cpa = float(c["central_pa_deg"])
                    cand_rf = float(c["r_front_px"])
                    cand_w = float(c["pa_span_deg"])

                    pa_diff_prev = abs(float(circular_diff_deg(cand_cpa, float(last_obs["central_pa"]))))
                    delta_ref = abs(float(circular_diff_deg(cand_cpa, ref_pa)))
                    radial_diff = abs(cand_rf - expected_r)

                    # Circular interval overlap
                    ov = circular_sector_overlap(
                        float(last_obs["central_pa"]), float(last_obs["angular_width"]),
                        cand_cpa, cand_w,
                    )
                    overlap = ov["cov_cand"]

                    rejection = "NONE"
                    cost = np.inf

                    # Directional constraint:
                    # If locked and deviation > 30 deg with zero angular overlap -> reject
                    if ref.status == "LOCKED" and delta_ref > dir_threshold and overlap == 0.0:
                        rejection = "DIRECTION_MISMATCH_EXCEEDS_30DEG"
                    elif (pa_diff_prev > PA_ASSOC_MAX_DEG and overlap == 0.0) or radial_diff > RADIAL_ASSOC_MAX_PX:
                        rejection = "INFINITE_COST"
                    else:
                        # Baseline cost + modest directional penalty
                        cost = (
                            pa_diff_prev / 10.0 +
                            radial_diff / 8.0 +
                            (1.0 - overlap) +
                            0.1 * (delta_ref / dir_threshold)
                        )
                        if cost > ASSOCIATION_COST_MAX:
                            rejection = "COST_EXCEEDS_MAX"

                    pair_evals.append({
                        "track_id": t.track_id,
                        "track_obj": t,
                        "cand_comp": c,
                        "cand_uid": c["component_uid"],
                        "priority_group": priority_group,
                        "cost": cost,
                        "rejection_reason": rejection,
                        "delta_ref_pa": delta_ref,
                        "selected": False,
                    })

            # 4-Tier Priority Greedy 1:1 Matching
            pair_evals.sort(key=lambda x: (x["priority_group"], x["cost"]))
            used_tracks: Set[int] = set()
            used_comps: Set[str] = set()
            matched_evals = []

            for ev in pair_evals:
                tid = ev["track_id"]
                c_uid = ev["cand_uid"]
                if not np.isfinite(ev["cost"]) or ev["cost"] > ASSOCIATION_COST_MAX:
                    continue
                if tid in used_tracks:
                    if ev["rejection_reason"] == "NONE":
                        ev["rejection_reason"] = "TRACK_ALREADY_MATCHED"
                    continue
                if c_uid in used_comps:
                    if ev["rejection_reason"] == "NONE":
                        ev["rejection_reason"] = "CANDIDATE_ALREADY_MATCHED"
                    continue

                ev["selected"] = True
                ev["rejection_reason"] = "SELECTED"
                used_tracks.add(tid)
                used_comps.add(c_uid)
                matched_evals.append(ev)

            # Apply matched records
            for m in matched_evals:
                t = m["track_obj"]
                c = m["cand_comp"]
                cand_cpa = float(c["central_pa_deg"])

                # Causal reference locking at second observation
                obs_history = [h for h in t.history if h["state"] == "OBSERVED"]
                if len(obs_history) == 1 and t.birth_ref.status == "PROVISIONAL":
                    t.birth_ref = update_birth_direction_reference(
                        ref=t.birth_ref,
                        current_pair_id=p_curr,
                        current_time=ts_curr,
                        current_cpa=cand_cpa,
                        is_consecutive=is_consecutive,
                        max_lock_deviation_deg=max_lock_dev,
                    )

                delta_ref = abs(float(circular_diff_deg(cand_cpa, t.birth_ref.reference_cpa)))
                st = {
                    "pair_index": p_curr,
                    "timestamp": ts_curr,
                    "track_id": t.track_id,
                    "component_id": int(c["component_id"]),
                    "component_uid": c["component_uid"],
                    "state": "OBSERVED",
                    "central_pa": cand_cpa,
                    "pa_min": float(c["pa_min_deg"]),
                    "pa_max": float(c["pa_max_deg"]),
                    "angular_width": float(c["pa_span_deg"]),
                    "front_radius": float(c["r_front_px"]),
                    "area_pixels": int(c["area_pixels"]),
                    "num_patches": int(c["num_patches"]),
                    "centroid_x": float(c["centroid_x"]),
                    "centroid_y": float(c["centroid_y"]),
                    "association_cost": m["cost"],
                    "reference_cpa": t.birth_ref.reference_cpa,
                    "reference_status": t.birth_ref.status,
                    "delta_reference_pa": delta_ref,
                }
                t.history.append(st)
                t.missed = 0

                track_state_rows.append({
                    "experiment": "Birth_Anchored_Track",
                    "track_id": t.track_id,
                    "pair_index": p_curr,
                    "timestamp": ts_curr.isoformat(),
                    "component_id": int(c["component_id"]),
                    "component_uid": c["component_uid"],
                    "state": "OBSERVED",
                    "central_pa": cand_cpa,
                    "pa_min": float(c["pa_min_deg"]),
                    "pa_max": float(c["pa_max_deg"]),
                    "radial_front": float(c["r_front_px"]),
                    "front_radius": float(c["r_front_px"]),
                    "area_pixels": int(c["area_pixels"]),
                    "num_patches": int(c["num_patches"]),
                    "centroid_x": float(c["centroid_x"]),
                    "centroid_y": float(c["centroid_y"]),
                    "association_cost": m["cost"],
                    "birth_r_min_px": t.birth_r_min,
                    "birth_proximity": t.birth_proximity,
                    "reference_cpa": t.birth_ref.reference_cpa,
                    "reference_status": t.birth_ref.status,
                    "delta_reference_pa": delta_ref,
                })

            # Handle unmatched active tracks (missed count)
            for t in active_tracks:
                if t.track_id not in used_tracks:
                    t.missed += 1
                    if t.missed >= 2:
                        t.terminated = True

            # Spawn new tracks for unmatched components in current frame
            for c in comps_curr:
                if c["component_uid"] not in used_comps:
                    tid = next_track_id
                    next_track_id += 1
                    b_rmin = float(c["r_min_px"])
                    b_prox = "NEAR" if b_rmin <= BIRTH_GATE_THRESHOLD_PX else "FAR"

                    b_ref = create_birth_direction_reference(
                        track_id=tid,
                        birth_pair_id=p_curr,
                        birth_time=ts_curr,
                        birth_r_min_px=b_rmin,
                        central_pa=float(c["central_pa_deg"]),
                        angular_extent=float(c["pa_span_deg"]),
                        birth_pa=float(c["central_pa_deg"]),
                        anchor_threshold_px=BIRTH_GATE_THRESHOLD_PX,
                    )

                    t = DirectionalTrack(
                        track_id=tid,
                        birth_r_min=b_rmin,
                        birth_proximity=b_prox,
                        birth_ref=b_ref,
                    )
                    st = {
                        "pair_index": p_curr,
                        "timestamp": ts_curr,
                        "track_id": tid,
                        "component_id": int(c["component_id"]),
                        "component_uid": c["component_uid"],
                        "state": "OBSERVED",
                        "central_pa": float(c["central_pa_deg"]),
                        "pa_min": float(c["pa_min_deg"]),
                        "pa_max": float(c["pa_max_deg"]),
                        "angular_width": float(c["pa_span_deg"]),
                        "front_radius": float(c["r_front_px"]),
                        "area_pixels": int(c["area_pixels"]),
                        "num_patches": int(c["num_patches"]),
                        "centroid_x": float(c["centroid_x"]),
                        "centroid_y": float(c["centroid_y"]),
                        "association_cost": 0.0,
                        "reference_cpa": b_ref.reference_cpa,
                        "reference_status": b_ref.status,
                        "delta_reference_pa": 0.0,
                    }
                    t.history.append(st)
                    tracks[tid] = t

                    track_state_rows.append({
                        "experiment": "Birth_Anchored_Track",
                        "track_id": tid,
                        "pair_index": p_curr,
                        "timestamp": ts_curr.isoformat(),
                        "component_id": int(c["component_id"]),
                        "component_uid": c["component_uid"],
                        "state": "OBSERVED",
                        "central_pa": float(c["central_pa_deg"]),
                        "pa_min": float(c["pa_min_deg"]),
                        "pa_max": float(c["pa_max_deg"]),
                        "radial_front": float(c["r_front_px"]),
                        "front_radius": float(c["r_front_px"]),
                        "area_pixels": int(c["area_pixels"]),
                        "num_patches": int(c["num_patches"]),
                        "centroid_x": float(c["centroid_x"]),
                        "centroid_y": float(c["centroid_y"]),
                        "association_cost": 0.0,
                        "birth_r_min_px": b_rmin,
                        "birth_proximity": b_prox,
                        "reference_cpa": b_ref.reference_cpa,
                        "reference_status": b_ref.status,
                        "delta_reference_pa": 0.0,
                    })

        df_track_states = pd.DataFrame(track_state_rows)
        birth_refs = {t.track_id: t.birth_ref for t in tracks.values()}
        locked_count = sum(1 for r in birth_refs.values() if r.status == "LOCKED")
        anchor_tracks = sorted([tid for tid, r in birth_refs.items() if r.is_anchor])

        context.info(
            f"BirthAnchoredTracking complete: {len(tracks)} total tracks, "
            f"{len(anchor_tracks)} anchors, {locked_count} locked directional references."
        )

        # Save stage outputs
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_track_states.to_csv(stage_dir / "birth_anchored_track_states.csv", index=False)

        ref_rows = []
        for r in birth_refs.values():
            ref_rows.append({
                "track_id": r.track_id,
                "birth_pair_id": r.birth_pair_id,
                "birth_time": r.birth_time.isoformat(),
                "birth_r_min_px": r.birth_r_min_px,
                "is_anchor": r.is_anchor,
                "birth_pa": r.birth_pa,
                "cpa_first": r.cpa_first,
                "cpa_second": r.cpa_second,
                "reference_cpa": r.reference_cpa,
                "initial_angular_extent": r.initial_angular_extent,
                "status": r.status,
                "confirmation_time": r.confirmation_time.isoformat() if r.confirmation_time else None,
                "confidence": r.confidence,
                "flags": ";".join(r.flags),
            })
        df_refs = pd.DataFrame(ref_rows)
        df_refs.to_csv(stage_dir / "birth_direction_references.csv", index=False)
        df_refs.to_csv(context.tables_dir / "00_birth_direction_references.csv", index=False)

        return TrackingOutput(
            manifest_df=base_out.manifest_df,
            track_states_df=df_track_states,
            patch_component_labels=base_out.patch_component_labels,
            patch_activation_grid4=base_out.patch_activation_grid4,
            frame_cache=base_out.frame_cache,
            geometry=base_out.geometry,
            total_tracks=len(tracks),
            anchor_tracks=anchor_tracks,
            birth_direction_references=birth_refs,
        )

