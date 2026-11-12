#!/usr/bin/env python3
"""Recompute frozen E1 tracking from NEW C0 outputs; compare independently to Frozen.

Historical Experiment 11 E1 matching logic has been preserved verbatim.
Only inputs, paths, and output logic are adapted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import argparse

# Allow direct execution: python scripts/recompute_e1_tracking.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS
from cme_pipeline.utils.geometry import circular_diff_deg, minimal_circular_enclosing_arc
IMAGE_SHAPE = (256,256)
P04 = 4
RSUN_PX = 8.163
PIXEL_TO_RSUN = 1.0 / RSUN_PX
KM_PER_PIXEL = 696340.0 / RSUN_PX
PA_ASSOC_MAX_DEG = 30.0
RADIAL_ASSOC_MAX_PX = 40.0
ASSOCIATION_COST_MAX = 4.0
PATIENCE_FRAMES = 1
FRONT_RADIUS_PERCENTILE = 95.0
CONTOUR_BIN_DEG = 2.0
BIRTH_GATE_THRESHOLD_PX = 30.0
MAX_CADENCE_GAP_SEC = 4500.0
def circular_mean_deg(angles, weights=None):
    rads = np.radians(angles)
    if weights is not None:
        sin_sum = np.sum(np.sin(rads) * weights)
        cos_sum = np.sum(np.cos(rads) * weights)
    else:
        sin_sum = np.sum(np.sin(rads))
        cos_sum = np.sum(np.cos(rads))
    return float(np.degrees(np.arctan2(sin_sum, cos_sum)) % 360.0)

def interval_overlap_fraction(center_a, offsets_a, center_b, offsets_b):
    bins = np.arange(360.0)
    ma = ((circular_diff_deg(bins, center_a) >= np.min(offsets_a)) &
          (circular_diff_deg(bins, center_a) <= np.max(offsets_a)))
    mb = ((circular_diff_deg(bins, center_b) >= np.min(offsets_b)) &
          (circular_diff_deg(bins, center_b) <= np.max(offsets_b)))
    union = int(np.sum(ma | mb))
    return float(np.sum(ma & mb) / union) if union else 0.0

def contour_from_pixels(pa_values, radius_values, center, bin_deg=2.0, pct=95.0):
    offsets = circular_diff_deg(pa_values, center)
    if not len(offsets):
        return np.asarray([], float), np.asarray([], float)
    lo = np.floor(offsets.min() / bin_deg) * bin_deg
    hi = np.ceil(offsets.max() / bin_deg) * bin_deg
    centers, values = [], []
    edge = lo
    while edge <= hi + 1e-9:
        use = ((offsets >= edge - bin_deg / 2) &
               (offsets < edge + bin_deg / 2))
        if np.any(use):
            centers.append(float(edge))
            values.append(float(np.percentile(radius_values[use], pct)))
        edge += bin_deg
    return np.asarray(centers, float), np.asarray(values, float)

@dataclass
class PatchComponent:
    component_uid: str
    pair_index: int
    timestamp: datetime
    component_id: int
    area_pixels: int
    num_patches: int
    centroid_x: float
    centroid_y: float
    central_pa: float
    pa_offsets: np.ndarray
    pa_min: float
    pa_max: float
    pa_width: float
    front_radius: float
    r_min: float
    r_max: float
    birth_r_min: float
    contour_offsets: np.ndarray
    contour_radii: np.ndarray
    max_Z: float
    mean_Z: float
    bbox_xmin: int
    bbox_ymin: int
    bbox_xmax: int
    bbox_ymax: int


@dataclass
class PatchMotionTrack:
    track_id: int
    birth_r_min: float
    birth_proximity: str  # "NEAR" or "FAR"
    history: list = field(default_factory=list)
    missed: int = 0
    terminated: bool = False


def history_state_from_patch(patch: PatchComponent, timestamp: datetime, state: str):
    return {
        "timestamp": timestamp,
        "state": state,
        "central_pa": patch.central_pa,
        "pa_min": patch.pa_min,
        "pa_max": patch.pa_max,
        "angular_width": patch.pa_width,
        "front_radius": patch.front_radius,
        "area_pixels": patch.area_pixels,
        "num_patches": patch.num_patches,
        "centroid_x": patch.centroid_x,
        "centroid_y": patch.centroid_y,
        "pa_offsets": patch.pa_offsets.copy(),
        "contour_offsets": patch.contour_offsets.copy(),
        "contour_radii": patch.contour_radii.copy(),
        "component_id": patch.component_id,
        "component_uid": patch.component_uid,
        "pair_index": patch.pair_index,
    }

def predict_contour(history: list, target_time: datetime):
    """E1 two-observation radial velocity extrapolation predictor."""
    evidence = [item for item in history if item["state"] != "PREDICTED_ONLY"]
    if len(evidence) < 2:
        return None
    a, b = evidence[-2], evidence[-1]
    dt_hist = (b["timestamp"] - a["timestamp"]).total_seconds()
    dt_future = (target_time - b["timestamp"]).total_seconds()
    if dt_hist <= 0 or dt_future <= 0:
        return None
    velocity = (b["front_radius"] - a["front_radius"]) / dt_hist
    outward = max(0.0, velocity)
    displacement = outward * dt_future
    return {
        "central_pa": b["central_pa"],
        "angular_width": b["angular_width"],
        "base_contour_offsets": np.asarray(b["contour_offsets"], float),
        "base_contour_radii": np.asarray(b["contour_radii"], float),
        "predicted_contour_radii": np.asarray(b["contour_radii"], float) + displacement,
        "predicted_front_radius": b["front_radius"] + displacement,
        "velocity": velocity,
        "outward_velocity": outward,
        "delta_t_seconds": dt_future,
        "displacement_px": displacement,
    }

def patch_association_cost(track: PatchMotionTrack, patch: PatchComponent, timestamp: datetime):
    last = next((item for item in reversed(track.history) if item["state"] != "PREDICTED_ONLY"), None)
    if last is None:
        return np.inf, 0.0, np.nan, np.nan

    pa_diff = abs(float(circular_diff_deg(patch.central_pa, last["central_pa"])))
    overlap = interval_overlap_fraction(last["central_pa"], last["contour_offsets"],
                                        patch.central_pa, patch.pa_offsets)

    prediction = predict_contour(track.history, timestamp)
    expected = prediction["predicted_front_radius"] if prediction else last["front_radius"]
    radial_diff = abs(patch.front_radius - expected)

    if (pa_diff > PA_ASSOC_MAX_DEG and overlap == 0.0) or radial_diff > RADIAL_ASSOC_MAX_PX:
        return np.inf, overlap, pa_diff, radial_diff

    cost = pa_diff / 10.0 + radial_diff / 8.0 + (1.0 - overlap)
    return float(cost), overlap, pa_diff, radial_diff


def run(detection_dir: Path, repo: Path, output_dir: Path, *,
        compare_frozen: bool = True, historical_geometry: bool = True):
    detection_dir, repo, output_dir = detection_dir.resolve(), repo.resolve(), output_dir.resolve()
    metadata = json.loads((detection_dir / "detection_metadata.json").read_text())
    pre = Path(metadata["input_dir"])
    df_man = pd.read_csv(pre / "01_rd_manifest.csv")
    df_det = pd.read_csv(detection_dir / "frame_detection_summary.csv")
    rd_all = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    val_all = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    grid_all = np.load(detection_dir / "patch_state_p04.npy", mmap_mode="r")
    label_all = np.load(detection_dir / "patch_component_labels.npy", mmap_mode="r")
    if len(df_man) != len(df_det) or len(df_man) != len(rd_all) or len(df_man) != len(grid_all):
        raise ValueError("Input lengths mismatch")
    eval_indices = np.flatnonzero((df_det["in_requested_date"].astype(str).str.lower() == "true") & (df_det["status"] == "VALID"))
    if not len(eval_indices): raise ValueError("No VALID evaluation observations")
    hdr = fits.Header.fromstring((pre / "target_wcs_header.txt").read_text(), sep="\n")
    wcs = WCS(hdr)
    # Exact Frozen E1 geometry center (not WCS-derived), required for benchmark consistency.
    SX, SY = (126.618, 127.503) if historical_geometry else tuple(
        float(v) for v in wcs.world_to_pixel_values(0.0, 0.0))
    yy, xx = np.indices(IMAGE_SHAPE, dtype=float)
    world = wcs.pixel_to_world_values(xx, yy)
    lon = (np.asarray(world[0], float) + 180.0) % 360.0 - 180.0
    lat = np.asarray(world[1], float)
    pa_raster = np.degrees(np.arctan2(-lon, lat)) % 360.0
    radius_raster = np.hypot(xx - SX, yy - SY)
    pairs_data = []
    comp_geom_rows = []
    for p_idx in eval_indices:
        r_man = df_man.iloc[p_idx]
        pair_num = int(r_man["pair_index"])
        ts = datetime.fromisoformat(r_man["end_time"].replace('Z','+00:00'))
        rd = np.asarray(rd_all[p_idx], dtype=np.float32)
        val = np.asarray(val_all[p_idx], dtype=bool)
        # Match original E1 float32 normalization, preserving small rounding differences.
        num = gaussian_filter(np.where(val, rd, 0.0), 1.0, mode="constant", cval=0.0)
        den = gaussian_filter(val.astype(float), 1.0, mode="constant", cval=0.0)
        usable = val & (den > 0)
        smoothed = np.full(rd.shape, np.nan, dtype=np.float32)
        smoothed[usable] = num[usable] / den[usable]
        vals = smoothed[usable & np.isfinite(smoothed)]
        m_val, s_val = float(np.mean(vals)), float(np.std(vals))
        z = np.zeros_like(rd, dtype=np.float32)
        z[usable] = (smoothed[usable] - m_val) / s_val
        grid4 = np.asarray(grid_all[p_idx] == 1)
        comp_labels = np.asarray(label_all[p_idx])
        f_patches = []
        for c_id in range(1, int(comp_labels.max()) + 1):
            c_mask = (comp_labels == c_id)
            n_patches_in_comp = int(c_mask.sum())
            support = np.kron(c_mask, np.ones((P04, P04), bool)) & val
            py, px = np.nonzero(support)
            if len(py) == 0:
                continue

            area = len(py)
            cx, cy = float(np.mean(px)), float(np.mean(py))
            pa_vals = pa_raster[py, px]
            r_vals = radius_raster[py, px]

            # Minimal enclosing arc for circular PA geometry
            arc = minimal_circular_enclosing_arc(pa_vals, margin_deg=0.0)
            c_pa = float(arc["arc_center"])
            offsets = circular_diff_deg(pa_vals, c_pa)
            pa_min = float(arc["arc_start"])
            pa_max = float(arc["arc_end"])
            pa_w = float(arc["arc_span"])

            r_min = float(np.min(r_vals))
            r_max = float(np.max(r_vals))
            birth_r_min = r_min

            anchor = np.abs(offsets) <= 5.0
            anchor_r = r_vals[anchor]
            front_r = float(np.percentile(anchor_r if len(anchor_r) >= 2 else r_vals, FRONT_RADIUS_PERCENTILE))
            co, cr = contour_from_pixels(pa_vals, r_vals, c_pa)

            z_vals = z[support]
            max_z = float(z_vals.max())
            mean_z = float(z_vals.mean())

            uid = f"p{pair_num}_c{c_id}"
            patch_obj = PatchComponent(
                component_uid=uid, pair_index=pair_num, timestamp=ts, component_id=c_id,
                area_pixels=area, num_patches=n_patches_in_comp, centroid_x=cx, centroid_y=cy,
                central_pa=c_pa, pa_offsets=offsets, pa_min=pa_min, pa_max=pa_max, pa_width=pa_w,
                front_radius=front_r, r_min=r_min, r_max=r_max, birth_r_min=birth_r_min,
                contour_offsets=co, contour_radii=cr, max_Z=max_z, mean_Z=mean_z,
                bbox_xmin=int(px.min()), bbox_ymin=int(py.min()),
                bbox_xmax=int(px.max()), bbox_ymax=int(py.max()),
            )
            f_patches.append(patch_obj)

            comp_geom_rows.append({
                "pair_index": pair_num, "rd_pair_id": r_man["rd_pair_id"], "timestamp": ts.isoformat(),
                "component_id": c_id, "component_uid": uid, "area_pixels": area,
                "num_patches": n_patches_in_comp, "centroid_x": cx, "centroid_y": cy,
                "central_pa_deg": c_pa, "pa_min_deg": pa_min, "pa_max_deg": pa_max,
                "pa_span_deg": pa_w, "r_min_px": r_min, "r_max_px": r_max,
                "r_front_px": front_r, "r_front_Rsun": front_r * PIXEL_TO_RSUN,
                "max_Z": max_z, "mean_Z": mean_z, "bbox_xmin": int(px.min()),
                "bbox_ymin": int(py.min()), "bbox_xmax": int(px.max()), "bbox_ymax": int(py.max()),
            })

        pairs_data.append({"pair_index": pair_num, "timestamp": ts, "patches": f_patches})
    tracks: dict[int, PatchMotionTrack] = {}
    next_track_id = 1
    candidate_logs = []
    track_state_rows = []

    for step_idx, pdata in enumerate(pairs_data):
        pair_idx = pdata["pair_index"]
        timestamp = pdata["timestamp"]
        patches = pdata["patches"]

        # Step 0: Initial frame spawning
        if step_idx == 0:
            for p in patches:
                tid = next_track_id
                next_track_id += 1
                b_prox = "NEAR" if p.birth_r_min <= BIRTH_GATE_THRESHOLD_PX else "FAR"
                t = PatchMotionTrack(track_id=tid, birth_r_min=p.birth_r_min, birth_proximity=b_prox)
                st = history_state_from_patch(p, timestamp, "OBSERVED")
                st["association_cost"] = 0.0
                st["priority_group"] = np.nan
                t.history.append(st)
                tracks[tid] = t

                track_state_rows.append({
                    "experiment": "E1_patch", "track_id": tid, "pair_index": pair_idx,
                    "timestamp": timestamp.isoformat(), "component_id": p.component_id,
                    "component_uid": p.component_uid, "state": "OBSERVED",
                    "central_pa": p.central_pa, "pa_min": p.pa_min, "pa_max": p.pa_max,
                    "radial_front": p.front_radius, "area_pixels": p.area_pixels,
                    "num_patches": p.num_patches, "centroid_x": p.centroid_x,
                    "centroid_y": p.centroid_y, "association_cost": 0.0,
                    "priority_group": np.nan, "birth_r_min_px": p.birth_r_min,
                    "birth_proximity": b_prox, "previous_component_uid": None,
                })
            continue

        prev_pair_idx = pairs_data[step_idx - 1]["pair_index"]
        prev_timestamp = pairs_data[step_idx - 1]["timestamp"]
        dt_sec = (timestamp - prev_timestamp).total_seconds()

        # Check observation gaps exceeding 75 minutes
        if dt_sec > MAX_CADENCE_GAP_SEC:
            for t in tracks.values():
                if not t.terminated:
                    t.terminated = True

        active_tracks = [t for t in tracks.values() if not t.terminated]
        pair_evals = []

        for t in active_tracks:
            last_obs = next((h for h in reversed(t.history) if h["state"] != "PREDICTED_ONLY"), None)
            src_uid = last_obs["component_uid"] if last_obs else "none"
            src_pa = last_obs["central_pa"] if last_obs else np.nan
            src_r = last_obs["front_radius"] if last_obs else np.nan

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

            for p in patches:
                cost, overlap, pa_diff, radial_diff = patch_association_cost(t, p, timestamp)

                rejection = "NONE"
                if not np.isfinite(cost):
                    rejection = "INFINITE_COST"
                elif cost > ASSOCIATION_COST_MAX:
                    rejection = "COST_EXCEEDS_MAX"

                eval_record = {
                    "experiment": "E1_patch", "source_pair": prev_pair_idx, "target_pair": pair_idx,
                    "source_track_id": t.track_id, "source_component_uid": src_uid,
                    "candidate_component_uid": p.component_uid, "candidate_component_id": p.component_id,
                    "source_PA_center": src_pa, "candidate_PA_center": p.central_pa,
                    "candidate_PA_min": p.pa_min, "candidate_PA_max": p.pa_max,
                    "circular_center_difference": pa_diff, "PA_overlap_ratio": overlap,
                    "source_radial_front": src_r, "candidate_radial_front": p.front_radius,
                    "radial_difference": radial_diff, "baseline_cost": cost,
                    "observed_in_prev_frame": observed_in_prev_frame, "birth_r_min_px": t.birth_r_min,
                    "birth_proximity": t.birth_proximity, "priority_group": priority_group,
                    "selected": False, "rejection_reason": rejection, "track_obj": t,
                    "patch_obj": p, "cost_val": cost,
                }
                pair_evals.append(eval_record)

        # 4-Tier Priority Greedy 1:1 Matching
        pair_evals.sort(key=lambda x: (x["priority_group"], x["cost_val"]))
        used_tracks = set()
        used_patches = set()
        matched_records = []

        for ev in pair_evals:
            tid = ev["source_track_id"]
            p_uid = ev["candidate_component_uid"]

            if not np.isfinite(ev["cost_val"]) or ev["cost_val"] > ASSOCIATION_COST_MAX:
                continue
            if tid in used_tracks:
                if ev["rejection_reason"] == "NONE":
                    ev["rejection_reason"] = "TRACK_ALREADY_MATCHED"
                continue
            if p_uid in used_patches:
                if ev["rejection_reason"] == "NONE":
                    ev["rejection_reason"] = "CANDIDATE_ALREADY_MATCHED"
                continue

            ev["selected"] = True
            ev["rejection_reason"] = "SELECTED"
            used_tracks.add(tid)
            used_patches.add(p_uid)
            matched_records.append(ev)

        for ev in pair_evals:
            if not ev["selected"] and ev["rejection_reason"] == "NONE":
                if ev["source_track_id"] in used_tracks:
                    ev["rejection_reason"] = "TRACK_ALREADY_MATCHED"
                elif ev["candidate_component_uid"] in used_patches:
                    ev["rejection_reason"] = "CANDIDATE_ALREADY_MATCHED"
            candidate_logs.append({k: v for k, v in ev.items() if not k.endswith("_obj") and k != "cost_val"})

        # Apply matched records
        for m in matched_records:
            t = m["track_obj"]
            p = m["patch_obj"]
            st = history_state_from_patch(p, timestamp, "OBSERVED")
            st["association_cost"] = m["cost_val"]
            st["priority_group"] = m["priority_group"]
            t.history.append(st)
            t.missed = 0

            track_state_rows.append({
                "experiment": "E1_patch", "track_id": t.track_id, "pair_index": pair_idx,
                "timestamp": timestamp.isoformat(), "component_id": p.component_id,
                "component_uid": p.component_uid, "state": "OBSERVED",
                "central_pa": p.central_pa, "pa_min": p.pa_min, "pa_max": p.pa_max,
                "radial_front": p.front_radius, "area_pixels": p.area_pixels,
                "num_patches": p.num_patches, "centroid_x": p.centroid_x,
                "centroid_y": p.centroid_y, "association_cost": m["cost_val"],
                "priority_group": m["priority_group"], "birth_r_min_px": t.birth_r_min,
                "birth_proximity": t.birth_proximity, "previous_component_uid": m["source_component_uid"],
            })

        # Process unmatched active tracks -> Gap handling or Termination
        for t in active_tracks:
            if t.track_id in used_tracks:
                continue

            pred = predict_contour(t.history, timestamp)
            can_gap = (t.missed < PATIENCE_FRAMES and pred is not None)
            pred_r = pred["predicted_front_radius"] if pred else None
            pred_pa = pred["central_pa"] if pred else None
            pred_w = pred["angular_width"] if pred else None
            co_offsets = pred["base_contour_offsets"].copy() if pred else np.asarray([])
            co_radii = pred["predicted_contour_radii"].copy() if pred else np.asarray([])

            if can_gap:
                t.missed += 1
                last_obs = next(h for h in reversed(t.history) if h["state"] != "PREDICTED_ONLY")
                pseudo = {
                    "timestamp": timestamp, "state": "PREDICTED_ONLY", "central_pa": pred_pa,
                    "pa_min": last_obs["pa_min"], "pa_max": last_obs["pa_max"], "angular_width": pred_w,
                    "front_radius": pred_r, "area_pixels": last_obs["area_pixels"],
                    "num_patches": last_obs["num_patches"], "centroid_x": last_obs["centroid_x"],
                    "centroid_y": last_obs["centroid_y"], "pa_offsets": last_obs["pa_offsets"].copy(),
                    "contour_offsets": co_offsets, "contour_radii": co_radii,
                    "component_id": -1, "component_uid": "predicted_gap", "pair_index": pair_idx,
                }
                t.history.append(pseudo)

                track_state_rows.append({
                    "experiment": "E1_patch", "track_id": t.track_id, "pair_index": pair_idx,
                    "timestamp": timestamp.isoformat(), "component_id": -1,
                    "component_uid": "predicted_gap", "state": "PREDICTED_ONLY",
                    "central_pa": pred_pa, "pa_min": last_obs["pa_min"],
                    "pa_max": last_obs["pa_max"], "radial_front": pred_r,
                    "area_pixels": last_obs["area_pixels"], "num_patches": last_obs["num_patches"],
                    "centroid_x": last_obs["centroid_x"], "centroid_y": last_obs["centroid_y"],
                    "association_cost": np.nan, "priority_group": np.nan,
                    "birth_r_min_px": t.birth_r_min, "birth_proximity": t.birth_proximity,
                    "previous_component_uid": last_obs["component_uid"],
                })
            else:
                t.terminated = True

        # Process unmatched patches -> Spawning new tracks (NO Birth Gate in E1)
        unmatched_patches = [p for p in patches if p.component_uid not in used_patches]
        for p in unmatched_patches:
            tid = next_track_id
            next_track_id += 1
            b_prox = "NEAR" if p.birth_r_min <= BIRTH_GATE_THRESHOLD_PX else "FAR"
            t = PatchMotionTrack(track_id=tid, birth_r_min=p.birth_r_min, birth_proximity=b_prox)
            st = history_state_from_patch(p, timestamp, "OBSERVED")
            st["association_cost"] = 0.0
            st["priority_group"] = np.nan
            t.history.append(st)
            tracks[tid] = t

            track_state_rows.append({
                "experiment": "E1_patch", "track_id": tid, "pair_index": pair_idx,
                "timestamp": timestamp.isoformat(), "component_id": p.component_id,
                "component_uid": p.component_uid, "state": "OBSERVED",
                "central_pa": p.central_pa, "pa_min": p.pa_min, "pa_max": p.pa_max,
                "radial_front": p.front_radius, "area_pixels": p.area_pixels,
                "num_patches": p.num_patches, "centroid_x": p.centroid_x,
                "centroid_y": p.centroid_y, "association_cost": 0.0,
                "priority_group": np.nan, "birth_r_min_px": p.birth_r_min,
                "birth_proximity": b_prox, "previous_component_uid": None,
            })


    output_dir.mkdir(parents=True, exist_ok=False)
    states = pd.DataFrame(track_state_rows)
    states.to_csv(output_dir / "03_patch_e1_track_states.csv", index=False)
    pd.DataFrame(comp_geom_rows).to_csv(output_dir / "01_patch_component_geometries.csv", index=False)
    pd.DataFrame(candidate_logs).to_csv(output_dir / "02_patch_e1_candidate_evaluations.csv", index=False)
    if not compare_frozen:
        report = {
            "status": "TRACKING_RECOMPUTED_NOT_REFERENCE_VALIDATED",
            "new_component_count": len(comp_geom_rows),
            "new_tracks": len(tracks),
            "new_track_states": len(states),
            "historical_geometry": bool(historical_geometry),
            "note": "Live E1 generation; no historical sample-count or Frozen CSV dependency.",
        }
        (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    refpath = repo / "validation_20101212/experiments/11_d7_patch_first_tracking_feasibility/results/tables/03_patch_e1_track_states.csv"
    frozen = pd.read_csv(refpath)
    key = ["track_id", "pair_index", "state", "component_id", "component_uid"]
    actual_key = states[key].fillna("<NA>").astype(str).sort_values(key).reset_index(drop=True)
    frozen_key = frozen[key].fillna("<NA>").astype(str).sort_values(key).reset_index(drop=True)
    exact = actual_key.equals(frozen_key)
    # Mismatch report retains track identifiers; 'merge' detects absent and extra states.
    delta = frozen_key.merge(actual_key, on=key, how="outer", indicator=True)
    delta.loc[delta["_merge"] != "both"].to_csv(output_dir / "track_state_identity_mismatches.csv", index=False)
    report = {
        "new_component_count": len(comp_geom_rows),
        "frozen_component_reference": 447,
        "new_tracks": len(tracks),
        "frozen_tracks": int(frozen["track_id"].nunique()),
        "new_track_states": len(states),
        "frozen_track_states": len(frozen),
        "track_state_identity_exact": bool(exact),
        "identity_mismatch_rows": int((delta["_merge"] != "both").sum()),
        "status": "PASS" if exact and len(tracks) == frozen["track_id"].nunique() else "MISMATCH",
        "note": "Identity test checks keys only. Numeric geometry equality and downstream replay are separate validations. Uses benchmark 2010 WCS center.",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--detection-dir", type=Path, default=Path("runs/dual_archive_20101212/frozen_replay/detection_c0_gaussian"))
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, default=Path("runs/dual_archive_20101212/frozen_replay/tracking_e1_recomputed"))
    args = p.parse_args()
    raise SystemExit(run(args.detection_dir, args.repo, args.output_dir))