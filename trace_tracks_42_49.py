import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from astropy.wcs import WCS
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg
from experiments.ret_tracking.ret_evidence_engine import (
    PolarGridGeometry,
    compute_pixel_evidence,
    apply_spatial_gaussian_spreading,
    sample_polar_evidence,
    accumulate_radial_evidence,
    detect_angular_candidates,
)
from experiments.ret_tracking.ret_tracker import RETConfig, RadialEvidenceTracker

data_dir = Path("runs/dual_archive_20101212/preprocessed")
manifest = pd.read_csv(data_dir / "01_rd_manifest.csv")
rd_images = np.load(data_dir / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
valid_masks = np.load(data_dir / "north_up_pair_valid_masks.npy", mmap_mode="r")
hdr_str = (data_dir / "target_wcs_header.txt").read_text(encoding="utf-8")
wcs = WCS(fits.Header.fromstring(hdr_str, sep="\n"))
sx, sy = wcs.world_to_pixel_values(0.0, 0.0)
grid = PolarGridGeometry(sx=float(sx), sy=float(sy))
timestamps = [datetime.fromisoformat(t) for t in manifest["end_time"]]

n_frames = len(manifest)
z_maps = np.zeros((n_frames, 256, 256), dtype=np.float32)
for idx in range(n_frames):
    image = np.asarray(rd_images[idx], dtype=np.float32)
    val = np.asarray(valid_masks[idx], dtype=bool)
    num = gaussian_filter(np.where(val, image, 0.0), 1.0, mode="constant", cval=0.0)
    den = gaussian_filter(val.astype(float), 1.0, mode="constant", cval=0.0)
    good = val & (den > 0)
    smoothed = np.full(image.shape, np.nan, dtype=np.float32)
    smoothed[good] = (num[good] / den[good]).astype(np.float32)
    fin = smoothed[good & np.isfinite(smoothed)]
    if fin.size and float(fin.std()) > 0:
        z_maps[idx, good] = (smoothed[good] - float(fin.mean())) / float(fin.std())

cfg = RETConfig(variant="RET-2", sigma_s=2.0)
tracker = RadialEvidenceTracker(config=cfg, grid=grid)

def inspect_track_gating(target_tid, frame_idx, trk, candidates, dt_hours):
    last_obs = trk.latest
    v_pred = trk.current_vr if abs(trk.current_vr) > 0.1 else 10.0
    r_pred = last_obs.r_front_px + v_pred * dt_hours
    print(f"\n--- GATING & COST EVALUATION FOR TRACK {target_tid} AT FRAME {frame_idx} ---")
    print(f"  Track state: status={trk.status}, hits={trk.hits}, missed={trk.missed_count}")
    print(f"  Last obs: CPA={last_obs.cpa_deg:.2f} deg, width={last_obs.pa_width_deg:.2f} deg, r_front={last_obs.r_front_px:.2f} px")
    print(f"  Motion prediction: v_pred={v_pred:.2f} px/hr, dt={dt_hours:.4f} hr -> r_pred={r_pred:.2f} px")
    for j, cand in enumerate(candidates):
        d_cpa = abs(circular_diff_deg(cand.weighted_cpa_deg, last_obs.cpa_deg))
        dr = cand.r_front_px - last_obs.r_front_px
        vr_instant = dr / dt_hours
        
        status_reasons = []
        if d_cpa > cfg.max_pa_diff_deg:
            status_reasons.append(f"FAIL CPA gate (d_cpa={d_cpa:.2f} > {cfg.max_pa_diff_deg})")
        if vr_instant < cfg.min_radial_velocity_px_hr:
            status_reasons.append(f"FAIL min vr gate (vr={vr_instant:.2f} < {cfg.min_radial_velocity_px_hr})")
        if vr_instant > cfg.max_radial_velocity_px_hr:
            status_reasons.append(f"FAIL max vr gate (vr={vr_instant:.2f} > {cfg.max_radial_velocity_px_hr})")
        
        pred_err = abs(cand.r_front_px - r_pred)
        w_diff = abs(cand.pa_width_deg - last_obs.pa_width_deg)
        cost = cfg.w_pa * (d_cpa / 5.0) + cfg.w_front * (pred_err / 5.0) + cfg.w_width * (w_diff / 10.0)
        
        if cost >= cfg.gating_cost_threshold:
            status_reasons.append(f"FAIL cost threshold (cost={cost:.2f} >= {cfg.gating_cost_threshold})")
        
        status_str = "PASS GATES -> cost=" + f"{cost:.2f}" if not status_reasons else "REJECTED: " + "; ".join(status_reasons)
        print(f"  -> Cand {cand.candidate_id} (CPA={cand.weighted_cpa_deg:.1f}, r_front={cand.r_front_px:.1f}, w={cand.pa_width_deg:.1f}): {status_str}")

for idx in range(n_frames):
    ts = timestamps[idx]
    active_before = [tid for tid, trk in tracker.tracks.items() if trk.status in ("tentative", "confirmed")]
    
    # Calculate dt
    dt_hours = 0.5
    if tracker.previous_timestamp is not None:
        dt_hours = max(0.01, (ts - tracker.previous_timestamp).total_seconds() / 3600.0)
    
    # Peek candidates
    ev_raw = compute_pixel_evidence(z_maps[idx], valid_masks[idx], cfg.evidence_formulation)
    ev_spread = apply_spatial_gaussian_spreading(ev_raw, valid_masks[idx], cfg.sigma_s)
    polar_ev, polar_val = sample_polar_evidence(ev_spread, valid_masks[idx], grid)
    a_raw, a_smooth, val_len = accumulate_radial_evidence(polar_ev, polar_val, grid, cfg.angular_smooth_sigma_deg, cfg.normalize_by_length)
    candidates = detect_angular_candidates(
        a_smooth, polar_ev, grid,
        cfg.min_prominence, cfg.min_separation_deg,
        cfg.min_lobe_width_deg, cfg.max_lobe_width_deg,
        cfg.front_evidence_quantile, cfg.front_min_threshold
    )
    
    if idx in (35, 36, 37) and 42 in tracker.tracks:
        inspect_track_gating(42, idx, tracker.tracks[42], candidates, dt_hours)
    
    if idx in (43, 44, 45, 46, 47) and 49 in tracker.tracks:
        inspect_track_gating(49, idx, tracker.tracks[49], candidates, dt_hours)
    
    res = tracker.step(frame_idx=idx, timestamp=ts, z_map=z_maps[idx], valid_mask=valid_masks[idx])
