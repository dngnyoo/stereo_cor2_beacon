import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from astropy.wcs import WCS
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from cme_pipeline.utils.geometry import circular_diff_deg
from experiments.ret_tracking.ret_evidence_engine import (
    PolarGridGeometry,
    compute_pixel_evidence,
    apply_spatial_gaussian_spreading,
    sample_polar_evidence,
    accumulate_radial_evidence,
    detect_angular_candidates,
)
from experiments.ret_tracking.ret_tracker import RETConfig, RadialEvidenceTracker

root = Path("/home/aababc/stereo_cor2_ret")
data_dir = root / "runs/dual_archive_20101212/preprocessed"
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

table_rows = []

for idx in range(n_frames):
    ts = timestamps[idx]
    pair_num = int(manifest.iloc[idx]["pair_index"])
    
    # Snapshot before step
    active_before_ids = [tid for tid, trk in tracker.tracks.items() if trk.status in ("tentative", "confirmed")]
    t42_before = None
    if 42 in tracker.tracks:
        t = tracker.tracks[42]
        t42_before = {
            "status": t.status,
            "hits": t.hits,
            "missed_count": t.missed_count,
            "cpa": t.latest.cpa_deg,
            "r_front": t.latest.r_front_px,
            "vr": t.current_vr,
            "pa_width": t.latest.pa_width_deg,
        }
    
    dt_hours = 0.5
    if tracker.previous_timestamp is not None:
        dt_hours = max(0.01, (ts - tracker.previous_timestamp).total_seconds() / 3600.0)
        
    res = tracker.step(frame_idx=idx, timestamp=ts, z_map=z_maps[idx], valid_mask=valid_masks[idx])
    
    if 35 <= pair_num <= 42:
        cands = res.evidence_result.candidates
        cme_cand = [c for c in cands if 25.0 <= c.weighted_cpa_deg <= 65.0][0]
        
        # Check matching
        matched_trk_id = None
        cost_val = None
        for tid, cid in res.matched_tracks:
            if cid == cme_cand.candidate_id:
                matched_trk_id = tid
                cost_val = [obs.association_cost for obs in tracker.tracks[tid].observations if obs.frame_idx == idx][0]
                break
                
        # Gating vs T42 before step
        t42_status = "N/A (dead)"
        t42_gating = "N/A"
        if t42_before and t42_before["status"] in ("tentative", "confirmed"):
            d_cpa = abs(circular_diff_deg(cme_cand.weighted_cpa_deg, t42_before["cpa"]))
            dr = cme_cand.r_front_px - t42_before["r_front"]
            vr = dr / dt_hours
            v_pred = t42_before["vr"] if abs(t42_before["vr"]) > 0.1 else 10.0
            r_pred = t42_before["r_front"] + v_pred * dt_hours
            pred_err = abs(cme_cand.r_front_px - r_pred)
            w_diff = abs(cme_cand.pa_width_deg - t42_before["pa_width"])
            c_calc = cfg.w_pa * (d_cpa / 5.0) + cfg.w_front * (pred_err / 5.0) + cfg.w_width * (w_diff / 10.0)
            
            reasons = []
            if d_cpa > cfg.max_pa_diff_deg:
                reasons.append(f"CPA {d_cpa:.1f}° > 20°")
            if vr < cfg.min_radial_velocity_px_hr:
                reasons.append(f"vr {vr:.2f} < -6.0 px/hr")
            if vr > cfg.max_radial_velocity_px_hr:
                reasons.append(f"vr {vr:.2f} > 120.0 px/hr")
            if c_calc >= cfg.gating_cost_threshold:
                reasons.append(f"cost {c_calc:.1f} >= 40.0")
                
            gating_str = "PASS" if not reasons else "FAIL: " + ", ".join(reasons)
            t42_gating = f"dCPA={d_cpa:.1f}°, dr={dr:.1f}px, vr={vr:.2f}px/hr, cost={c_calc:.2f} [{gating_str}]"
            t42_status = f"{t42_before['status']} (hits={t42_before['hits']}, miss={t42_before['missed_count']})"
            
        post_t42_status = "Terminated" if 42 not in tracker.tracks or tracker.tracks[42].status == "terminated" else f"{tracker.tracks[42].status} (miss={tracker.tracks[42].missed_count})"
        
        table_rows.append({
            "Pair": pair_num,
            "UTC": ts.strftime("%H:%M"),
            "Cand_ID": cme_cand.candidate_id,
            "CPA_deg": round(cme_cand.weighted_cpa_deg, 1),
            "Peak_PA_deg": round(cme_cand.peak_pa_deg, 1),
            "A_theta": round(cme_cand.peak_evidence, 1),
            "Total_Ev": round(cme_cand.total_lobe_evidence, 1),
            "Width_deg": round(cme_cand.pa_width_deg, 1),
            "r_front_px": round(cme_cand.r_front_px, 1),
            "r_peak_px": round(cme_cand.r_peak_px, 1),
            "Eligible_Trks": [tid for tid in active_before_ids if abs(circular_diff_deg(tracker.tracks[tid].latest.cpa_deg, cme_cand.weighted_cpa_deg)) <= 30.0],
            "T42_Pre_State": t42_status,
            "T42_Gating": t42_gating,
            "Match_Outcome": f"Matched Track {matched_trk_id} (cost={cost_val:.2f})" if matched_trk_id else "UNMATCHED",
            "Birth_Allowed": cme_cand.r_front_px <= cfg.r_birth_max,
            "T42_Post_State": post_t42_status,
        })

df_diag = pd.DataFrame(table_rows)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_diag.to_string())
