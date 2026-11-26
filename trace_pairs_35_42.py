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

# Run tracker up to frame 44, recording every detail
records = []
for idx in range(n_frames):
    ts = timestamps[idx]
    pair_num = manifest.iloc[idx]["pair_index"]
    
    # Pre-step state
    active_before = {tid: (trk.status, trk.hits, trk.missed_count, trk.latest.cpa_deg, trk.latest.r_front_px, trk.current_vr)
                     for tid, trk in tracker.tracks.items() if trk.status in ("tentative", "confirmed")}
    
    res = tracker.step(frame_idx=idx, timestamp=ts, z_map=z_maps[idx], valid_mask=valid_masks[idx])
    
    if 34 <= pair_num <= 43:
        print(f"\n================================================================================")
        print(f"PAIR {pair_num} (frame_idx={idx}) | {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Active tracks before step: {active_before}")
        print(f"Active confirmed after step: {[t.track_id for t in res.active_confirmed_tracks]}")
        print(f"Active tentative after step: {[t.track_id for t in res.active_tentative_tracks]}")
        print(f"Matched pairs: {res.matched_tracks}")
        print(f"Newly born: {res.newly_born_tracks}")
        print(f"Terminated: {res.terminated_tracks}")
        
        # Look at candidates
        cands = res.evidence_result.candidates
        print(f"Total candidates: {len(cands)}")
        for c in cands:
            is_cpa_46_candidate = (25.0 <= c.weighted_cpa_deg <= 65.0) or (25.0 <= c.peak_pa_deg <= 65.0)
            flag = " *** [NE / ~46 deg CANDIDATE] ***" if is_cpa_46_candidate else ""
            print(f"  Cand {c.candidate_id}: CPA={c.weighted_cpa_deg:.2f}° (pk_pa={c.peak_pa_deg:.1f}°), "
                  f"peak_ev={c.peak_evidence:.2f}, total_ev={c.total_lobe_evidence:.1f}, prom={c.prominence:.2f}, "
                  f"width={c.pa_width_deg:.1f}° [{c.pa_start_deg:.1f}° - {c.pa_end_deg:.1f}°], "
                  f"r_front={c.r_front_px:.2f}px, r_peak={c.r_peak_px:.2f}px, r_min={c.r_min_px:.2f}px{flag}")

