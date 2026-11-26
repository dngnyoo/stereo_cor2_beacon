import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Paths
root = Path(__file__).resolve().parent
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

# Step through and save frame data for pairs 35 to 42 (frame_idx 34 to 41)
frame_data = {}
for idx in range(n_frames):
    ts = timestamps[idx]
    pair_num = int(manifest.iloc[idx]["pair_index"])
    
    # State before
    active_before = {tid: (trk.status, trk.hits, trk.missed_count, trk.latest.cpa_deg, trk.latest.r_front_px, trk.current_vr)
                     for tid, trk in tracker.tracks.items() if trk.status in ("tentative", "confirmed")}
    trk42_state_before = None
    if 42 in tracker.tracks:
        t42 = tracker.tracks[42]
        trk42_state_before = (t42.status, t42.hits, t42.missed_count, t42.latest.cpa_deg, t42.latest.r_front_px, t42.current_vr)
    
    res = tracker.step(frame_idx=idx, timestamp=ts, z_map=z_maps[idx], valid_mask=valid_masks[idx])
    
    if 35 <= pair_num <= 42:
        frame_data[pair_num] = {
            "idx": idx,
            "ts": ts,
            "pair_num": pair_num,
            "active_before": active_before,
            "trk42_before": trk42_state_before,
            "res": res,
            "active_confirmed": list(res.active_confirmed_tracks),
            "matched_pairs": list(res.matched_tracks),
            "newly_born": list(res.newly_born_tracks),
            "terminated": list(res.terminated_tracks),
            "candidates": res.evidence_result.candidates,
        }

print("Collected data for pairs 35-42. Now generating detailed diagnostic plots...")

# Generate high-resolution 8-frame montage of the 46 deg CME propagation & failure
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
axes = axes.flatten()

for i, p_num in enumerate(range(35, 43)):
    ax = axes[i]
    d = frame_data[p_num]
    idx = d["idx"]
    ts = d["ts"]
    rd_img = rd_images[idx]
    val_m = valid_masks[idx]
    vlim = float(np.nanpercentile(np.abs(rd_img[val_m]), 98)) if np.any(val_m) else 1.0
    vlim = max(0.1, vlim)
    
    ax.imshow(rd_img, cmap="gray", vmin=-vlim, vmax=vlim)
    
    # Occulter and outer bounds
    c_in = plt.Circle((sx, sy), grid.r_min_px, color="cyan", fill=False, linestyle="--", linewidth=1.2)
    c_out = plt.Circle((sx, sy), grid.r_max_px, color="cyan", fill=False, linestyle=":", linewidth=1.2)
    c_birth = plt.Circle((sx, sy), cfg.r_birth_max, color="red", fill=False, linestyle=":", linewidth=1.2, label=f"Birth Limit ({cfg.r_birth_max}px)")
    ax.add_patch(c_in)
    ax.add_patch(c_out)
    ax.add_patch(c_birth)
    ax.plot(sx, sy, "+", color="yellow", markersize=6)
    
    # Find the ~46 deg candidate
    cme_cand = None
    for c in d["candidates"]:
        if 25.0 <= c.weighted_cpa_deg <= 65.0:
            cme_cand = c
            break
            
    # Draw CME candidate front arc / ray
    if cme_cand is not None:
        cpa_rad = np.radians(cme_cand.weighted_cpa_deg)
        x_ray = sx - grid.r_max_px * np.sin(cpa_rad)
        y_ray = sy + grid.r_max_px * np.cos(cpa_rad)
        ax.plot([sx, x_ray], [sy, y_ray], color="magenta", linestyle="--", linewidth=1.5, alpha=0.8)
        
        x_front = sx - cme_cand.r_front_px * np.sin(cpa_rad)
        y_front = sy + cme_cand.r_front_px * np.cos(cpa_rad)
        ax.plot(x_front, y_front, "s", color="magenta", markersize=10, markeredgecolor="white", label="Detected Front")
        
        # Draw front arc across lobe width
        theta_lobe = np.linspace(cme_cand.pa_start_deg, cme_cand.pa_end_deg, 30)
        # Handle wraparound
        arc_rad = np.radians(theta_lobe)
        arc_x = sx - cme_cand.r_front_px * np.sin(arc_rad)
        arc_y = sy + cme_cand.r_front_px * np.cos(arc_rad)
        ax.plot(arc_x, arc_y, color="magenta", linewidth=2.0, alpha=0.7)
        
        status_txt = "UNTRACKED / ORPHANED"
        color_box = "red"
        # Check if matched to any track
        matched_trk = [t for t, c in d["matched_pairs"] if c == cme_cand.candidate_id]
        if matched_trk:
            status_txt = f"TRACKED: Track {matched_trk[0]}"
            color_box = "lime"
        elif p_num == 38:
            status_txt = "REJECTED (vr = -6.02 < -6.0)\nTrack 42 TERMINATED"
            color_box = "darkred"
        elif p_num == 37:
            status_txt = "REJECTED (vr = -20.10 < -6.0)\nMiss count = 1"
            color_box = "orange"
        else:
            status_txt = f"UNTRACKED (r={cme_cand.r_front_px:.0f} > 48px)\nBirth Forbidden"
            color_box = "crimson"
            
        ax.annotate(
            f"Cand {cme_cand.candidate_id} (CPA {cme_cand.weighted_cpa_deg:.0f}°)\n"
            f"Front: {cme_cand.r_front_px:.1f}px (pk={cme_cand.r_peak_px:.1f})\n"
            f"A_peak: {cme_cand.peak_evidence:.0f} (W={cme_cand.pa_width_deg:.0f}°)\n"
            f"{status_txt}",
            (x_front + 6, y_front - 6),
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color_box, alpha=0.85, edgecolor="white")
        )

    # Plot confirmed tracks (e.g. 41 and 43)
    for trk in d["active_confirmed"]:
        if trk.track_id == 42:
            continue
        obs = trk.latest
        cpa_rad = np.radians(obs.cpa_deg)
        x_front = sx - obs.r_front_px * np.sin(cpa_rad)
        y_front = sy + obs.r_front_px * np.cos(cpa_rad)
        ax.plot(x_front, y_front, "^", color="cyan", markersize=8, markeredgecolor="black")
        ax.annotate(f"Trk {trk.track_id} ({obs.cpa_deg:.0f}°)", (x_front + 4, y_front + 4), color="cyan", fontsize=7, fontweight="bold")

    ax.set_title(f"Pair {p_num:02d} | {ts.strftime('%H:%M UTC')}", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")
    ax.axis("off")

plt.suptitle("CME Propagation & Tracking Failure Sequence (Pairs 35–42, 2010-12-12)\nDemonstrating Front Overestimation in Pair 36, Inward-Velocity Gating Rejections in Pairs 37-38, and Mid-FOV Orphan State", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
out_montage = root / "experiments/ret_tracking/visualizations/cme_46deg_pairs_35_42_montage.png"
plt.savefig(out_montage, dpi=180)
plt.close()
print(f"Saved {out_montage}")

# Second Figure: Detailed Quantitative Failure Mechanism Analysis
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2)

# Panel A: Radius vs Frame (Front Trajectory & Anomaly)
ax_rad = fig.add_subplot(gs[0, 0])
pairs = list(range(34, 44))
r_fronts = []
r_peaks = []
cpa_vals = []
ev_peaks = []
cpa_widths = []

for p in pairs:
    idx = manifest[manifest["pair_index"] == p].index[0]
    ts = timestamps[idx]
    # We can evaluate candidate from frame_data or run engine
    # Look into candidate from run
    res_ev = compute_pixel_evidence(z_maps[idx], valid_masks[idx], cfg.evidence_formulation)
    res_sp = apply_spatial_gaussian_spreading(res_ev, valid_masks[idx], cfg.sigma_s)
    p_ev, p_val = sample_polar_evidence(res_sp, valid_masks[idx], grid)
    a_raw, a_smooth, val_l = accumulate_radial_evidence(p_ev, p_val, grid, cfg.angular_smooth_sigma_deg, cfg.normalize_by_length)
    cands = detect_angular_candidates(a_smooth, p_ev, grid, cfg.min_prominence, cfg.min_separation_deg, cfg.min_lobe_width_deg, cfg.max_lobe_width_deg, cfg.front_evidence_quantile, cfg.front_min_threshold)
    c_ne = [c for c in cands if 25.0 <= c.weighted_cpa_deg <= 65.0][0]
    r_fronts.append(c_ne.r_front_px)
    r_peaks.append(c_ne.r_peak_px)
    cpa_vals.append(c_ne.weighted_cpa_deg)
    ev_peaks.append(c_ne.peak_evidence)
    cpa_widths.append(c_ne.pa_width_deg)

ax_rad.plot(pairs, r_fronts, "s-", color="red", linewidth=2.5, markersize=8, label="Estimated r_front (px)")
ax_rad.plot(pairs, r_peaks, "o--", color="gold", linewidth=1.8, markersize=6, label="Peak Evidence r_peak (px)")
# Linear fit excluding pair 36 spike
fit_pairs = [p for p in pairs if p != 36]
fit_r = [r for p, r in zip(pairs, r_fronts) if p != 36]
poly = np.polyfit(fit_pairs, fit_r, 1)
trend = np.poly1d(poly)
ax_rad.plot(pairs, trend(pairs), ":", color="blue", linewidth=1.5, label=f"True Physical CME Trend (~{poly[0]*2:.1f} px/hr)")

# Annotate Pair 36 anomaly
ax_rad.axvspan(35.5, 36.5, color="pink", alpha=0.3, label="Pair 36: Lobe Width Explosion (W=120°)")
ax_rad.annotate(f"Pair 36 Spike: r_front=62.2px\n(Outward leap +24.1px)", (36, 62.2), (36.2, 53),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", color="darkred", fontsize=9)

# Annotate Pair 37 drop
ax_rad.annotate(f"Pair 37: r_front=52.2px\n(dr = -10.0px, vr = -20.1 px/hr)\nREJECTED (< -6.0 px/hr)", (37, 52.15), (37.2, 43),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", color="orange", fontsize=9)

# Annotate Pair 38 miss
ax_rad.annotate(f"Pair 38: r_front=59.2px\n(dr = -3.0px vs Pair 36!)\nvr = -6.02 px/hr (REJECTED by 0.02!)\nTrack 42 TERMINATED", (38, 59.19), (38.3, 67),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", color="crimson", fontsize=9)

ax_rad.axhline(cfg.r_birth_max, color="green", linestyle="--", linewidth=1.5, label=f"r_birth_max = {cfg.r_birth_max} px")
ax_rad.set_title("A. Radial Trajectory of 46° Lobe vs Gating Failures", fontsize=12, fontweight="bold")
ax_rad.set_xlabel("RD Pair Index", fontsize=10)
ax_rad.set_ylabel("Radius (pixels)", fontsize=10)
ax_rad.grid(True, linestyle="--", alpha=0.5)
ax_rad.legend(fontsize=8, loc="lower right")

# Panel B: Lobe Width & Evidence Strength Evolution
ax_ev = fig.add_subplot(gs[0, 1])
ax_ev2 = ax_ev.twinx()
ax_ev.bar([p - 0.15 for p in pairs], ev_peaks, width=0.3, color="#2b5c8f", alpha=0.85, label="Peak Evidence A(theta)")
ax_ev2.plot(pairs, cpa_widths, "d-", color="#e67e22", linewidth=2.0, markersize=7, label="Angular Lobe Width (deg)")

ax_ev.set_title("B. Angular Evidence Peak A(theta) & Lobe Width", fontsize=12, fontweight="bold")
ax_ev.set_xlabel("RD Pair Index", fontsize=10)
ax_ev.set_ylabel("Peak Evidence A_t(theta)", color="#2b5c8f", fontsize=10)
ax_ev2.set_ylabel("Lobe Angular Width (deg)", color="#e67e22", fontsize=10)
ax_ev.grid(True, linestyle="--", alpha=0.5)

# Highlight Pair 38 peak
ax_ev.annotate("Pair 38 Peak A ≈ 190\n(Strongest in sequence)", (38, 189.96), (38.2, 160),
               arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
               fontweight="bold", color="darkred", fontsize=9)

# Panel C: Angular Evidence Profile for Pair 38 (Comparison of 46° vs 107° and 276°)
ax_prof = fig.add_subplot(gs[1, 0])
idx_38 = manifest[manifest["pair_index"] == 38].index[0]
res_ev = compute_pixel_evidence(z_maps[idx_38], valid_masks[idx_38], cfg.evidence_formulation)
res_sp = apply_spatial_gaussian_spreading(res_ev, valid_masks[idx_38], cfg.sigma_s)
p_ev, p_val = sample_polar_evidence(res_sp, valid_masks[idx_38], grid)
a_raw, a_smooth, val_l = accumulate_radial_evidence(p_ev, p_val, grid, cfg.angular_smooth_sigma_deg, cfg.normalize_by_length)
cands_38 = detect_angular_candidates(a_smooth, p_ev, grid, cfg.min_prominence, cfg.min_separation_deg, cfg.min_lobe_width_deg, cfg.max_lobe_width_deg, cfg.front_evidence_quantile, cfg.front_min_threshold)

ax_prof.plot(grid.pa_bins, a_smooth, color="#2b5c8f", linewidth=2.2, label="Accumulated Evidence A_t(theta)")
ax_prof.set_xlim(0, 360)
ax_prof.set_xlabel("Position Angle (degrees)", fontsize=10)
ax_prof.set_ylabel("Radial Accumulation A_t(theta)", fontsize=10)
ax_prof.set_title("C. Pair 38 (07:54 UTC) Angular Evidence Profile\nDominant 46° Peak vs Weaker Tracked Lobes", fontsize=12, fontweight="bold")
ax_prof.grid(True, linestyle="--", alpha=0.5)

for c in cands_38:
    if 25 <= c.weighted_cpa_deg <= 65:
        ax_prof.axvline(c.weighted_cpa_deg, color="crimson", linestyle="--", linewidth=1.5)
        ax_prof.plot(c.weighted_cpa_deg, c.peak_evidence, "o", color="crimson", markersize=9)
        ax_prof.annotate(f"CPA {c.weighted_cpa_deg:.1f}° (A = {c.peak_evidence:.1f})\nUNTRACKED / TERMINATED",
                         (c.weighted_cpa_deg, c.peak_evidence), (c.weighted_cpa_deg + 8, c.peak_evidence - 15),
                         fontweight="bold", color="crimson", fontsize=9,
                         arrowprops=dict(facecolor="crimson", shrink=0.08, width=1.5, headwidth=6))
    elif 95 <= c.weighted_cpa_deg <= 120:
        ax_prof.axvline(c.weighted_cpa_deg, color="green", linestyle="--", linewidth=1.5)
        ax_prof.plot(c.weighted_cpa_deg, c.peak_evidence, "o", color="green", markersize=8)
        ax_prof.annotate(f"CPA {c.weighted_cpa_deg:.1f}° (A = {c.peak_evidence:.1f})\nTRACKED: Track 41",
                         (c.weighted_cpa_deg, c.peak_evidence), (c.weighted_cpa_deg + 8, c.peak_evidence + 5),
                         fontweight="bold", color="green", fontsize=9,
                         arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=6))
    elif 260 <= c.weighted_cpa_deg <= 290:
        ax_prof.axvline(c.weighted_cpa_deg, color="teal", linestyle="--", linewidth=1.5)
        ax_prof.plot(c.weighted_cpa_deg, c.peak_evidence, "o", color="teal", markersize=8)
        ax_prof.annotate(f"CPA {c.weighted_cpa_deg:.1f}° (A = {c.peak_evidence:.1f})\nTRACKED: Track 43",
                         (c.weighted_cpa_deg, c.peak_evidence), (c.weighted_cpa_deg - 55, c.peak_evidence + 15),
                         fontweight="bold", color="teal", fontsize=9,
                         arrowprops=dict(facecolor="teal", shrink=0.08, width=1.5, headwidth=6))

# Panel D: Flowchart of Exact Failure Architecture
ax_flow = fig.add_subplot(gs[1, 1])
ax_flow.axis("off")
ax_flow.set_title("D. Root Cause Architecture & Decision Flow", fontsize=12, fontweight="bold")

flow_text = """
1. PAIR 36 (06:54 UTC) — FRONT OVER-ESTIMATION
   - Angular lobe width expands to max allowed: W = 120.0° [340° to 100°].
   - Integrating diffuse northern background causes r_front to falsely leap:
     38.08 px (Pair 35)  --->  62.20 px (Pair 36)   (+24.1 px, vr = +48.2 px/hr)
   - Track 42 absorbs this inflated front.

2. PAIR 37 (07:24 UTC) — INWARD VELOCITY GATE REJECTION
   - Lobe narrows back to core structure (W = 65.5°); actual front r_front = 52.15 px.
   - Gating computes: dr = 52.15 - 62.20 = -10.05 px  -->  vr = -20.10 px/hr.
   - Code Rule: vr < min_radial_velocity_px_hr (-6.0 px/hr)  ==>  REJECTED!
   - Track 42 Missed Count increments from 0 to 1.
   - Birth Check: r_front = 52.15 px > r_birth_max (48.0 px)  ==>  Birth FORBIDDEN!

3. PAIR 38 (07:54 UTC) — TERMINATION BY 0.02 px/hr MARGIN
   - Front advances physically to r_front = 59.19 px.
   - Track 42's last recorded front is STILL 62.20 px (from Pair 36).
   - Gating computes: dr = 59.19 - 62.20 = -3.01 px  -->  vr = -6.02 px/hr.
   - Code Rule: -6.02 < -6.00 px/hr  ==>  REJECTED BY 0.02 px/hr!
   - Track 42 Missed Count reaches 2  ==>  PERMANENTLY TERMINATED!
   - Birth Check: 59.19 px > 48.0 px  ==>  Birth FORBIDDEN!

4. VISUALIZATION OMISSION
   - visualize_results.py plots ONLY res.active_confirmed_tracks.
   - Track 42 is terminated; candidate is unborn; overlay draws NOTHING at 46°!
   - Meanwhile, Tracks 41 (107°) and 43 (276°) never suffered front jumps and survive.
"""

ax_flow.text(0.02, 0.95, flow_text, fontsize=9.5, fontfamily="monospace",
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.5))

plt.tight_layout()
out_analysis = root / "experiments/ret_tracking/visualizations/cme_46deg_failure_mechanism.png"
plt.savefig(out_analysis, dpi=180)
plt.close()
print(f"Saved {out_analysis}")
