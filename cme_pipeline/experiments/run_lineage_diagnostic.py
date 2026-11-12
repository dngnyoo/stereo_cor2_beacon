#!/usr/bin/env python3
"""
Stereo_COR2 — Adaptive PA-ROI Tracking Lineage Diagnostic
=========================================================
Pairs 30–35 (2010-12-12 Beacon d7 Data)

Root-cause diagnosis of patch lineages between:
- Baseline E1 tracks: T15, T17, T19, T24, T26
- Adaptive PA-ROI tracks: T3, T4, T5, T6, T7, T8

Outputs:
1. tables/
   - patch_level_temporal_lineage.csv
   - track_identity_audit.csv
   - htm_ridge_diagnostic.csv
2. visualizations/
   - fig_01_rd_e1_overlay_pairs30_35.png
   - fig_02_adaptive_ownership_overlay_pairs30_35.png
   - fig_03_birth_seed_connection_paths.png
   - fig_04_t24_to_t3_absorption_zoom.png
   - fig_05_t19_split_boundary_zoom.png
   - fig_06_t26_to_t7_independent_birth_zoom.png
   - fig_07_t3_t4_t7_htm_ridge_comparison.png
   - fig_08_tracking_lineage_6frames.gif
3. frames/fig_08_tracking_lineage_6frames/
   - pair_030.png ... pair_035.png
"""

import sys
from pathlib import Path

# Resolve repository root
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.ndimage import label, binary_dilation
import imageio.v3 as iio

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v01_baseline_tracking import E1TrackingStage
from cme_pipeline.stages.tracking.v02_adaptive_pa_roi import (
    circular_diff_deg, circular_mean_deg, minimal_circular_enclosing_arc,
    STRUCT_8, compute_radial_histogram
)

# Output directories
out_dir = repo_root / "runs/pilot_adaptive_roi_lineage_diagnostic"
tbl_dir = out_dir / "tables"
vis_dir = out_dir / "visualizations"
frm_dir = out_dir / "frames/fig_08_tracking_lineage_6frames"

tbl_dir.mkdir(parents=True, exist_ok=True)
vis_dir.mkdir(parents=True, exist_ok=True)
frm_dir.mkdir(parents=True, exist_ok=True)

# Locked visualization standards
VMIN_RD = -600.0
VMAX_RD = +600.0

def get_rd_cmap():
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("black")
    return cmap

print("=== Starting Adaptive PA-ROI Lineage Diagnostic ===")

# 1. Load Data
cfg = PipelineConfig(run_name="pilot_adaptive_roi_lineage_diagnostic")
ctx = PipelineContext(cfg)

base_stage = E1TrackingStage()
base_out = base_stage.run(None, cfg, ctx)
geom = base_out.geometry
man_df = base_out.manifest_df
grid4_all = base_out.patch_activation_grid4

patch_pa = geom.patch_pa
patch_rad = geom.patch_rad

exp01_dir = repo_root / "validation/frozen/20101212/experiments/01_d7_baseline_segmentation"
rd_all = np.load(exp01_dir / "results/science_arrays/north_up_RD_DNs.npy", mmap_mode="r")
val_all = np.load(exp01_dir / "results/science_arrays/north_up_pair_valid_masks.npy", mmap_mode="r")

exp11_dir = repo_root / "validation/frozen/20101212/experiments/11_d7_patch_first_tracking_feasibility"
df_states_e1 = pd.read_csv(exp11_dir / "results/tables/03_patch_e1_track_states.csv")
lbls_e1_all = np.load(exp11_dir / "results/science_arrays/patch_component_labels.npy")

pilot_dir = repo_root / "runs/pilot_adaptive_roi_pair29_39"
df_states_ad = pd.read_csv(pilot_dir / "tables/02_adaptive_track_states.csv")
df_conflicts_ad = pd.read_csv(pilot_dir / "tables/03_patch_ownership_conflicts.csv")
lbls_ad_all = np.load(pilot_dir / "arrays/adaptive_patch_component_labels.npy")
t_own_ad_all = np.load(pilot_dir / "arrays/adaptive_track_ownership_masks.npy")

target_pairs = [30, 31, 32, 33, 34, 35]
pair_to_fidx = {int(r["pair_index"]): int(r["f_idx"]) for _, r in man_df.iterrows()}
target_fidxs = [pair_to_fidx[p] for p in target_pairs]

print(f"Target Pairs: {target_pairs} -> Frame Indices: {target_fidxs}")

# -------------------------------------------------------------------------
# Phase 1: Track Identity Audit Table
# -------------------------------------------------------------------------
print("\n--- Phase 1: Building Track Identity Audit Table ---")
audit_rows = []

# Audit E1 tracks: T15, T17, T19, T24, T26
for tid in [15, 17, 19, 24, 26]:
    t_rows = df_states_e1[df_states_e1["track_id"] == tid].sort_values("pair_index")
    birth_row = t_rows.iloc[0]
    birth_pair = int(birth_row["pair_index"])
    birth_rmin = float(birth_row["birth_r_min_px"])
    is_anchor = (birth_rmin <= 30.0)

    for p in target_pairs:
        p_row = t_rows[t_rows["pair_index"] == p]
        if len(p_row) > 0:
            r = p_row.iloc[0]
            st = r["state"]
            cid = int(r["component_id"])
            cpa = float(r["central_pa"])
            n_p = int(r["num_patches"])
            rf = float(r["radial_front"]) if "radial_front" in r and not pd.isna(r["radial_front"]) else np.nan
        else:
            st = "ABSENT"
            cid = -1
            cpa = np.nan
            n_p = 0
            rf = np.nan

        # Counterpart note
        counterpart = ""
        diag_note = ""
        if tid == 15 and p == 30:
            counterpart = "Adaptive T3"
            diag_note = "Exact 1:1 match with Adaptive T3 at birth (36 patches, CPA ~108.6°)"
        elif tid == 17 and p in [30, 31]:
            counterpart = "Adaptive T4"
            diag_note = "Match with Adaptive T4 (NE feature, CPA ~41.9°)"
        elif tid == 19:
            if p in [31, 32, 33]:
                counterpart = "Adaptive T3 (East) + T4 (NE) + T5 (North)"
                diag_note = f"Monolithic composite bridging East CME and NE streamer; CPA ~{cpa:.1f}°"
            elif p in [34, 35]:
                counterpart = "Adaptive T4 (NE feature only)"
                diag_note = f"Abruptly snapped North to NE streamer (comp 6, CPA ~{cpa:.1f}°), abandoning East CME!"
        elif tid == 24 and p in [34, 35]:
            counterpart = "Adaptive T3"
            diag_note = f"Downstream front/flank of East CME (45 patches, CPA ~123.8°), absorbed into continuous T3"
        elif tid == 26 and p in [34, 35]:
            counterpart = "Adaptive T7"
            diag_note = f"Upstream root of East CME (46 patches, CPA ~108.4°, r_min={birth_rmin:.1f}px), spawned as Adaptive T7"

        audit_rows.append({
            "track_system": "E1_BASELINE",
            "track_id": tid,
            "pair_index": p,
            "component_id": cid,
            "state": st,
            "num_patches": n_p,
            "central_pa_deg": cpa,
            "tracked_cpa_deg": np.nan,
            "roi_center_pa_deg": np.nan,
            "roi_half_width_deg": np.nan,
            "radial_min_px": np.nan,
            "radial_front_px": rf,
            "birth_pair_index": birth_pair,
            "birth_r_min_px": birth_rmin,
            "original_birth_pa_deg": float(birth_row["central_pa"]),
            "original_birth_cpa_deg": float(birth_row["central_pa"]),
            "is_solar_root_anchor": is_anchor,
            "corresponding_counterpart_id": counterpart,
            "audit_diagnostic_summary": diag_note,
        })

# Audit Adaptive tracks: T3, T4, T5, T6, T7, T8
for tid in [3, 4, 5, 6, 7, 8]:
    t_rows = df_states_ad[df_states_ad["track_id"] == tid].sort_values("pair_index")
    if len(t_rows) == 0:
        continue
    birth_row = t_rows.iloc[0]
    birth_pair = int(birth_row["pair_index"])
    birth_rmin = float(birth_row["birth_r_min_px"])
    orig_bpa = float(birth_row["original_birth_pa"])
    orig_bcpa = float(birth_row["original_birth_cpa"])
    is_anchor = bool(birth_row["is_solar_root_anchor"])

    for p in target_pairs:
        p_row = t_rows[t_rows["pair_index"] == p]
        if len(p_row) > 0:
            r = p_row.iloc[0]
            st = r["state"]
            cid = int(r["component_id"])
            cpa = float(r["central_pa"])
            tcpa = float(r["tracked_cpa"])
            roip = float(r["roi_center_pa"])
            roih = float(r["roi_half_width"])
            rmin = float(r["radial_min"])
            rf = float(r["radial_front"])
            n_p = int(r["num_patches"])
        else:
            st = "ABSENT" if p < birth_pair else "TERMINATED"
            cid = -1
            cpa = np.nan
            tcpa = np.nan
            roip = np.nan
            roih = np.nan
            rmin = np.nan
            rf = np.nan
            n_p = 0

        counterpart = ""
        diag_note = ""
        if tid == 3:
            if p == 30:
                counterpart = "E1 T15"
                diag_note = "Born at CPA 108.6°, exact match with E1 T15"
            elif p in [31, 32, 33]:
                counterpart = "E1 T19 (East portion)"
                diag_note = f"Preserved East CME identity ({n_p} patches, CPA ~{cpa:.1f}°), segregating from NE feature"
            elif p in [34, 35]:
                counterpart = "E1 T24 (Downstream front)"
                diag_note = f"Continuous tracking of downstream front ({n_p} patches, CPA ~{cpa:.1f}°), won seed count over comp 4"
        elif tid == 4:
            if p == 30:
                counterpart = "E1 T17"
                diag_note = "Born at CPA 41.9°, exact match with E1 T17"
            elif p in [31, 32, 33]:
                counterpart = "E1 T19 (NE portion)"
                diag_note = f"Preserved NE coronal feature identity ({n_p} patches, CPA ~{cpa:.1f}°)"
            elif p in [34, 35]:
                counterpart = "E1 T19 (NE feature)"
                diag_note = f"Tracked NE coronal feature continuously ({n_p} patches, CPA ~{cpa:.1f}°)"
        elif tid == 7:
            if p in [34, 35]:
                counterpart = "E1 T26"
                diag_note = f"Born at Pair 34 (46 patches, CPA ~108.4°, r_min={birth_rmin:.1f}px) as independent root anchor"

        audit_rows.append({
            "track_system": "ADAPTIVE_PA_ROI",
            "track_id": tid,
            "pair_index": p,
            "component_id": cid,
            "state": st,
            "num_patches": n_p,
            "central_pa_deg": cpa,
            "tracked_cpa_deg": tcpa,
            "roi_center_pa_deg": roip,
            "roi_half_width_deg": roih,
            "radial_min_px": rmin,
            "radial_front_px": rf,
            "birth_pair_index": birth_pair,
            "birth_r_min_px": birth_rmin,
            "original_birth_pa_deg": orig_bpa,
            "original_birth_cpa_deg": orig_bcpa,
            "is_solar_root_anchor": is_anchor,
            "corresponding_counterpart_id": counterpart,
            "audit_diagnostic_summary": diag_note,
        })

df_audit = pd.DataFrame(audit_rows)
df_audit.to_csv(tbl_dir / "track_identity_audit.csv", index=False)
print(f"Saved track_identity_audit.csv ({len(df_audit)} rows).")

# -------------------------------------------------------------------------
# Phase 2: Patch-Level Temporal Lineage Table
# -------------------------------------------------------------------------
print("\n--- Phase 2: Extracting Patch-Level Temporal Lineage ---")
lineage_rows = []

# Map of E1 components to track IDs per frame
e1_comp_to_tid = {}
for _, r in df_states_e1[df_states_e1["state"] == "OBSERVED"].iterrows():
    p = int(r["pair_index"])
    cid = int(r["component_id"])
    tid = int(r["track_id"])
    e1_comp_to_tid[(p, cid)] = tid

# Active tracks per frame in Adaptive tracking
adaptive_tracks_hist = {}
for tid in df_states_ad["track_id"].unique():
    sub = df_states_ad[df_states_ad["track_id"] == tid]
    adaptive_tracks_hist[tid] = {int(r["pair_index"]): r for _, r in sub.iterrows()}

for p_num in target_pairs:
    f_idx = pair_to_fidx[p_num]
    cur_grid = grid4_all[f_idx]
    ts_str = man_df.loc[man_df["pair_index"] == p_num, "end_time"].iloc[0]

    lbl_img, n_comps = label(cur_grid, structure=STRUCT_8)
    e1_lbl = lbls_e1_all[f_idx]
    ad_lbl = lbls_ad_all[f_idx]
    ad_own = t_own_ad_all[f_idx]

    prev_fidx = pair_to_fidx.get(p_num - 1, None)
    prev_ad_own = t_own_ad_all[prev_fidx] if prev_fidx is not None else np.zeros((64, 64), dtype=int)
    prev_grid = grid4_all[prev_fidx] if prev_fidx is not None else np.zeros((64, 64), dtype=bool)

    active_rcs = list(zip(*np.nonzero(cur_grid)))

    for r, c in active_rcs:
        ppa = float(patch_pa[r, c])
        prad = float(patch_rad[r, c])
        g4_cid = int(lbl_img[r, c])
        e1_cid = int(e1_lbl[r, c])
        e1_tid = e1_comp_to_tid.get((p_num, e1_cid), -1)
        ad_tid = int(ad_own[r, c])

        prev_tid = int(prev_ad_own[r, c])
        is_direct_overlap = (ad_tid > 0 and prev_tid == ad_tid)

        # Connection type classification
        conn_type = "UNKNOWN"
        diag_note = ""

        if is_direct_overlap:
            conn_type = "DIRECT_TEMPORAL_OVERLAP"
            diag_note = f"Direct spatial persistence: Patch was owned by Track {ad_tid} in previous frame"
        elif ad_tid > 0:
            # Check if 8-connected neighbor in current frame had direct overlap
            r_min_n, r_max_n = max(0, r - 1), min(64, r + 2)
            c_min_n, c_max_n = max(0, c - 1), min(64, c + 2)
            nbr_prev_owned = (prev_ad_own[r_min_n:r_max_n, c_min_n:c_max_n] == ad_tid).any()

            # Check conflict records
            conf_match = df_conflicts_ad[(df_conflicts_ad["pair_index"] == p_num) &
                                         (df_conflicts_ad["patch_r"] == r) &
                                         (df_conflicts_ad["patch_c"] == c)]
            is_conflict = len(conf_match) > 0

            if nbr_prev_owned:
                conn_type = "8CONN_TEMPORAL_GROWTH"
                diag_note = f"8-connected neighbor of Track {ad_tid} previous-frame core"
            elif is_conflict:
                cand_ids = conf_match.iloc[0]["candidate_track_ids"]
                conn_type = "ROI_CONTESTED_WINNER"
                diag_note = f"Contested between {cand_ids}, won by Track {ad_tid} based on CPA proximity"
            else:
                # Check if it was born in this frame
                b_info = df_states_ad[(df_states_ad["track_id"] == ad_tid) & (df_states_ad["pair_index"] == p_num)]
                if len(b_info) > 0 and b_info.iloc[0]["pair_index"] == df_states_ad[df_states_ad["track_id"] == ad_tid]["pair_index"].min():
                    conn_type = "BIRTH_SEED_QUALIFIED"
                    diag_note = f"Initial birth seed for new Track {ad_tid} (r_min <= 30 px)"
                else:
                    conn_type = "ROI_UNCONTESTED_SEED"
                    diag_note = f"Within Adaptive ROI corridor of Track {ad_tid}"
        else:
            conn_type = "UNCLAIMED_DISCARDED"
            diag_note = "Active patch not claimed by any track (outside ROIs or rejected by 1-cluster constraint)"

        lineage_rows.append({
            "pair_index": p_num,
            "f_idx": f_idx,
            "timestamp": ts_str,
            "patch_r": r,
            "patch_c": c,
            "patch_pa_deg": ppa,
            "patch_rad_px": prad,
            "grid4_comp_id": g4_cid,
            "e1_comp_id": e1_cid,
            "e1_track_id": e1_tid,
            "adaptive_track_id": ad_tid,
            "is_direct_temporal_overlap": is_direct_overlap,
            "prev_frame_track_id": prev_tid,
            "connection_type": conn_type,
            "lineage_diagnostic_note": diag_note,
        })

df_lineage = pd.DataFrame(lineage_rows)
df_lineage.to_csv(tbl_dir / "patch_level_temporal_lineage.csv", index=False)
print(f"Saved patch_level_temporal_lineage.csv ({len(df_lineage)} rows).")

# -------------------------------------------------------------------------
# Phase 3: Height-Time Map Ridge Diagnostic
# -------------------------------------------------------------------------
print("\n--- Phase 3: Computing Height-Time Map Ridge Diagnostic ---")
htm_rows = []
radial_bins = np.linspace(15.0, 130.0, 47)
r_centers = (radial_bins[:-1] + radial_bins[1:]) / 2.0

for tid in [3, 4, 7]:
    t_states = df_states_ad[df_states_ad["track_id"] == tid].sort_values("pair_index")
    if len(t_states) == 0:
        continue
    orig_bpa = float(t_states.iloc[0]["original_birth_pa"])
    b_cpa = float(t_states.iloc[0]["original_birth_cpa"])

    prev_front = None
    prev_time = None

    for p_num in target_pairs:
        f_idx = pair_to_fidx[p_num]
        cur_grid = grid4_all[f_idx]
        ts_curr = pd.to_datetime(man_df.loc[man_df["pair_index"] == p_num, "end_time"].iloc[0])

        p_row = t_states[t_states["pair_index"] == p_num]
        if len(p_row) > 0 and p_row.iloc[0]["state"] == "OBSERVED":
            r_st = p_row.iloc[0]
            tcpa = float(r_st["tracked_cpa"])
            ocpa = float(r_st["central_pa"])
            roip = float(r_st["roi_center_pa"])
            roih = float(r_st["roi_half_width"])
            front = float(r_st["radial_front"])
            rmin = float(r_st["radial_min"])
            n_assign = int(r_st["num_patches"])

            # Corridor mask
            in_corridor = cur_grid & (np.abs(circular_diff_deg(patch_pa, roip)) <= roih)
            n_raw_roi = int(in_corridor.sum())

            # Patches rejected or assigned to other tracks
            assigned_mask = (t_own_ad_all[f_idx] == tid)
            other_tracks_mask = in_corridor & (~assigned_mask)
            other_tids = [int(x) for x in np.unique(t_own_ad_all[f_idx][other_tracks_mask]) if x > 0]
            n_rej = int(other_tracks_mask.sum())

            # Radial speed
            if prev_front is not None and prev_time is not None:
                dt_hr = (ts_curr - prev_time).total_seconds() / 3600.0
                v_px_hr = (front - prev_front) / dt_hr if dt_hr > 0 else 0.0
            else:
                v_px_hr = np.nan
            prev_front = front
            prev_time = ts_curr

            # Continuity score
            prev_fidx = pair_to_fidx.get(p_num - 1)
            if prev_fidx is not None and (t_own_ad_all[prev_fidx] == tid).any():
                m_prev = (t_own_ad_all[prev_fidx] == tid)
                inter = np.logical_and(m_prev, assigned_mask).sum()
                union = np.logical_or(m_prev, assigned_mask).sum()
                cont_score = float(inter / max(1, union))
            else:
                cont_score = 1.0
        else:
            tcpa = np.nan
            ocpa = np.nan
            front = np.nan
            rmin = np.nan
            n_assign = 0
            n_raw_roi = 0
            n_rej = 0
            other_tids = []
            v_px_hr = np.nan
            cont_score = 0.0

        htm_rows.append({
            "pair_index": p_num,
            "track_id": tid,
            "birth_pa_deg": orig_bpa,
            "birth_cpa_deg": b_cpa,
            "tracked_cpa_deg": tcpa,
            "observed_cpa_deg": ocpa,
            "radial_front_px": front,
            "radial_min_px": rmin,
            "num_assigned_patches": n_assign,
            "num_roi_candidate_patches": n_raw_roi,
            "num_rejected_or_other_patches": n_rej,
            "assigned_to_other_tracks": str(other_tids),
            "radial_speed_px_per_hr": v_px_hr,
            "temporal_continuity_score": cont_score,
        })

df_htm = pd.DataFrame(htm_rows)
df_htm.to_csv(tbl_dir / "htm_ridge_diagnostic.csv", index=False)
print(f"Saved htm_ridge_diagnostic.csv ({len(df_htm)} rows).")

# -------------------------------------------------------------------------
# Phase 4: Qualitative Visualizations (8 Required Figures)
# -------------------------------------------------------------------------
print("\n--- Phase 4: Generating Qualitative Visualizations ---")
cmap_rd = get_rd_cmap()
norm_rd = Normalize(vmin=VMIN_RD, vmax=VMAX_RD)

# Colors locked across all figures
TRACK_COLORS = {
    # E1
    "E1_T15": "#00e5ff", # Cyan
    "E1_T17": "#ffd600", # Yellow
    "E1_T19": "#00e5ff", # Cyan
    "E1_T24": "#76ff03", # Lime
    "E1_T26": "#ff4081", # Magenta
    "E1_T16": "#ff9100", # Orange
    "E1_T22": "#b388ff", # Purple
    # Adaptive
    3: "#00e5ff", # Cyan (East CME)
    4: "#ffd600", # Yellow (NE Feature)
    5: "#76ff03", # Lime (North Micro)
    6: "#ff9100", # Orange (Intermediary)
    7: "#ff4081", # Magenta (East Root)
    8: "#f48fb1", # Pink (North Eruption)
}

def draw_patch_boxes(ax, mask64, edgecolor, linewidth=1.5, hatch=None, facecolor="none", alpha=1.0):
    """Draw 4x4 patch grid boundaries on 256x256 image coordinate axis."""
    rcs = list(zip(*np.nonzero(mask64)))
    for r, c in rcs:
        rect = plt.Rectangle(
            (c * 4, r * 4), 4, 4,
            linewidth=linewidth, edgecolor=edgecolor,
            facecolor=facecolor, hatch=hatch, alpha=alpha
        )
        ax.add_patch(rect)

# -------------------------------------------------------------------------
# Fig 1: RD + E1 Tracks Overlay (6 frames)
# -------------------------------------------------------------------------
print("Generating Fig 01: RD + E1 Tracks Overlay (6 frames)...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="#101010")
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.92, wspace=0.08, hspace=0.14)

for idx, p_num in enumerate(target_pairs):
    ax = axes[idx // 3, idx % 3]
    ax.set_facecolor("#000000")
    f_idx = pair_to_fidx[p_num]
    p_raw = p_num - 1

    rd = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
    ax.imshow(rd, origin="lower", cmap=cmap_rd, norm=norm_rd)

    # Plot E1 tracks
    e1_sub = df_states_e1[(df_states_e1["pair_index"] == p_num) & (df_states_e1["state"] == "OBSERVED")]
    for _, r in e1_sub.iterrows():
        tid = int(r["track_id"])
        cid = int(r["component_id"])
        c_mask = (lbls_e1_all[f_idx] == cid)
        col = TRACK_COLORS.get(f"E1_T{tid}", "#ffffff")
        draw_patch_boxes(ax, c_mask, edgecolor=col, linewidth=1.4)
        # Centroid text
        rcs = np.argwhere(c_mask)
        if len(rcs) > 0:
            mr, mc = rcs.mean(axis=0)
            ax.text(mc * 4 + 2, mr * 4 + 2, f"T{tid}", color="black", fontsize=8.5, fontweight="bold",
                    ha="center", va="center", bbox=dict(boxstyle="square,pad=0.15", facecolor=col, edgecolor="none", alpha=0.85))

    ts = man_df.loc[man_df["pair_index"] == p_num, "end_time"].iloc[0][11:19]
    ax.set_title(f"Pair {p_num:02d} ({ts} UTC) — Baseline E1 Tracking", color="white", fontsize=11, fontweight="bold", pad=6)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.tick_params(colors="gray", labelsize=8)

# Legend
legend_patches = [
    mpatches.Patch(edgecolor="#00e5ff", facecolor="none", linewidth=2.0, label="T15 / T19 (Cyan)"),
    mpatches.Patch(edgecolor="#ffd600", facecolor="none", linewidth=2.0, label="T17 (Yellow)"),
    mpatches.Patch(edgecolor="#76ff03", facecolor="none", linewidth=2.0, label="T24 (Lime)"),
    mpatches.Patch(edgecolor="#ff4081", facecolor="none", linewidth=2.0, label="T26 (Magenta)"),
    mpatches.Patch(edgecolor="#ff9100", facecolor="none", linewidth=2.0, label="T16 (Orange)"),
]
fig.legend(handles=legend_patches, loc="upper center", ncol=5, facecolor="#1e1e1e", edgecolor="gray",
           labelcolor="white", fontsize=9.5, bbox_to_anchor=(0.5, 0.98))

fig.suptitle("Figure 1: Baseline E1 Tracking Overlays (Pairs 30–35) [Raw RD [-600, +600] DN]",
             color="white", fontsize=14, fontweight="bold", y=0.99)
plt.savefig(vis_dir / "fig_01_rd_e1_overlay_pairs30_35.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Fig 2: Adaptive Track Ownership Overlay (6 frames)
# -------------------------------------------------------------------------
print("Generating Fig 02: Adaptive Track Ownership Overlay (6 frames)...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="#101010")
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.92, wspace=0.08, hspace=0.14)

for idx, p_num in enumerate(target_pairs):
    ax = axes[idx // 3, idx % 3]
    ax.set_facecolor("#000000")
    f_idx = pair_to_fidx[p_num]
    p_raw = p_num - 1

    rd = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
    ax.imshow(rd, origin="lower", cmap=cmap_rd, norm=norm_rd)

    # Plot Adaptive tracks
    ad_sub = df_states_ad[(df_states_ad["pair_index"] == p_num) & (df_states_ad["state"] == "OBSERVED")]
    for _, r in ad_sub.iterrows():
        tid = int(r["track_id"])
        c_mask = (t_own_ad_all[f_idx] == tid)
        col = TRACK_COLORS.get(tid, "#ffffff")
        draw_patch_boxes(ax, c_mask, edgecolor=col, linewidth=1.4)
        rcs = np.argwhere(c_mask)
        if len(rcs) > 0:
            mr, mc = rcs.mean(axis=0)
            ax.text(mc * 4 + 2, mr * 4 + 2, f"T{tid}", color="black", fontsize=8.5, fontweight="bold",
                    ha="center", va="center", bbox=dict(boxstyle="square,pad=0.15", facecolor=col, edgecolor="none", alpha=0.85))

    ts = man_df.loc[man_df["pair_index"] == p_num, "end_time"].iloc[0][11:19]
    ax.set_title(f"Pair {p_num:02d} ({ts} UTC) — Adaptive PA-ROI Tracking", color="white", fontsize=11, fontweight="bold", pad=6)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.tick_params(colors="gray", labelsize=8)

# Legend
legend_patches_ad = [
    mpatches.Patch(edgecolor="#00e5ff", facecolor="none", linewidth=2.0, label="Track 3 (East CME)"),
    mpatches.Patch(edgecolor="#ffd600", facecolor="none", linewidth=2.0, label="Track 4 (NE Feature)"),
    mpatches.Patch(edgecolor="#ff4081", facecolor="none", linewidth=2.0, label="Track 7 (East Root)"),
    mpatches.Patch(edgecolor="#76ff03", facecolor="none", linewidth=2.0, label="Track 5 (North Micro)"),
    mpatches.Patch(edgecolor="#ff9100", facecolor="none", linewidth=2.0, label="Track 6 (Intermediary)"),
    mpatches.Patch(edgecolor="#f48fb1", facecolor="none", linewidth=2.0, label="Track 8 (North Erupt)"),
]
fig.legend(handles=legend_patches_ad, loc="upper center", ncol=6, facecolor="#1e1e1e", edgecolor="gray",
           labelcolor="white", fontsize=9.5, bbox_to_anchor=(0.5, 0.98))

fig.suptitle("Figure 2: Adaptive PA-ROI Track Ownership Overlays (Pairs 30–35) [Raw RD [-600, +600] DN]",
             color="white", fontsize=14, fontweight="bold", y=0.99)
plt.savefig(vis_dir / "fig_02_adaptive_ownership_overlay_pairs30_35.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Fig 3: Birth Seed Connection Paths
# -------------------------------------------------------------------------
print("Generating Fig 03: Birth Seed Connection Paths...")
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#101010")
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.88, wspace=0.12)

# Background: Pair 34 RD
p34_raw = 34 - 1
rd34 = np.where(val_all[p34_raw], rd_all[p34_raw], np.nan)

for ax in [ax0, ax1]:
    ax.set_facecolor("#000000")
    ax.imshow(rd34, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.tick_params(colors="gray")

# Left: E1 Lineage Trails (T15 -> T19 -> T24/T26)
ax0.set_title("Baseline E1 Temporal Lineage (Pairs 30–35)", color="white", fontsize=12, fontweight="bold")
# Draw centroid trajectories
for tid, col in [(15, "#00e5ff"), (17, "#ffd600"), (19, "#00e5ff"), (24, "#76ff03"), (26, "#ff4081")]:
    sub = df_states_e1[(df_states_e1["track_id"] == tid) & (df_states_e1["pair_index"].isin(target_pairs)) & (df_states_e1["state"] == "OBSERVED")].sort_values("pair_index")
    if len(sub) > 0:
        xs, ys = [], []
        for _, r in sub.iterrows():
            p = int(r["pair_index"])
            f = pair_to_fidx[p]
            cid = int(r["component_id"])
            m = (lbls_e1_all[f] == cid)
            rcs = np.argwhere(m)
            if len(rcs) > 0:
                mr, mc = rcs.mean(axis=0)
                xs.append(mc * 4 + 2)
                ys.append(mr * 4 + 2)
        ax0.plot(xs, ys, marker="o", color=col, linewidth=2.0, markersize=6, label=f"T{tid}")
        for x, y, p in zip(xs, ys, sub["pair_index"]):
            ax0.text(x + 2, y + 2, f"P{p}", color=col, fontsize=7.5, fontweight="bold")

# Add break annotation for E1 T15 -> T19 and T19 -> T24
ax0.annotate("T15 Terminated\nT19 Born (P31)", xy=(165, 115), xytext=(195, 85),
             arrowprops=dict(facecolor="white", edgecolor="none", arrowstyle="->", lw=1.5),
             color="white", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", facecolor="#262626"))
ax0.annotate("T19 Snaps North (P34)\nT24 & T26 Spawned", xy=(150, 165), xytext=(170, 205),
             arrowprops=dict(facecolor="white", edgecolor="none", arrowstyle="->", lw=1.5),
             color="white", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", facecolor="#262626"))
ax0.legend(loc="lower left", facecolor="#1e1e1e", labelcolor="white", fontsize=8.5)

# Right: Adaptive Lineage Trails (T3, T4, T7)
ax1.set_title("Adaptive PA-ROI Temporal Lineage (Pairs 30–35)", color="white", fontsize=12, fontweight="bold")
for tid, col in [(3, "#00e5ff"), (4, "#ffd600"), (7, "#ff4081")]:
    sub = df_states_ad[(df_states_ad["track_id"] == tid) & (df_states_ad["pair_index"].isin(target_pairs)) & (df_states_ad["state"] == "OBSERVED")].sort_values("pair_index")
    if len(sub) > 0:
        xs, ys = [], []
        for _, r in sub.iterrows():
            p = int(r["pair_index"])
            f = pair_to_fidx[p]
            m = (t_own_ad_all[f] == tid)
            rcs = np.argwhere(m)
            if len(rcs) > 0:
                mr, mc = rcs.mean(axis=0)
                xs.append(mc * 4 + 2)
                ys.append(mr * 4 + 2)
        ax1.plot(xs, ys, marker="s", color=col, linewidth=2.2, markersize=6, label=f"Track {tid}")
        for x, y, p in zip(xs, ys, sub["pair_index"]):
            ax1.text(x + 2, y + 2, f"P{p}", color=col, fontsize=7.5, fontweight="bold")

ax1.annotate("Track 3 Continuous\n(Absorbs E1 T24 Front)", xy=(180, 110), xytext=(195, 70),
             arrowprops=dict(facecolor="#00e5ff", edgecolor="none", arrowstyle="->", lw=1.5),
             color="#00e5ff", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", facecolor="#262626"))
ax1.annotate("Track 7 Born P34\n(Independent East Root)", xy=(150, 125), xytext=(165, 155),
             arrowprops=dict(facecolor="#ff4081", edgecolor="none", arrowstyle="->", lw=1.5),
             color="#ff4081", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", facecolor="#262626"))
ax1.legend(loc="lower left", facecolor="#1e1e1e", labelcolor="white", fontsize=8.5)

fig.suptitle("Figure 3: Centroid Propagation & Birth Seed Lineage Comparison (Pairs 30–35)",
             color="white", fontsize=14, fontweight="bold", y=0.96)
plt.savefig(vis_dir / "fig_03_birth_seed_connection_paths.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Fig 4: T24 -> T3 Absorption Zoom (Pair 34)
# -------------------------------------------------------------------------
print("Generating Fig 04: T24 -> T3 Absorption Zoom (Pair 34)...")
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="#101010")
plt.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.88, wspace=0.10)

f34 = pair_to_fidx[34]
# Zoom window on East CME: x=[120, 240], y=[70, 170]
zoom_x = (115, 235)
zoom_y = (65, 175)

for ax in [ax0, ax1]:
    ax.set_facecolor("#000000")
    ax.imshow(rd34, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax.set_xlim(zoom_x)
    ax.set_ylim(zoom_y)
    ax.tick_params(colors="gray")

# Left: Baseline E1 in Pair 34
ax0.set_title("Pair 34: Baseline E1 Separation (T24 Front vs T26 Root)", color="white", fontsize=11.5, fontweight="bold")
# Component 2 (T24, 45p)
m_t24 = (lbls_e1_all[f34] == 2)
draw_patch_boxes(ax0, m_t24, edgecolor="#76ff03", linewidth=1.8)
# Component 4 (T26, 46p)
m_t26 = (lbls_e1_all[f34] == 4)
draw_patch_boxes(ax0, m_t26, edgecolor="#ff4081", linewidth=1.8)

ax0.text(195, 105, "E1 T24\n(45 patches, Lime)\nBirth r_min=39.7px", color="#76ff03", fontsize=9.5, fontweight="bold",
         ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#76ff03"))
ax0.text(145, 130, "E1 T26\n(46 patches, Magenta)\nBirth r_min=20.6px", color="#ff4081", fontsize=9.5, fontweight="bold",
         ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#ff4081"))

# Right: Adaptive Tracking in Pair 34
ax1.set_title("Pair 34: Adaptive PA-ROI Tracking (Track 3 Continuous vs Track 7 Root)", color="white", fontsize=11.5, fontweight="bold")
m_ad3 = (t_own_ad_all[f34] == 3)
m_ad7 = (t_own_ad_all[f34] == 7)

# Distinguish direct temporal overlap from t-1 vs expansion
f33 = pair_to_fidx[33]
m_t3_prev = (t_own_ad_all[f33] == 3)
m_t3_direct = m_ad3 & m_t3_prev
m_t3_expanded = m_ad3 & (~m_t3_prev)

# Draw direct overlap with solid fill, expanded with outline
draw_patch_boxes(ax1, m_t3_direct, edgecolor="#00e5ff", linewidth=1.8, facecolor="#00e5ff", alpha=0.35)
draw_patch_boxes(ax1, m_t3_expanded, edgecolor="#00e5ff", linewidth=1.8, facecolor="none")
draw_patch_boxes(ax1, m_ad7, edgecolor="#ff4081", linewidth=1.8, facecolor="none")

ax1.text(195, 105, "Adaptive Track 3\n(45 patches = E1 T24)\n[18 direct overlap, 27 exp]", color="#00e5ff", fontsize=9.5, fontweight="bold",
         ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#00e5ff"))
ax1.text(145, 130, "Adaptive Track 7\n(46 patches = E1 T26)\n[Independent Birth Seed]", color="#ff4081", fontsize=9.5, fontweight="bold",
         ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#ff4081"))

fig.suptitle("Figure 4: Zoomed Diagnostic of E1 T24 Absorption into Continuous Adaptive Track 3 (Pair 34)",
             color="white", fontsize=13.5, fontweight="bold", y=0.96)
plt.savefig(vis_dir / "fig_04_t24_to_t3_absorption_zoom.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Fig 5: T19 -> T3/T4 Split Boundary Zoom (Pairs 31–33)
# -------------------------------------------------------------------------
print("Generating Fig 05: T19 -> T3/T4 Split Boundary Zoom (Pairs 31–33)...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#101010")
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.90, wspace=0.08, hspace=0.14)

zoom_split_x = (100, 240)
zoom_split_y = (60, 210)

for col_idx, p_num in enumerate([31, 32, 33]):
    f_idx = pair_to_fidx[p_num]
    p_raw = p_num - 1
    rd_cur = np.where(val_all[p_raw], rd_all[p_raw], np.nan)

    # Top Row: Baseline E1 (T19 monolithic)
    ax_top = axes[0, col_idx]
    ax_top.set_facecolor("#000000")
    ax_top.imshow(rd_cur, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax_top.set_xlim(zoom_split_x)
    ax_top.set_ylim(zoom_split_y)
    ax_top.tick_params(colors="gray")

    cid_t19 = df_states_e1[(df_states_e1["pair_index"] == p_num) & (df_states_e1["track_id"] == 19)].iloc[0]["component_id"]
    m_e1_19 = (lbls_e1_all[f_idx] == cid_t19)
    draw_patch_boxes(ax_top, m_e1_19, edgecolor="#00e5ff", linewidth=1.5)
    ax_top.set_title(f"Pair {p_num}: E1 T19 Monolithic ({m_e1_19.sum()} patches)", color="white", fontsize=11, fontweight="bold")
    ax_top.text(170, 190, f"CPA = {df_states_e1[(df_states_e1['pair_index'] == p_num) & (df_states_e1['track_id'] == 19)].iloc[0]['central_pa']:.1f}°",
                color="#00e5ff", fontsize=9.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a"))

    # Bottom Row: Adaptive Tracking (T3 East vs T4 NE)
    ax_bot = axes[1, col_idx]
    ax_bot.set_facecolor("#000000")
    ax_bot.imshow(rd_cur, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax_bot.set_xlim(zoom_split_x)
    ax_bot.set_ylim(zoom_split_y)
    ax_bot.tick_params(colors="gray")

    m_ad_3 = (t_own_ad_all[f_idx] == 3)
    m_ad_4 = (t_own_ad_all[f_idx] == 4)
    m_ad_5 = (t_own_ad_all[f_idx] == 5)

    draw_patch_boxes(ax_bot, m_ad_3, edgecolor="#00e5ff", linewidth=1.5)
    draw_patch_boxes(ax_bot, m_ad_4, edgecolor="#ffd600", linewidth=1.5)
    draw_patch_boxes(ax_bot, m_ad_5, edgecolor="#76ff03", linewidth=1.2)

    ax_bot.set_title(f"Pair {p_num}: Adaptive T3 (East: {m_ad_3.sum()}p) & T4 (NE: {m_ad_4.sum()}p)", color="white", fontsize=11, fontweight="bold")
    ax_bot.text(180, 95, f"T3 CPA={df_states_ad[(df_states_ad['pair_index'] == p_num) & (df_states_ad['track_id'] == 3)].iloc[0]['central_pa']:.1f}°",
                color="#00e5ff", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a"))
    ax_bot.text(140, 185, f"T4 CPA={df_states_ad[(df_states_ad['pair_index'] == p_num) & (df_states_ad['track_id'] == 4)].iloc[0]['central_pa']:.1f}°",
                color="#ffd600", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a"))

fig.suptitle("Figure 5: Zoomed Evolution of E1 T19 Monolithic Component vs Adaptive T3/T4 Clean Separation (Pairs 31–33)",
             color="white", fontsize=13.5, fontweight="bold", y=0.97)
plt.savefig(vis_dir / "fig_05_t19_split_boundary_zoom.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Fig 6: T26 -> T7 Independent Birth Zoom (Pairs 34 & 35)
# -------------------------------------------------------------------------
print("Generating Fig 06: T26 -> T7 Independent Birth Zoom (Pairs 34 & 35)...")
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="#101010")
plt.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.88, wspace=0.10)

zoom_root_x = (110, 190)
zoom_root_y = (90, 170)

for idx, p_num in enumerate([34, 35]):
    ax = axes[idx]
    ax.set_facecolor("#000000")
    f_idx = pair_to_fidx[p_num]
    p_raw = p_num - 1
    rd_cur = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
    ax.imshow(rd_cur, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax.set_xlim(zoom_root_x)
    ax.set_ylim(zoom_root_y)
    ax.tick_params(colors="gray")

    # Draw occulter edge reference (radius ~ 20 px)
    # Circle at center (128, 128), radius 20 px * 4 = 80 px in full image? Wait, patch coordinates: center is (31.5, 31.5) patches = (126, 126) px
    occ_circle = plt.Circle((128, 128), 21.0 * 4, edgecolor="gray", facecolor="none", linestyle="--", linewidth=1.5, label="Occulter Boundary (~21px)")
    ax.add_patch(occ_circle)

    # Adaptive Track 7 patches
    m_ad_7 = (t_own_ad_all[f_idx] == 7)
    draw_patch_boxes(ax, m_ad_7, edgecolor="#ff4081", linewidth=2.0)

    # Adaptive Track 3 patches (if in view)
    m_ad_3 = (t_own_ad_all[f_idx] == 3)
    draw_patch_boxes(ax, m_ad_3, edgecolor="#00e5ff", linewidth=1.5)

    n_p7 = m_ad_7.sum()
    cpa_7 = df_states_ad[(df_states_ad["pair_index"] == p_num) & (df_states_ad["track_id"] == 7)].iloc[0]["central_pa"]
    rmin_7 = df_states_ad[(df_states_ad["pair_index"] == p_num) & (df_states_ad["track_id"] == 7)].iloc[0]["radial_min"]

    ax.set_title(f"Pair {p_num:02d}: Track 7 ({n_p7}p, CPA={cpa_7:.1f}°, r_min={rmin_7:.1f}px)", color="white", fontsize=11.5, fontweight="bold")
    ax.text(145, 115, f"T7 Root Anchor\nr_min = {rmin_7:.1f}px <= 30px", color="#ff4081", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#ff4081"))

axes[0].legend(loc="upper left", facecolor="#1e1e1e", labelcolor="white", fontsize=8.5)
fig.suptitle("Figure 6: Diagnostic Zoom of Independent Solar Root Anchor Birth (Track 7 / E1 T26 at Pairs 34 & 35)",
             color="white", fontsize=13.5, fontweight="bold", y=0.96)
plt.savefig(vis_dir / "fig_06_t26_to_t7_independent_birth_zoom.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Fig 7: T3/T4/T7 Height-Time Map Comparison
# -------------------------------------------------------------------------
print("Generating Fig 07: T3/T4/T7 Height-Time Map Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor="#101010")
plt.subplots_adjust(left=0.06, right=0.95, bottom=0.12, top=0.86, wspace=0.18)

r_bins = radial_bins
r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0

for ax_idx, tid in enumerate([3, 4, 7]):
    ax = axes[ax_idx]
    ax.set_facecolor("#000000")

    sub_htm = df_htm[df_htm["track_id"] == tid].sort_values("pair_index")
    # Build 2D assigned patch count array (bins x pairs)
    mat = np.zeros((len(r_centers), len(target_pairs)))
    fronts = []
    pairs_with_obs = []

    for c_idx, p_num in enumerate(target_pairs):
        f_idx = pair_to_fidx[p_num]
        m_t = (t_own_ad_all[f_idx] == tid)
        if m_t.any():
            rads = patch_rad[m_t]
            h, _ = np.histogram(rads, bins=r_bins)
            mat[:, c_idx] = h
            fronts.append(np.max(rads))
            pairs_with_obs.append(p_num)

    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="inferno",
                   extent=[min(target_pairs) - 0.5, max(target_pairs) + 0.5, r_bins[0], r_bins[-1]])
    if fronts:
        ax.plot(pairs_with_obs, fronts, color="cyan", linewidth=2.2, marker="o", markersize=6, label="Radial Front")

    t_bpa = df_states_ad[df_states_ad["track_id"] == tid].iloc[0]["original_birth_pa"]
    ax.set_title(f"Track {tid} HTM Ridge (Birth PA = {t_bpa:.1f}°)", color="white", fontsize=11.5, fontweight="bold", pad=8)
    ax.set_xlabel("Pair Index", color="white", fontsize=9.5)
    ax.set_ylabel("Heliocentric Radius (px)", color="white", fontsize=9.5)
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", facecolor="#262626", labelcolor="white", fontsize=8.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Assigned Patches", color="white", fontsize=8.5)
    cb.ax.tick_params(colors="white")

fig.suptitle("Figure 7: Diagnostic Height–Time Map (HTM) Ridge Comparison along Birth PAs (Pairs 30–35)",
             color="white", fontsize=13.5, fontweight="bold", y=0.96)
plt.savefig(vis_dir / "fig_07_t3_t4_t7_htm_ridge_comparison.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Fig 8: 6-frame Tracking Lineage GIF & Individual Frames Archive
# -------------------------------------------------------------------------
print("Generating Fig 08: 6-frame Tracking Lineage GIF & Archiving PNG Frames...")
gif_frames = []

for p_num in target_pairs:
    f_idx = pair_to_fidx[p_num]
    p_raw = p_num - 1

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0e0e0e")
    plt.subplots_adjust(left=0.03, right=0.95, bottom=0.06, top=0.88, wspace=0.08)

    rd = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
    ts_str = man_df.loc[man_df["pair_index"] == p_num, "end_time"].iloc[0]

    # Panel 1: E1 Baseline
    ax0.set_facecolor("#000000")
    ax0.imshow(rd, origin="lower", cmap=cmap_rd, norm=norm_rd)
    e1_sub = df_states_e1[(df_states_e1["pair_index"] == p_num) & (df_states_e1["state"] == "OBSERVED")]
    for _, r in e1_sub.iterrows():
        tid = int(r["track_id"])
        cid = int(r["component_id"])
        c_mask = (lbls_e1_all[f_idx] == cid)
        col = TRACK_COLORS.get(f"E1_T{tid}", "#ffffff")
        draw_patch_boxes(ax0, c_mask, edgecolor=col, linewidth=1.5)
        rcs = np.argwhere(c_mask)
        if len(rcs) > 0:
            mr, mc = rcs.mean(axis=0)
            ax0.text(mc * 4 + 2, mr * 4 + 2, f"T{tid}", color="black", fontsize=9, fontweight="bold",
                     ha="center", va="center", bbox=dict(boxstyle="square,pad=0.15", facecolor=col, edgecolor="none", alpha=0.85))

    ax0.set_title(f"Baseline E1 Tracking — Pair {p_num:02d} ({ts_str[11:19]} UTC)", color="white", fontsize=11.5, fontweight="bold")
    ax0.set_xlim(0, 256)
    ax0.set_ylim(0, 256)
    ax0.tick_params(colors="gray")

    # Panel 2: Adaptive Tracking with Connection Type distinctions
    ax1.set_facecolor("#000000")
    ax1.imshow(rd, origin="lower", cmap=cmap_rd, norm=norm_rd)

    # Classify patches in this frame
    p_lineage = df_lineage[df_lineage["pair_index"] == p_num]
    for tid in [3, 4, 5, 6, 7, 8]:
        sub_t = p_lineage[p_lineage["adaptive_track_id"] == tid]
        if len(sub_t) == 0:
            continue
        col = TRACK_COLORS.get(tid, "#ffffff")

        # 1. Direct temporal overlap (solid fill)
        t_dir = sub_t[sub_t["connection_type"] == "DIRECT_TEMPORAL_OVERLAP"]
        for _, pr in t_dir.iterrows():
            rect = plt.Rectangle((pr["patch_c"] * 4, pr["patch_r"] * 4), 4, 4,
                                 linewidth=1.2, edgecolor=col, facecolor=col, alpha=0.35)
            ax1.add_patch(rect)

        # 2. 8-connected growth (hatched)
        t_8c = sub_t[sub_t["connection_type"] == "8CONN_TEMPORAL_GROWTH"]
        for _, pr in t_8c.iterrows():
            rect = plt.Rectangle((pr["patch_c"] * 4, pr["patch_r"] * 4), 4, 4,
                                 linewidth=1.2, edgecolor=col, facecolor="none", hatch="///")
            ax1.add_patch(rect)

        # 3. ROI seeds / birth seeds (outline only)
        t_oth = sub_t[~sub_t["connection_type"].isin(["DIRECT_TEMPORAL_OVERLAP", "8CONN_TEMPORAL_GROWTH"])]
        for _, pr in t_oth.iterrows():
            rect = plt.Rectangle((pr["patch_c"] * 4, pr["patch_r"] * 4), 4, 4,
                                 linewidth=1.4, edgecolor=col, facecolor="none")
            ax1.add_patch(rect)

        # Track Label
        mr = sub_t["patch_r"].mean()
        mc = sub_t["patch_c"].mean()
        ax1.text(mc * 4 + 2, mr * 4 + 2, f"T{tid}", color="black", fontsize=9, fontweight="bold",
                 ha="center", va="center", bbox=dict(boxstyle="square,pad=0.15", facecolor=col, edgecolor="none", alpha=0.85))

    ax1.set_title(f"Adaptive PA-ROI Tracking — Pair {p_num:02d} ({ts_str[11:19]} UTC)", color="white", fontsize=11.5, fontweight="bold")
    ax1.set_xlim(0, 256)
    ax1.set_ylim(0, 256)
    ax1.tick_params(colors="gray")

    fig.suptitle(f"Figure 8: Side-by-Side Lineage Animation — Pair {p_num:02d} (2010-12-12 {ts_str[11:19]} UTC)",
                 color="white", fontsize=13, fontweight="bold", y=0.96)

    # Archive high-res PNG frame per VISUALIZATION_STANDARDS.md
    frame_png_path = frm_dir / f"pair_{p_num:03d}.png"
    plt.savefig(frame_png_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # Read back saved PNG for GIF construction
    frame_img = iio.imread(frame_png_path)
    gif_frames.append(frame_img)

# Save animated GIF
gif_path = vis_dir / "fig_08_tracking_lineage_6frames.gif"
iio.imwrite(gif_path, gif_frames, duration=1000, loop=0)
print(f"Saved fig_08_tracking_lineage_6frames.gif ({len(gif_frames)} frames).")
print(f"Archived {len(gif_frames)} individual frames in {frm_dir}/.")

print("\n=== All Lineage Diagnostic Computations & Visualizations Completed Successfully ===")

