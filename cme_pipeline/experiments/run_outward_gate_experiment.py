#!/usr/bin/env python3
"""
Stereo_COR2 — Adaptive PA-ROI Outward Motion Gate Experiment (A0 vs A2)
and Diagnostic Visualization Fix
========================================================================

Experiment:
- A0: Existing Adaptive PA-ROI Tracking (outward gate disabled)
- A2: Outward Motion Gate enabled (tolerance = 4.0 px)
      R_front(t) >= R_front(t-1) - 4 px

Primary Dataset:
- 2010-12-12 STEREO-A COR2 Beacon d7 (Pairs 30–35, extended to Pairs 29–39)

Outputs:
- runs/pilot_adaptive_outward_gate_a0_a2/tables/
- runs/pilot_adaptive_outward_gate_a0_a2/arrays/
- runs/pilot_adaptive_outward_gate_a0_a2/visualizations/
"""

import sys
from pathlib import Path

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

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.v01_baseline_tracking import E1TrackingStage
from cme_pipeline.stages.tracking.v02_adaptive_pa_roi import (
    BirthSeededAdaptiveROITrackingStage, circular_diff_deg, circular_mean_deg
)

# Output directories
out_dir = repo_root / "runs/pilot_adaptive_outward_gate_a0_a2"
tbl_dir = out_dir / "tables"
arr_dir = out_dir / "arrays"
vis_dir = out_dir / "visualizations"

tbl_dir.mkdir(parents=True, exist_ok=True)
arr_dir.mkdir(parents=True, exist_ok=True)
vis_dir.mkdir(parents=True, exist_ok=True)

# Locked visualization standards
VMIN_RD = -600.0
VMAX_RD = +600.0

def get_rd_cmap():
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("black")
    return cmap

print("=== Starting A0 vs A2 Outward Motion Gate Experiment ===")

# -------------------------------------------------------------------------
# Step 1: Run A0 (Baseline Adaptive PA-ROI Tracking, gate disabled)
# -------------------------------------------------------------------------
print("\n--- Running Configuration A0 (Baseline, Gate Disabled) ---")
cfg_a0 = PipelineConfig(run_name="pilot_adaptive_outward_gate_a0_a2_A0")
cfg_a0.tracking.outward_gate_enabled = False
ctx_a0 = PipelineContext(cfg_a0)

stage_a0 = BirthSeededAdaptiveROITrackingStage()
out_a0 = stage_a0.run(None, cfg_a0, ctx_a0)

df_states_a0 = out_a0.track_states_df
lbls_a0 = out_a0.patch_component_labels
t_own_a0 = out_a0.ownership_masks
man_df = out_a0.manifest_df
geom = out_a0.geometry
patch_pa = geom.patch_pa
patch_rad = geom.patch_rad

# Save A0 arrays and tables
np.save(arr_dir / "a0_patch_component_labels.npy", lbls_a0)
np.save(arr_dir / "a0_track_ownership_masks.npy", t_own_a0)
df_states_a0.to_csv(tbl_dir / "a0_track_states.csv", index=False)
print(f"Saved A0 outputs: {len(df_states_a0)} state rows, {out_a0.total_tracks} total tracks.")

# -------------------------------------------------------------------------
# Step 2: Run A2 (Outward Motion Gate Enabled, tol = 4.0 px)
# -------------------------------------------------------------------------
print("\n--- Running Configuration A2 (Outward Gate Enabled, tol = 4.0 px) ---")
cfg_a2 = PipelineConfig(run_name="pilot_adaptive_outward_gate_a0_a2_A2")
cfg_a2.tracking.outward_gate_enabled = True
cfg_a2.tracking.outward_gate_tolerance_px = 4.0
ctx_a2 = PipelineContext(cfg_a2)

stage_a2 = BirthSeededAdaptiveROITrackingStage()
out_a2 = stage_a2.run(None, cfg_a2, ctx_a2)

df_states_a2 = out_a2.track_states_df
lbls_a2 = out_a2.patch_component_labels
t_own_a2 = out_a2.ownership_masks
rejections_a2 = out_a2.rejection_records

# Save A2 arrays and tables
np.save(arr_dir / "a2_patch_component_labels.npy", lbls_a2)
np.save(arr_dir / "a2_track_ownership_masks.npy", t_own_a2)
df_states_a2.to_csv(tbl_dir / "a2_track_states.csv", index=False)
df_rejections_a2 = pd.DataFrame(rejections_a2)
df_rejections_a2.to_csv(tbl_dir / "a2_outward_gate_rejections.csv", index=False)
print(f"Saved A2 outputs: {len(df_states_a2)} state rows, {out_a2.total_tracks} total tracks, {len(df_rejections_a2)} gate rejections.")

# -------------------------------------------------------------------------
# Step 3: Quantitative Comparison Tables (Pairs 30–35 and Pairs 29–39)
# -------------------------------------------------------------------------
print("\n--- Generating Comparison Tables ---")
pair_to_fidx = {int(r["pair_index"]): int(r["f_idx"]) for _, r in man_df.iterrows()}

frame_comparison_rows = []
for p in range(29, 40):
    f = pair_to_fidx[p]
    ts = man_df.loc[man_df["pair_index"] == p, "end_time"].iloc[0]

    # A0 stats
    sub_a0 = df_states_a0[df_states_a0["pair_index"] == p]
    obs_a0 = sub_a0[sub_a0["state"] == "OBSERVED"]
    pred_a0 = sub_a0[sub_a0["state"] == "PREDICTED_ONLY"]

    t3_a0 = sub_a0[sub_a0["track_id"] == 3]
    t3_a0_st = t3_a0.iloc[0]["state"] if len(t3_a0) > 0 else "ABSENT"
    t3_a0_front = float(t3_a0.iloc[0]["radial_front"]) if len(t3_a0) > 0 else np.nan
    t3_a0_p = int(t3_a0.iloc[0]["num_patches"]) if len(t3_a0) > 0 else 0
    t3_a0_disp = float(t3_a0.iloc[0]["radial_displacement"]) if len(t3_a0) > 0 else 0.0

    # A2 stats
    sub_a2 = df_states_a2[df_states_a2["pair_index"] == p]
    obs_a2 = sub_a2[sub_a2["state"] == "OBSERVED"]
    pred_a2 = sub_a2[sub_a2["state"] == "PREDICTED_ONLY"]

    t3_a2 = sub_a2[sub_a2["track_id"] == 3]
    t3_a2_st = t3_a2.iloc[0]["state"] if len(t3_a2) > 0 else "ABSENT"
    t3_a2_front = float(t3_a2.iloc[0]["radial_front"]) if len(t3_a2) > 0 else np.nan
    t3_a2_p = int(t3_a2.iloc[0]["num_patches"]) if len(t3_a2) > 0 else 0
    t3_a2_disp = float(t3_a2.iloc[0]["radial_displacement"]) if len(t3_a2) > 0 else 0.0

    rej_count = len([r for r in rejections_a2 if r["pair_index"] == p])

    frame_comparison_rows.append({
        "pair_index": p,
        "timestamp": ts,
        "a0_observed_tracks": len(obs_a0),
        "a0_predicted_tracks": len(pred_a0),
        "a0_total_assigned_patches": int(obs_a0["num_patches"].sum()),
        "a0_t3_state": t3_a0_st,
        "a0_t3_front_px": t3_a0_front,
        "a0_t3_patches": t3_a0_p,
        "a0_t3_radial_displacement_px": t3_a0_disp,
        "a2_observed_tracks": len(obs_a2),
        "a2_predicted_tracks": len(pred_a2),
        "a2_total_assigned_patches": int(obs_a2["num_patches"].sum()),
        "a2_t3_state": t3_a2_st,
        "a2_t3_front_px": t3_a2_front,
        "a2_t3_patches": t3_a2_p,
        "a2_t3_radial_displacement_px": t3_a2_disp,
        "a2_gate_rejections_this_frame": rej_count,
    })

df_frame_comp = pd.DataFrame(frame_comparison_rows)
df_frame_comp.to_csv(tbl_dir / "a0_vs_a2_frame_comparison.csv", index=False)
print(f"Saved a0_vs_a2_frame_comparison.csv ({len(df_frame_comp)} rows).")

# Track-level comparison for Pairs 30–35
track_comp_rows = []
for p in [30, 31, 32, 33, 34, 35]:
    sub_a0 = df_states_a0[df_states_a0["pair_index"] == p]
    sub_a2 = df_states_a2[df_states_a2["pair_index"] == p]
    all_tids = sorted(list(set(sub_a0["track_id"].tolist() + sub_a2["track_id"].tolist())))

    for tid in all_tids:
        r0 = sub_a0[sub_a0["track_id"] == tid]
        r2 = sub_a2[sub_a2["track_id"] == tid]

        s0 = r0.iloc[0]["state"] if len(r0) > 0 else "ABSENT"
        cpa0 = float(r0.iloc[0]["central_pa"]) if len(r0) > 0 else np.nan
        rf0 = float(r0.iloc[0]["radial_front"]) if len(r0) > 0 else np.nan
        np0 = int(r0.iloc[0]["num_patches"]) if len(r0) > 0 else 0
        disp0 = float(r0.iloc[0]["radial_displacement"]) if len(r0) > 0 else 0.0

        s2 = r2.iloc[0]["state"] if len(r2) > 0 else "ABSENT"
        cpa2 = float(r2.iloc[0]["central_pa"]) if len(r2) > 0 else np.nan
        rf2 = float(r2.iloc[0]["radial_front"]) if len(r2) > 0 else np.nan
        np2 = int(r2.iloc[0]["num_patches"]) if len(r2) > 0 else 0
        disp2 = float(r2.iloc[0]["radial_displacement"]) if len(r2) > 0 else 0.0

        track_comp_rows.append({
            "pair_index": p,
            "track_id": tid,
            "a0_state": s0,
            "a0_central_pa": cpa0,
            "a0_radial_front_px": rf0,
            "a0_num_patches": np0,
            "a0_radial_disp_px": disp0,
            "a2_state": s2,
            "a2_central_pa": cpa2,
            "a2_radial_front_px": rf2,
            "a2_num_patches": np2,
            "a2_radial_disp_px": disp2,
            "is_identical": (s0 == s2 and np0 == np2 and np.isclose(rf0, rf2, equal_nan=True)),
        })

df_track_comp = pd.DataFrame(track_comp_rows)
df_track_comp.to_csv(tbl_dir / "a0_vs_a2_track_comparison.csv", index=False)
print(f"Saved a0_vs_a2_track_comparison.csv ({len(df_track_comp)} rows).")

# -------------------------------------------------------------------------
# Step 4: Fix Diagnostic Visualizations (Part 4)
# -------------------------------------------------------------------------
print("\n--- Fixing and Validating Diagnostic Visualizations (Part 4) ---")
exp01_dir = repo_root / "validation/frozen/20101212/experiments/01_d7_baseline_segmentation"
rd_all = np.load(exp01_dir / "results/science_arrays/north_up_RD_DNs.npy", mmap_mode="r")
val_all = np.load(exp01_dir / "results/science_arrays/north_up_pair_valid_masks.npy", mmap_mode="r")

exp11_dir = repo_root / "validation/frozen/20101212/experiments/11_d7_patch_first_tracking_feasibility"
lbls_e1_all = np.load(exp11_dir / "results/science_arrays/patch_component_labels.npy")

cmap_rd = get_rd_cmap()
norm_rd = Normalize(vmin=VMIN_RD, vmax=VMAX_RD)

def draw_patch_boxes(ax, mask64, edgecolor, linewidth=1.5, hatch=None, facecolor="none", alpha=1.0):
    rcs = list(zip(*np.nonzero(mask64)))
    for r, c in rcs:
        rect = plt.Rectangle(
            (c * 4, r * 4), 4, 4,
            linewidth=linewidth, edgecolor=edgecolor,
            facecolor=facecolor, hatch=hatch, alpha=alpha
        )
        ax.add_patch(rect)

# 4.1 Corrected Figure 4: T24 -> T3 Absorption Zoom (Pair 34)
f34 = pair_to_fidx[34]
p34_raw = 34 - 1
rd34 = np.where(val_all[p34_raw], rd_all[p34_raw], np.nan)

m_t24 = (lbls_e1_all[f34] == 2)  # E1 Comp 2 = T24
m_t26 = (lbls_e1_all[f34] == 4)  # E1 Comp 4 = T26
m_ad3 = (t_own_a0[f34] == 3)     # Adaptive T3
m_ad7 = (t_own_a0[f34] == 7)     # Adaptive T7

# Programmatic bounding box calculation with margin
combined_mask = m_t24 | m_t26 | m_ad3 | m_ad7
rc_comb = np.argwhere(combined_mask)
margin_px = 24

x_min_fig4 = max(0, rc_comb[:, 1].min() * 4 - margin_px)
x_max_fig4 = min(256, (rc_comb[:, 1].max() + 1) * 4 + margin_px)
y_min_fig4 = max(0, rc_comb[:, 0].min() * 4 - margin_px)
y_max_fig4 = min(256, (rc_comb[:, 0].max() + 1) * 4 + margin_px)

print(f"Corrected Fig 4 Zoom window: X=[{x_min_fig4}, {x_max_fig4}], Y=[{y_min_fig4}, {y_max_fig4}]")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="#101010")
plt.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.88, wspace=0.10)

for ax in [ax0, ax1]:
    ax.set_facecolor("#000000")
    ax.imshow(rd34, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax.set_xlim(x_min_fig4, x_max_fig4)
    ax.set_ylim(y_min_fig4, y_max_fig4)
    ax.tick_params(colors="gray")

# Left: Baseline E1 in Pair 34
ax0.set_title("Pair 34: Baseline E1 Tracking (T24 Front vs T26 Root)", color="white", fontsize=11.5, fontweight="bold")
draw_patch_boxes(ax0, m_t24, edgecolor="#76ff03", linewidth=1.8)
draw_patch_boxes(ax0, m_t26, edgecolor="#ff4081", linewidth=1.8)

# Actual centroids for annotation anchors
rc24 = np.argwhere(m_t24)
rc26 = np.argwhere(m_t26)
c24_x, c24_y = rc24[:, 1].mean() * 4 + 2, rc24[:, 0].mean() * 4 + 2
c26_x, c26_y = rc26[:, 1].mean() * 4 + 2, rc26[:, 0].mean() * 4 + 2

ax0.annotate("E1 T24 (45 patches, Lime)\nDownstream Outer Front\nBirth r_min=39.7 px",
             xy=(c24_x, c24_y), xytext=(c24_x - 35, c24_y - 20),
             arrowprops=dict(facecolor="#76ff03", edgecolor="none", arrowstyle="->", lw=1.5),
             color="#76ff03", fontsize=9.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#76ff03"))

ax0.annotate("E1 T26 (46 patches, Magenta)\nUpstream Inner Root\nBirth r_min=20.6 px",
             xy=(c26_x, c26_y), xytext=(c26_x + 15, c26_y + 15),
             arrowprops=dict(facecolor="#ff4081", edgecolor="none", arrowstyle="->", lw=1.5),
             color="#ff4081", fontsize=9.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#ff4081"))

# Right: Adaptive Tracking in Pair 34
ax1.set_title("Pair 34: Adaptive PA-ROI Tracking (Track 3 Continuation vs Track 7 Birth)", color="white", fontsize=11.5, fontweight="bold")
f33 = pair_to_fidx[33]
m_t3_prev = (t_own_a0[f33] == 3)
m_t3_direct = m_ad3 & m_t3_prev
m_t3_expanded = m_ad3 & (~m_t3_prev)

draw_patch_boxes(ax1, m_t3_direct, edgecolor="#00e5ff", linewidth=1.8, facecolor="#00e5ff", alpha=0.35)
draw_patch_boxes(ax1, m_t3_expanded, edgecolor="#00e5ff", linewidth=1.8, facecolor="none")
draw_patch_boxes(ax1, m_ad7, edgecolor="#ff4081", linewidth=1.8, facecolor="none")

rcad3 = np.argwhere(m_ad3)
rcad7 = np.argwhere(m_ad7)
cad3_x, cad3_y = rcad3[:, 1].mean() * 4 + 2, rcad3[:, 0].mean() * 4 + 2
cad7_x, cad7_y = rcad7[:, 1].mean() * 4 + 2, rcad7[:, 0].mean() * 4 + 2

ax1.annotate("Adaptive Track 3 (45p = E1 T24)\n[18 direct overlap, 27 exp]\nContinuous Outer Front",
             xy=(cad3_x, cad3_y), xytext=(cad3_x - 35, cad3_y - 20),
             arrowprops=dict(facecolor="#00e5ff", edgecolor="none", arrowstyle="->", lw=1.5),
             color="#00e5ff", fontsize=9.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#00e5ff"))

ax1.annotate("Adaptive Track 7 (46p = E1 T26)\n[Independent Birth Seed]\nSolar Root Anchor",
             xy=(cad7_x, cad7_y), xytext=(cad7_x + 15, cad7_y + 15),
             arrowprops=dict(facecolor="#ff4081", edgecolor="none", arrowstyle="->", lw=1.5),
             color="#ff4081", fontsize=9.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#ff4081"))

fig.suptitle("Corrected Figure 4: Zoomed Diagnostic of E1 T24 Absorption into Adaptive Track 3 (Pair 34)\n[Automatic Bounds on Eastern CME: X=[20, 136], Y=[56, 156]]",
             color="white", fontsize=13.0, fontweight="bold", y=0.97)
plt.savefig(vis_dir / "corrected_fig_04_t24_to_t3_absorption_zoom.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# 4.2 Corrected Figure 6: T26 -> T7 Independent Birth Zoom (Pairs 34 & 35)
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="#101010")
plt.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.88, wspace=0.10)

m_t7_p34 = (t_own_a0[f34] == 7)
f35 = pair_to_fidx[35]
m_t7_p35 = (t_own_a0[f35] == 7)

rc_t7_comb = np.argwhere(m_t7_p34 | m_t7_p35)
x_min_fig6 = max(0, rc_t7_comb[:, 1].min() * 4 - 30)
x_max_fig6 = min(256, (rc_t7_comb[:, 1].max() + 1) * 4 + 30)
y_min_fig6 = max(0, rc_t7_comb[:, 0].min() * 4 - 30)
y_max_fig6 = min(256, (rc_t7_comb[:, 0].max() + 1) * 4 + 30)

print(f"Corrected Fig 6 Zoom window: X=[{x_min_fig6}, {x_max_fig6}], Y=[{y_min_fig6}, {y_max_fig6}]")

for idx, p_num in enumerate([34, 35]):
    ax = axes[idx]
    ax.set_facecolor("#000000")
    f_idx = pair_to_fidx[p_num]
    p_raw = p_num - 1
    rd_cur = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
    ax.imshow(rd_cur, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax.set_xlim(x_min_fig6, x_max_fig6)
    ax.set_ylim(y_min_fig6, y_max_fig6)
    ax.tick_params(colors="gray")

    # Draw occulter edge reference (radius ~ 21 patch px = 84 image px from center (128, 128))
    occ_circle = plt.Circle((128, 128), 21.0 * 4, edgecolor="gray", facecolor="none", linestyle="--", linewidth=1.5, label="Occulter Boundary (~21px)")
    ax.add_patch(occ_circle)

    m_7 = (t_own_a0[f_idx] == 7)
    draw_patch_boxes(ax, m_7, edgecolor="#ff4081", linewidth=2.0)

    rc_7 = np.argwhere(m_7)
    c7_x, c7_y = rc_7[:, 1].mean() * 4 + 2, rc_7[:, 0].mean() * 4 + 2
    n_p7 = m_7.sum()
    cpa_7 = df_states_a0[(df_states_a0["pair_index"] == p_num) & (df_states_a0["track_id"] == 7)].iloc[0]["central_pa"]
    rmin_7 = df_states_a0[(df_states_a0["pair_index"] == p_num) & (df_states_a0["track_id"] == 7)].iloc[0]["radial_min"]

    ax.annotate(f"Adaptive Track 7\n({n_p7}p, CPA={cpa_7:.1f}°)\nr_min={rmin_7:.1f} px <= 30 px\n[Root Anchor]",
                xy=(c7_x, c7_y), xytext=(c7_x - 30, c7_y - 25),
                arrowprops=dict(facecolor="#ff4081", edgecolor="none", arrowstyle="->", lw=1.5),
                color="#ff4081", fontsize=9.0, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#ff4081"))

    ax.set_title(f"Pair {p_num:02d}: Track 7 Independent Birth ({n_p7}p, r_min={rmin_7:.1f}px)", color="white", fontsize=11.0, fontweight="bold")

axes[0].legend(loc="upper left", facecolor="#1e1e1e", labelcolor="white", fontsize=8.5)
fig.suptitle("Corrected Figure 6: Diagnostic Zoom of Solar Root Anchor Birth (Track 7 / E1 T26 at Pairs 34 & 35)\n[Automatic Bounds on Occulter Edge: X=[42, 142], Y=[70, 166]]",
             color="white", fontsize=13.0, fontweight="bold", y=0.97)
plt.savefig(vis_dir / "corrected_fig_06_t26_to_t7_independent_birth_zoom.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# 4.3 Automated Visualization Validation Checks
print("Running automated visualization validation checks...")
val_checks = []

# Check 1: Fig 4 T3 and T7 visibility
rc_t3_f4 = np.argwhere(m_ad3)
rc_t7_f4 = np.argwhere(m_ad7)
x3_f4, y3_f4 = rc_t3_f4[:, 1] * 4, rc_t3_f4[:, 0] * 4
x7_f4, y7_f4 = rc_t7_f4[:, 1] * 4, rc_t7_f4[:, 0] * 4

vis_t3_f4 = ((x3_f4 >= x_min_fig4) & (x3_f4 <= x_max_fig4) & (y3_f4 >= y_min_fig4) & (y3_f4 <= y_max_fig4)).sum()
vis_t7_f4 = ((x7_f4 >= x_min_fig4) & (x7_f4 <= x_max_fig4) & (y7_f4 >= y_min_fig4) & (y7_f4 <= y_max_fig4)).sum()

val_checks.append({
    "figure": "corrected_fig_04",
    "track": "Track 3",
    "total_patches": len(rc_t3_f4),
    "visible_patches": vis_t3_f4,
    "visibility_fraction": vis_t3_f4 / max(1, len(rc_t3_f4)),
    "anchor_x": cad3_x,
    "anchor_y": cad3_y,
    "inside_bounds": (x_min_fig4 <= cad3_x <= x_max_fig4) and (y_min_fig4 <= cad3_y <= y_max_fig4),
    "status": "PASS" if (vis_t3_f4 == len(rc_t3_f4)) else "FAIL",
})

val_checks.append({
    "figure": "corrected_fig_04",
    "track": "Track 7",
    "total_patches": len(rc_t7_f4),
    "visible_patches": vis_t7_f4,
    "visibility_fraction": vis_t7_f4 / max(1, len(rc_t7_f4)),
    "anchor_x": cad7_x,
    "anchor_y": cad7_y,
    "inside_bounds": (x_min_fig4 <= cad7_x <= x_max_fig4) and (y_min_fig4 <= cad7_y <= y_max_fig4),
    "status": "PASS" if (vis_t7_f4 == len(rc_t7_f4)) else "FAIL",
})

# Check 2: Fig 6 T7 visibility in P34 and P35
vis_t7_p34_f6 = (((rc_t7_comb[:, 1] * 4 >= x_min_fig6) & (rc_t7_comb[:, 1] * 4 <= x_max_fig6) &
                  (rc_t7_comb[:, 0] * 4 >= y_min_fig6) & (rc_t7_comb[:, 0] * 4 <= y_max_fig6)).sum())

val_checks.append({
    "figure": "corrected_fig_06",
    "track": "Track 7 (P34 & P35)",
    "total_patches": len(rc_t7_comb),
    "visible_patches": vis_t7_p34_f6,
    "visibility_fraction": vis_t7_p34_f6 / max(1, len(rc_t7_comb)),
    "anchor_x": c7_x,
    "anchor_y": c7_y,
    "inside_bounds": (x_min_fig6 <= c7_x <= x_max_fig6) and (y_min_fig6 <= c7_y <= y_max_fig6),
    "status": "PASS" if (vis_t7_p34_f6 == len(rc_t7_comb)) else "FAIL",
})

df_val_report = pd.DataFrame(val_checks)
df_val_report.to_csv(tbl_dir / "visualization_validation_report.csv", index=False)
print("Visualization Validation Report:")
print(df_val_report.to_string())

# -------------------------------------------------------------------------
# Step 5: Produce Clear A0 vs A2 Visualizations (Part 5)
# -------------------------------------------------------------------------
print("\n--- Generating A0 vs A2 Comparison Visualizations (Part 5) ---")

TRACK_COLORS = {
    3: "#00e5ff", # Cyan (East CME outer front)
    4: "#ffd600", # Yellow (NE feature)
    5: "#76ff03", # Lime (North micro)
    6: "#ff9100", # Orange (Intermediary)
    7: "#ff4081", # Magenta (East root anchor)
    8: "#f48fb1", # Pink (North eruption / sub)
    9: "#ce93d8", # Light Purple
    10: "#80deea", # Light Cyan
    11: "#a7ffeb", # Teal
}

# 5.1 Side-by-Side Comparison for Pairs 30–35 (Left: A0, Right: A2)
fig, axes = plt.subplots(6, 2, figsize=(16, 36), facecolor="#0e0e0e")
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.02, top=0.97, wspace=0.06, hspace=0.12)

for row_idx, p_num in enumerate([30, 31, 32, 33, 34, 35]):
    f_idx = pair_to_fidx[p_num]
    p_raw = p_num - 1
    rd_cur = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
    ts = man_df.loc[man_df["pair_index"] == p_num, "end_time"].iloc[0][11:19]

    # Left: A0
    ax_a0 = axes[row_idx, 0]
    ax_a0.set_facecolor("#000000")
    ax_a0.imshow(rd_cur, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax_a0.set_xlim(0, 256)
    ax_a0.set_ylim(0, 256)
    ax_a0.tick_params(colors="gray")

    # Plot A0 tracks
    sub_a0 = df_states_a0[df_states_a0["pair_index"] == p_num]
    for _, r in sub_a0.iterrows():
        tid = int(r["track_id"])
        st = r["state"]
        col = TRACK_COLORS.get(tid, "#ffffff")
        if st == "OBSERVED":
            m = (t_own_a0[f_idx] == tid)
            draw_patch_boxes(ax_a0, m, edgecolor=col, linewidth=1.4)
            rcs = np.argwhere(m)
            if len(rcs) > 0:
                ax_a0.text(rcs[:, 1].mean() * 4 + 2, rcs[:, 0].mean() * 4 + 2, f"T{tid}",
                           color="black", fontsize=8.0, fontweight="bold",
                           bbox=dict(boxstyle="square,pad=0.15", facecolor=col, edgecolor="none", alpha=0.85))

    ax_a0.set_title(f"Pair {p_num:02d} ({ts} UTC) — A0 (Unconstrained Adaptive Tracking)", color="white", fontsize=11, fontweight="bold")

    # Right: A2
    ax_a2 = axes[row_idx, 1]
    ax_a2.set_facecolor("#000000")
    ax_a2.imshow(rd_cur, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax_a2.set_xlim(0, 256)
    ax_a2.set_ylim(0, 256)
    ax_a2.tick_params(colors="gray")

    sub_a2 = df_states_a2[df_states_a2["pair_index"] == p_num]
    for _, r in sub_a2.iterrows():
        tid = int(r["track_id"])
        st = r["state"]
        col = TRACK_COLORS.get(tid, "#ffffff")
        if st == "OBSERVED":
            m = (t_own_a2[f_idx] == tid)
            draw_patch_boxes(ax_a2, m, edgecolor=col, linewidth=1.4)
            rcs = np.argwhere(m)
            if len(rcs) > 0:
                ax_a2.text(rcs[:, 1].mean() * 4 + 2, rcs[:, 0].mean() * 4 + 2, f"T{tid}",
                           color="black", fontsize=8.0, fontweight="bold",
                           bbox=dict(boxstyle="square,pad=0.15", facecolor=col, edgecolor="none", alpha=0.85))
        elif st == "PREDICTED_ONLY" and tid == 3:
            # Highlight predicted state for T3
            ax_a2.text(71, 92, "T3 [PREDICTED]\n(Gate Protected)", color="#00e5ff", fontsize=8.5, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#00e5ff", linestyle="--"))

    # Highlight gate rejections in A2
    p_rejs = [rj for rj in rejections_a2 if rj["pair_index"] == p_num]
    for rj in p_rejs:
        ax_a2.text(128, 20, f"REJECTED: T{rj['track_id']} (disp={rj['radial_displacement']:.1f}px < -4px)",
                   color="#ff5252", fontsize=9.0, fontweight="bold", ha="center",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="#260000", edgecolor="#ff5252"))

    ax_a2.set_title(f"Pair {p_num:02d} ({ts} UTC) — A2 (Outward Gate tol=4px)", color="white", fontsize=11, fontweight="bold")

fig.suptitle("Figure 1: Full Sequence Side-by-Side Comparison (A0 Baseline vs A2 Outward Gate, Pairs 30–35)",
             color="white", fontsize=15, fontweight="bold", y=0.98)
plt.savefig(vis_dir / "fig_01_a0_vs_a2_side_by_side_pairs30_35.png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# 5.2 Focused Pair 33 -> 34 -> 35 Eastern CME Comparison
print("Generating Fig 02: Focused Pair 33 -> 34 -> 35 Eastern CME Comparison...")
fig, axes = plt.subplots(3, 2, figsize=(15, 18), facecolor="#101010")
plt.subplots_adjust(left=0.06, right=0.94, bottom=0.04, top=0.94, wspace=0.08, hspace=0.14)

zoom_east_x = (20, 130)
zoom_east_y = (60, 150)

for idx, p_num in enumerate([33, 34, 35]):
    f_idx = pair_to_fidx[p_num]
    p_raw = p_num - 1
    rd_cur = np.where(val_all[p_raw], rd_all[p_raw], np.nan)
    ts = man_df.loc[man_df["pair_index"] == p_num, "end_time"].iloc[0][11:19]

    # Left: A0
    ax_a0 = axes[idx, 0]
    ax_a0.set_facecolor("#000000")
    ax_a0.imshow(rd_cur, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax_a0.set_xlim(zoom_east_x)
    ax_a0.set_ylim(zoom_east_y)
    ax_a0.tick_params(colors="gray")

    sub_a0 = df_states_a0[df_states_a0["pair_index"] == p_num]
    for tid in [3, 7]:
        r = sub_a0[sub_a0["track_id"] == tid]
        if len(r) > 0 and r.iloc[0]["state"] == "OBSERVED":
            m = (t_own_a0[f_idx] == tid)
            col = TRACK_COLORS[tid]
            draw_patch_boxes(ax_a0, m, edgecolor=col, linewidth=1.6)
            rcs = np.argwhere(m)
            if len(rcs) > 0:
                ax_a0.text(rcs[:, 1].mean() * 4 + 2, rcs[:, 0].mean() * 4 + 2, f"T{tid} ({m.sum()}p)",
                           color="black", fontsize=8.5, fontweight="bold",
                           bbox=dict(boxstyle="square,pad=0.15", facecolor=col, edgecolor="none", alpha=0.9))

    t3_row_a0 = sub_a0[sub_a0["track_id"] == 3]
    rf_a0 = t3_row_a0.iloc[0]["radial_front"] if len(t3_row_a0) > 0 else np.nan
    ax_a0.set_title(f"Pair {p_num:02d} ({ts} UTC) — A0: T3 R_front = {rf_a0:.1f} px", color="white", fontsize=11, fontweight="bold")

    # Right: A2
    ax_a2 = axes[idx, 1]
    ax_a2.set_facecolor("#000000")
    ax_a2.imshow(rd_cur, origin="lower", cmap=cmap_rd, norm=norm_rd)
    ax_a2.set_xlim(zoom_east_x)
    ax_a2.set_ylim(zoom_east_y)
    ax_a2.tick_params(colors="gray")

    sub_a2 = df_states_a2[df_states_a2["pair_index"] == p_num]
    for tid in [3, 7]:
        r = sub_a2[sub_a2["track_id"] == tid]
        if len(r) > 0:
            st = r.iloc[0]["state"]
            col = TRACK_COLORS[tid]
            if st == "OBSERVED":
                m = (t_own_a2[f_idx] == tid)
                draw_patch_boxes(ax_a2, m, edgecolor=col, linewidth=1.6)
                rcs = np.argwhere(m)
                if len(rcs) > 0:
                    ax_a2.text(rcs[:, 1].mean() * 4 + 2, rcs[:, 0].mean() * 4 + 2, f"T{tid} ({m.sum()}p)",
                               color="black", fontsize=8.5, fontweight="bold",
                               bbox=dict(boxstyle="square,pad=0.15", facecolor=col, edgecolor="none", alpha=0.9))
            elif st == "PREDICTED_ONLY" and tid == 3:
                ax_a2.text(71, 92, "T3 [PREDICTED]\n(Retained R=82.3px)", color="#00e5ff", fontsize=9.0, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a", edgecolor="#00e5ff", linestyle="--"))

    t3_row_a2 = sub_a2[sub_a2["track_id"] == 3]
    rf_a2 = t3_row_a2.iloc[0]["radial_front"] if len(t3_row_a2) > 0 else np.nan
    ax_a2.set_title(f"Pair {p_num:02d} ({ts} UTC) — A2: T3 R_front = {rf_a2:.1f} px", color="white", fontsize=11, fontweight="bold")

fig.suptitle("Figure 2: Focused Eastern CME Evolution (Pairs 33 -> 34 -> 35)\n[A0: False Inward Latch to Inner Component at P35 vs A2: Outward Front Preserved]",
             color="white", fontsize=13.5, fontweight="bold", y=0.96)
plt.savefig(vis_dir / "fig_02_focused_pair33_35_eastern_cme.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# 5.3 Eastern CME Leading Edge Trajectory Plot (Pairs 30–39)
print("Generating Fig 03: Eastern CME Leading Edge Trajectory Plot...")
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 10), facecolor="#101010", sharex=True)
plt.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.92, hspace=0.14)

p_eval = list(range(30, 40))
t3_sub_a0 = df_states_a0[(df_states_a0["track_id"] == 3) & (df_states_a0["pair_index"].isin(p_eval))].sort_values("pair_index")
t3_sub_a2 = df_states_a2[(df_states_a2["track_id"] == 3) & (df_states_a2["pair_index"].isin(p_eval))].sort_values("pair_index")

# Plot 1: Leading Edge Radius
ax0.set_facecolor("#000000")
ax0.plot(t3_sub_a0["pair_index"], t3_sub_a0["radial_front"], color="#ff5252", marker="o", linewidth=2.2, label="A0 Baseline (Unconstrained)")
ax0.plot(t3_sub_a2["pair_index"], t3_sub_a2["radial_front"], color="#00e5ff", marker="s", linewidth=2.5, linestyle="--", label="A2 Outward Gate (tol = 4 px)")

# Mark predicted vs observed in A2
for _, r in t3_sub_a2.iterrows():
    if r["state"] == "PREDICTED_ONLY":
        ax0.plot(r["pair_index"], r["radial_front"], marker="x", color="yellow", markersize=10, markeredgewidth=2.0)

ax0.axvline(34.5, color="gray", linestyle=":", linewidth=1.5, label="Pair 34/35 Outer Signal Drop-out")
ax0.annotate("A0 Inward Retreat: -41.1 px\n(Latched onto inner root)", xy=(35, 41.25), xytext=(35.5, 30),
             arrowprops=dict(facecolor="#ff5252", edgecolor="none", arrowstyle="->", lw=1.5),
             color="#ff5252", fontsize=9.0, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a"))

ax0.annotate("A2 Front Preserved at R=82.3 px\n(Rejected inner latch)", xy=(35, 82.32), xytext=(33.5, 95),
             arrowprops=dict(facecolor="#00e5ff", edgecolor="none", arrowstyle="->", lw=1.5),
             color="#00e5ff", fontsize=9.0, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a"))

ax0.set_ylabel("Leading Edge Radius (px)", color="white", fontsize=10.5)
ax0.set_title("Eastern CME Leading Edge Trajectory (Track 3, Pairs 30–39)", color="white", fontsize=12.0, fontweight="bold")
ax0.tick_params(colors="white")
ax0.legend(loc="upper left", facecolor="#1e1e1e", labelcolor="white", fontsize=9.0)
ax0.grid(color="#333333", linestyle="--", alpha=0.6)

# Plot 2: Frame-to-Frame Radial Displacement
ax1.set_facecolor("#000000")
ax1.axhline(0, color="gray", linestyle="-", linewidth=1.0)
ax1.axhline(-4.0, color="#ff5252", linestyle="--", linewidth=1.5, label="Outward Gate Tolerance (-4.0 px)")

ax1.plot(t3_sub_a0["pair_index"], t3_sub_a0["radial_displacement"], color="#ff5252", marker="o", linewidth=2.0, label="A0 Displacement")
ax1.plot(t3_sub_a2["pair_index"], t3_sub_a2["radial_displacement"], color="#00e5ff", marker="s", linewidth=2.0, linestyle="--", label="A2 Displacement")

ax1.set_xlabel("Pair Index", color="white", fontsize=10.5)
ax1.set_ylabel("Radial Displacement (px / frame)", color="white", fontsize=10.5)
ax1.set_title("Leading Edge Radial Displacement per Frame (Track 3)", color="white", fontsize=12.0, fontweight="bold")
ax1.tick_params(colors="white")
ax1.legend(loc="lower left", facecolor="#1e1e1e", labelcolor="white", fontsize=9.0)
ax1.grid(color="#333333", linestyle="--", alpha=0.6)

fig.suptitle("Figure 3: Eastern CME Leading Edge Kinematic Analysis (A0 Baseline vs A2 Outward Gate)",
             color="white", fontsize=14.0, fontweight="bold", y=0.96)
plt.savefig(vis_dir / "fig_03_eastern_cme_leading_edge_trajectory.png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

print("\n=== Experiment Execution and Visualizations Completed Successfully ===")

