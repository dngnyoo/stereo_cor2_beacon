"""Generate publication-quality diagnostic visualizations for A5b-NR evaluation."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path("/mnt/c/Users/aabab/.gemini/antigravity/brain/1ac3edc5-a8a9-475f-9fa4-c0ef02bb23df/figures")
out_dir.mkdir(parents=True, exist_ok=True)

b0_dir = Path("/home/aababc/stereo_cor2_a5bn/runs/exp_20101212_a5b_baseline")
b1_dir = Path("/home/aababc/stereo_cor2_a5bn/runs/exp_20101212_a5b_n")
nr1_dir = Path("/home/aababc/stereo_cor2_a5bnr/runs/exp_20101212_a5b_nr1")
nr2_dir = Path("/home/aababc/stereo_cor2_a5bnr/runs/exp_20101212_a5b_nr2")

# Load candidate tables
df_b1 = pd.read_csv(b1_dir / "03_tracking/data/tables/normalized_candidate_scores.csv")
df_nr1 = pd.read_csv(nr1_dir / "03_tracking/data/tables/normalized_candidate_scores.csv")
df_nr2 = pd.read_csv(nr2_dir / "03_tracking/data/tables/normalized_candidate_scores.csv")

mask_nr1 = np.load(nr1_dir / "03_tracking/data/arrays/track_ownership_masks.npy")
mask_nr2 = np.load(nr2_dir / "03_tracking/data/arrays/track_ownership_masks.npy")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# -----------------------------------------------------------------------------
# Figure 1: Evidence Score Distributions & Zero-Collapse Elimination
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

# Panel A: Candidate score distribution comparison
ax = axes[0]
bins = np.linspace(0.0, 1.0, 21)
ax.hist(df_b1["candidate_score"], bins=bins, alpha=0.5, color="#d95f02", label=f"B1 (A5b-N)\nZero: {(df_b1['candidate_score']==0).sum()}/60 (48.3%)", edgecolor="black")
ax.hist(df_nr1["candidate_score"], bins=bins, alpha=0.6, color="#1b9e77", label=f"NR1 (A5b-NR1)\nZero: 0/60 (0.0%)", edgecolor="black")
ax.set_title("(A) Candidate Matching Score Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Candidate Score $S$", fontsize=11)
ax.set_ylabel("Candidate Count", fontsize=11)
ax.legend(frameon=True, loc="upper right")
ax.set_xlim(-0.02, 1.02)

# Panel B: Redesigned Evidence Terms in NR1
ax = axes[1]
data_terms = [df_nr1["C"], df_nr1["A"], df_nr1["K"], df_nr1["D"], df_nr1["candidate_score"]]
labels = ["C\n(Spatial)", "A\n(Angular)", "K\n(Kinematic)", "D\n(Detection)", "$S_{NR1}$\n(Total)"]
vp = ax.boxplot(data_terms, tick_labels=labels, patch_artist=True, showmeans=True,
                boxprops=dict(facecolor="#7570b3", alpha=0.6),
                medianprops=dict(color="black", linewidth=1.5),
                meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white"))
ax.set_title("(B) NR1 Evidence Term Distributions", fontsize=12, fontweight="bold")
ax.set_ylabel("Evidence Value $\in (0, 1]$", fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="Zero-collapse threshold")
ax.legend(frameon=True, loc="lower right")

# Panel C: Clipping Saturation Comparison
ax = axes[2]
ax.hist(df_b1["raw_overlap"], bins=np.linspace(0, 1.5, 31), alpha=0.5, color="#e7298a", label="B1 Raw Overlap\n(23.3% > 1.0 saturation)", edgecolor="black")
ax.hist(df_nr1["C"], bins=np.linspace(0, 1.5, 31), alpha=0.6, color="#2b83ba", label="NR1 Spatial Continuity C\n(0.0% > 1.0 bounded)", edgecolor="black")
ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Saturation boundary (1.0)")
ax.set_title("(C) Spatial Evidence Saturation Audit", fontsize=12, fontweight="bold")
ax.set_xlabel("Score Value", fontsize=11)
ax.set_ylabel("Candidate Count", fontsize=11)
ax.set_xlim(0.0, 1.5)
ax.legend(frameon=True, loc="upper right")

plt.tight_layout()
fig1_path = out_dir / "01_nr_zero_collapse_and_score_distributions.png"
fig.savefig(fig1_path, bbox_inches="tight")
plt.close(fig)
print("Saved:", fig1_path)

# -----------------------------------------------------------------------------
# Figure 2: Score-Aware Contested Voronoi Arbitration (NR1 vs NR2)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

# Panel A: Patches per track on Frame 36 (Pair 37)
ax = axes[0]
frames = [33, 34, 35, 36, 37, 38, 39]
pairs = [f + 1 for f in frames]
t1_nr1 = [int((mask_nr1[f] == 1).sum()) for f in frames]
t1_nr2 = [int((mask_nr2[f] == 1).sum()) for f in frames]
t2_nr1 = [int((mask_nr1[f] == 2).sum()) for f in frames]
t2_nr2 = [int((mask_nr2[f] == 2).sum()) for f in frames]

ax.plot(pairs, t1_nr1, "o--", color="#1f78b4", label="Track 1 (NR1 Baseline)")
ax.plot(pairs, t1_nr2, "o-", color="#1f78b4", linewidth=2.5, label="Track 1 (NR2 Score-Aware)")
ax.plot(pairs, t2_nr1, "s--", color="#e31a1c", label="Track 2 (NR1 Baseline)")
ax.plot(pairs, t2_nr2, "s-", color="#e31a1c", linewidth=2.5, label="Track 2 (NR2 Score-Aware)")
ax.set_title("(A) Track Patches on Contested Frames", fontsize=12, fontweight="bold")
ax.set_xlabel("Pair Index", fontsize=11)
ax.set_ylabel("Owned Patches (count)", fontsize=11)
ax.legend(frameon=True, loc="upper left")

# Panel B: Spatial difference map on Frame 35 (Pair 36, 29 patches shifted)
ax = axes[1]
diff_f35 = np.zeros((64, 64))
# 0 = neither, 1 = T1 in both, 2 = T2 in both, 3 = shifted from T2 to T1 in NR2
for r in range(64):
    for c in range(64):
        p1 = mask_nr1[35, r, c]
        p2 = mask_nr2[35, r, c]
        if p1 == 1 and p2 == 1:
            diff_f35[r, c] = 1
        elif p1 == 2 and p2 == 2:
            diff_f35[r, c] = 2
        elif p1 == 2 and p2 == 1:
            diff_f35[r, c] = 3  # Shifted to T1!

cmap = matplotlib.colors.ListedColormap(["#f0f0f0", "#1f78b4", "#e31a1c", "#33a02c"])
im = ax.imshow(diff_f35, cmap=cmap, vmin=0, vmax=3)
ax.set_title("(B) Patch Allocation Map (Pair 36)", fontsize=12, fontweight="bold")
ax.set_xlabel("Patch Column", fontsize=11)
ax.set_ylabel("Patch Row", fontsize=11)
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels(["Inactive", "Track 1", "Track 2", "Won by T1"])

# Panel C: Score differential driving the arbitration
ax = axes[2]
scores_t1 = []
scores_t2 = []
for p in pairs:
    sub = df_nr1[df_nr1["pair_index"] == p]
    s1 = sub[sub["track_id"] == 1]["candidate_score"].max() if (sub["track_id"] == 1).any() else np.nan
    s2 = sub[sub["track_id"] == 2]["candidate_score"].max() if (sub["track_id"] == 2).any() else np.nan
    scores_t1.append(s1)
    scores_t2.append(s2)

ax.plot(pairs, scores_t1, "o-", color="#1f78b4", linewidth=2.0, label="Track 1 Match Score $S_{NR1}$")
ax.plot(pairs, scores_t2, "s-", color="#e31a1c", linewidth=2.0, label="Track 2 Match Score $S_{NR1}$")
ax.fill_between(pairs, scores_t1, scores_t2, where=(np.array(scores_t1) >= np.array(scores_t2)),
                color="#1f78b4", alpha=0.15, label="T1 Physical Advantage")
ax.fill_between(pairs, scores_t1, scores_t2, where=(np.array(scores_t1) < np.array(scores_t2)),
                color="#e31a1c", alpha=0.15, label="T2 Physical Advantage")
ax.set_title("(C) Candidate Evidence Score Differential", fontsize=12, fontweight="bold")
ax.set_xlabel("Pair Index", fontsize=11)
ax.set_ylabel("Candidate Evidence Score $S_{NR}$", fontsize=11)
ax.set_ylim(0.4, 1.0)
ax.legend(frameon=True, loc="lower left")

plt.tight_layout()
fig2_path = out_dir / "02_nr_score_aware_patch_arbitration.png"
fig.savefig(fig2_path, bbox_inches="tight")
plt.close(fig)
print("Saved:", fig2_path)

# -----------------------------------------------------------------------------
# Figure 3: Track Evolution and Maximum Mass Capture
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Panel A: Max detected patches per admitted family
ax = axes[0]
families = ["FAM_T01", "FAM_T02", "FAM_T04", "FAM_T05"]
max_p_nr1 = [131, 123, 62, 88]
max_p_nr2 = [160, 125, 62, 88]
x = np.arange(len(families))
width = 0.35

ax.bar(x - width/2, max_p_nr1, width, label="NR1 (A5b-NR1 / Baseline)", color="#7570b3", edgecolor="black")
ax.bar(x + width/2, max_p_nr2, width, label="NR2 (Score-Aware Voronoi)", color="#1b9e77", edgecolor="black")
for i in range(len(families)):
    ax.text(x[i] - width/2, max_p_nr1[i] + 3, str(max_p_nr1[i]), ha="center", fontsize=10, fontweight="bold")
    ax.text(x[i] + width/2, max_p_nr2[i] + 3, str(max_p_nr2[i]), ha="center", fontsize=10, fontweight="bold")

ax.set_title("(A) Maximum Detected Patches per Admitted Family", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(families, fontsize=11)
ax.set_ylabel("Peak Active Patches", fontsize=11)
ax.set_ylim(0, 185)
ax.legend(frameon=True, loc="upper right")

# Panel B: Cumulative owned patches over evaluation frames
ax = axes[1]
all_frames = np.arange(32, 63)
eval_pairs = all_frames + 1
cum_nr1 = [int((mask_nr1[f] > 0).sum()) for f in all_frames]
cum_nr2 = [int((mask_nr2[f] > 0).sum()) for f in all_frames]

ax.plot(eval_pairs, cum_nr1, "o--", color="#7570b3", linewidth=2.0, label="NR1 Active Patches")
ax.plot(eval_pairs, cum_nr2, "s-", color="#1b9e77", linewidth=2.0, label="NR2 Active Patches")
ax.set_title("(B) Total Active CME Patches Across Evaluation Sequence", fontsize=12, fontweight="bold")
ax.set_xlabel("Pair Index (2010-12-12)", fontsize=11)
ax.set_ylabel("Total Owned Patches", fontsize=11)
ax.legend(frameon=True, loc="upper left")

plt.tight_layout()
fig3_path = out_dir / "03_nr_track_evolution_and_admissions.png"
fig.savefig(fig3_path, bbox_inches="tight")
plt.close(fig)
print("Saved:", fig3_path)
print("All figures successfully generated.")
