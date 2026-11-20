#!/usr/bin/env python3
"""Generate controlled comparison visualizations for A5b vs A5b-N on 2010-12-12."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup output directory
out_dir = Path("visualizations/a5b_vs_a5bn")
out_dir.mkdir(parents=True, exist_ok=True)

# Load data
b0_dir = Path("runs/exp_20101212_a5b_baseline")
b1_dir = Path("runs/exp_20101212_a5b_n")

df_b0 = pd.read_csv(b0_dir / "03_tracking/data/tables/live_tracking_states.csv")
df_b1 = pd.read_csv(b1_dir / "03_tracking/data/tables/live_tracking_states.csv")
df_diag = pd.read_csv(b1_dir / "03_tracking/data/tables/normalized_candidate_scores.csv")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
})

# ==============================================================================
# Figure 1: Score Distributions and Zero Collapse
# ==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1a. Raw A5b Score vs Normalized A5b-N Score
ax = axs[0, 0]
scatter = ax.scatter(
    df_diag["raw_a5b_score"],
    df_diag["candidate_score"],
    c=df_diag["selected_rejected_outcome"].map({"SELECTED": "#1f77b4", "REJECTED": "#d62728"}),
    s=45,
    alpha=0.75,
    edgecolors="k",
    linewidths=0.5,
)
ax.set_title("(a) Additive Score ($S_{A5b}$) vs Multiplicative Score ($S_N$)")
ax.set_xlabel("Original Additive Score $S_{A5b} = 4O + 2A + 6K + 3F$")
ax.set_ylabel("Normalized Geometric Mean $S_N = (O_N A_N K_N F_N)^{1/4}$")
ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
# Annotation for zero-collapse
n_zeros = (df_diag["candidate_score"] == 0).sum()
ax.annotate(
    f"48.3% of candidates (29/60)\ncollapse to $S_N = 0.0$\n(Single factor vanishing)",
    xy=(8, 0), xytext=(4, 0.4),
    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2ae", alpha=0.9),
)
ax.set_ylim(-0.05, 1.05)

# 1b. Histogram of candidate scores
ax = axs[0, 1]
ax.hist(df_diag["candidate_score"], bins=20, range=(0, 1), color="#2ca02c", edgecolor="black", alpha=0.7)
ax.set_title("(b) Distribution of A5b-N Candidate Scores ($S_N$)")
ax.set_xlabel("Normalized Score $S_N$")
ax.set_ylabel("Candidate Count")
ax.set_xlim(-0.02, 1.02)

# 1c. Factor Breakdown (O, A, K, F)
ax = axs[1, 0]
factors_df = df_diag[["O", "A", "K", "F"]]
factors_df.boxplot(ax=ax, patch_artist=True,
                   boxprops=dict(facecolor="#aec7e8", color="navy"),
                   medianprops=dict(color="red", linewidth=2))
ax.set_title("(c) Normalized Component Score Distributions ($O_N, A_N, K_N, F_N$)")
ax.set_ylabel("Score Value [0, 1]")
ax.set_ylim(-0.05, 1.05)

# 1d. Factor Vanishing Rate
ax = axs[1, 1]
zero_counts = [
    (df_diag["O"] == 0).sum(),
    (df_diag["A"] == 0).sum(),
    (df_diag["K"] == 0).sum(),
    (df_diag["F"] == 0).sum(),
    (df_diag["candidate_score"] == 0).sum(),
]
bars = ax.bar(["$O=0$", "$A=0$", "$K=0$", "$F=0$", "$S_N=0$ Total"], zero_counts,
       color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"], edgecolor="black")
ax.set_title("(d) Zero-Score Incidence by Factor")
ax.set_ylabel("Number of Candidates")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, 35)

plt.tight_layout()
fig_path_1 = out_dir / "01_score_distributions_and_collapse.png"
plt.savefig(fig_path_1, dpi=200)
plt.close()

# ==============================================================================
# Figure 2: Candidate Ranking Flips and Multi-Candidate Evaluations
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 2a. Candidate Ranking Scatter
ax1.scatter(
    df_diag["raw_a5b_score"],
    df_diag["candidate_score"],
    c=df_diag["candidate_rank"],
    cmap="viridis",
    s=60,
    edgecolors="black",
    linewidths=0.5,
)
ax1.set_title("Candidate Evaluation: Score vs Rank")
ax1.set_xlabel("Original Additive Score $S_{A5b}$")
ax1.set_ylabel("Normalized Multiplicative Score $S_N$")

# 2b. Case study of Ranking Flips
flip_cases = [
    {"frame": 36, "track": 1, "comp": 1, "a5b": 9.81, "sn": 0.418, "rank_a5b": 1, "rank_sn": 1, "sel": "SELECTED"},
    {"frame": 36, "track": 1, "comp": 2, "a5b": 4.72, "sn": 0.000, "rank_a5b": 3, "rank_sn": 2, "sel": "SELECTED"},
    {"frame": 36, "track": 1, "comp": 3, "a5b": 8.75, "sn": 0.000, "rank_a5b": 2, "rank_sn": 3, "sel": "SELECTED"},
    {"frame": 48, "track": 2, "comp": 4, "a5b": 13.56, "sn": 0.742, "rank_a5b": 1, "rank_sn": 1, "sel": "SELECTED"},
    {"frame": 48, "track": 2, "comp": 2, "a5b": 5.09, "sn": 0.000, "rank_a5b": 3, "rank_sn": 2, "sel": "SELECTED"},
    {"frame": 48, "track": 2, "comp": 3, "a5b": 7.39, "sn": 0.000, "rank_a5b": 2, "rank_sn": 3, "sel": "SELECTED"},
]
df_flips = pd.DataFrame(flip_cases)
y_pos = np.arange(len(df_flips))
width = 0.35

ax2.barh(y_pos - width/2, df_flips["rank_a5b"], width, label="A5b Additive Rank", color="#ff7f0e", edgecolor="k")
ax2.barh(y_pos + width/2, df_flips["rank_sn"], width, label="A5b-N Multiplicative Rank", color="#1f77b4", edgecolor="k")
labels = [f"F{r.frame} T{r.track} C{r.comp}" for _, r in df_flips.iterrows()]
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.set_xlabel("Priority Ranking within Track (1 = Highest Priority)")
ax2.set_title("Rank Inversion Cases (F36 & F48)\n(Both accepted: No mask divergence)")
ax2.set_xlim(0, 4)
ax2.legend()

plt.tight_layout()
fig_path_2 = out_dir / "02_candidate_ranking_flips.png"
plt.savefig(fig_path_2, dpi=200)
plt.close()

# ==============================================================================
# Figure 3: Track Lifecycle and Ownership Mask Invariance
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# 3a. Lifecycles
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for tid in sorted(df_b0["track_id"].unique()):
    sub0 = df_b0[df_b0["track_id"] == tid].sort_values("pair_index")
    sub1 = df_b1[df_b1["track_id"] == tid].sort_values("pair_index")
    color = colors[tid % 10]
    ax1.plot(sub0["pair_index"], [tid - 0.08]*len(sub0), "o-", color=color, markersize=5, label=f"Track {tid} (A5b)" if tid == 1 else "")
    ax1.plot(sub1["pair_index"], [tid + 0.08]*len(sub1), "s--", color=color, markersize=4, label=f"Track {tid} (A5b-N)" if tid == 1 else "")

ax1.set_title("(a) Track Lifecycles: Baseline A5b vs Experimental A5b-N (100% Identical)")
ax1.set_ylabel("Track ID")
ax1.set_yticks(range(1, 9))
ax1.legend(["A5b Baseline", "A5b-N Normalized"])

# 3b. Patches per track over time
for tid in sorted(df_b0["track_id"].unique()):
    sub0 = df_b0[(df_b0["track_id"] == tid) & (df_b0["state"] == "OBSERVED")].sort_values("pair_index")
    color = colors[tid % 10]
    ax2.plot(sub0["pair_index"], sub0["num_patches"], "o-", color=color, label=f"Track {tid}")

ax2.set_title("(b) Owned Active Patches per Track over Time")
ax2.set_xlabel("Observation Pair Index (2010-12-12)")
ax2.set_ylabel("Owned Patches (64x64 grid4)")
ax2.legend(ncol=4, loc="upper right")

plt.tight_layout()
fig_path_3 = out_dir / "03_track_lifecycles_and_patches.png"
plt.savefig(fig_path_3, dpi=200)
plt.close()

# ==============================================================================
# Figure 4: Saturation and Normalization Geometry Analysis
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 4a. Overlap saturation
raw_ov = df_diag["raw_overlap"]
clipped_ov = df_diag["O"]
ax1.scatter(raw_ov, clipped_ov, color="#2b5c8f", s=45, alpha=0.8, edgecolors="k")
ax1.plot([0, 1], [0, 1], "k--", label="Identity (unclipped)")
ax1.plot([1, max(raw_ov)], [1, 1], "r-", linewidth=2, label="Saturation (clipped to 1.0)")
ax1.set_title("(a) Overlap Score Clipping Saturation ($O_N = \operatorname{clip}(O, 0, 1)$)")
ax1.set_xlabel("Raw Overlap Score $O$ (includes 8-connected dilation)")
ax1.set_ylabel("Normalized Overlap $O_N$")
ax1.legend()

# 4b. Front bonus vs Leading front radius
r_fronts = df_diag["F"] * 122.5
f_bonus_a5b = df_diag["raw_front_bonus"]
f_bonus_a5bn = df_diag["F"]
ax2.plot(r_fronts, f_bonus_a5b, "o", color="#ff7f0e", alpha=0.7, label="A5b Front Bonus ($r / 100$)")
ax2.plot(r_fronts, f_bonus_a5bn, "s", color="#1f77b4", alpha=0.7, label="A5b-N Front Factor ($r / 122.5$)")
ax2.set_title("(b) Leading-Front Radial Extent Comparison ($R_{FOV} = 122.5$ px)")
ax2.set_xlabel("Leading Front Radius $r_{front}$ (px)")
ax2.set_ylabel("Front Term Value")
ax2.legend()

plt.tight_layout()
fig_path_4 = out_dir / "04_saturation_and_radial_geometry.png"
plt.savefig(fig_path_4, dpi=200)
plt.close()

print(f"Generated 4 comparison figures in {out_dir}")
