"""Plots synthetic validation suite summary matrix and performance comparisons."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

res_path = root / "experiments/gaussian_tracking/results/synthetic_validation_results.csv"
df = pd.read_csv(res_path)

out_dir = root / "experiments/gaussian_tracking/visualizations/synthetic"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Performance Heatmap / Grid (Scenario vs Variant)
scenarios = df[["scenario_id", "scenario_name"]].drop_duplicates().sort_values("scenario_id")
scenario_labels = [f"S{r.scenario_id}: {r.scenario_name}" for _, r in scenarios.iterrows()]
variants = ["G0", "G1", "G2", "G3"]

status_matrix = np.zeros((len(scenarios), len(variants)))
for _, row in df.iterrows():
    s_idx = row["scenario_id"] - 1
    v_idx = variants.index(row["variant"])
    # 2: PASS, 1: WARN, 0: FAIL
    val = 2 if row["status"] == "PASS" else (1 if row["status"] == "WARN" else 0)
    status_matrix[s_idx, v_idx] = val

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.4, 1.0]})

# Heatmap
cmap = plt.cm.colors.ListedColormap(["#d9534f", "#f0ad4e", "#5cb85c"])  # red, orange, green
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

im = ax1.imshow(status_matrix, cmap=cmap, norm=norm, aspect="auto")
ax1.set_xticks(np.arange(len(variants)))
ax1.set_xticklabels(["G0 (Raw)", "G1 (Spatial)", "G2 (Temporal Mem)", "G3 (Motion-Aware)"], fontsize=11, fontweight="bold")
ax1.set_yticks(np.arange(len(scenarios)))
ax1.set_yticklabels(scenario_labels, fontsize=10)
ax1.set_title("Synthetic Test Suite Results (10 Scenarios)", fontsize=13, fontweight="bold", pad=12)

# Text annotations in heatmap
status_texts = {2: "PASS", 1: "WARN", 0: "FAIL"}
for i in range(len(scenarios)):
    for j in range(len(variants)):
        val = int(status_matrix[i, j])
        ax1.text(j, i, status_texts[val], ha="center", va="center", color="white" if val != 1 else "black", fontweight="bold", fontsize=10)

# Colorbar / Legend
cbar = fig.colorbar(im, ax=ax1, ticks=[0, 1, 2], orientation="horizontal", pad=0.08, shrink=0.7)
cbar.ax.set_xticklabels(["FAIL", "WARN", "PASS"], fontsize=10, fontweight="bold")

# Bar Chart: Scenario Success Rate per Variant
pass_counts = [(status_matrix[:, j] == 2).sum() for j in range(len(variants))]
warn_counts = [(status_matrix[:, j] == 1).sum() for j in range(len(variants))]
fail_counts = [(status_matrix[:, j] == 0).sum() for j in range(len(variants))]

x = np.arange(len(variants))
width = 0.55
ax2.bar(x, pass_counts, width, label="PASS", color="#5cb85c", edgecolor="black")
ax2.bar(x, warn_counts, width, bottom=pass_counts, label="WARN", color="#f0ad4e", edgecolor="black")
ax2.bar(x, fail_counts, width, bottom=[p + w for p, w in zip(pass_counts, warn_counts)], label="FAIL", color="#d9534f", edgecolor="black")

ax2.set_xticks(x)
ax2.set_xticklabels(["G0\nRaw", "G1\nSpatial", "G2\nTemporal", "G3\nMotion-Aware"], fontsize=11, fontweight="bold")
ax2.set_ylabel("Scenario Count (out of 10)", fontsize=11)
ax2.set_title("Variant Pass/Fail Breakdown", fontsize=13, fontweight="bold", pad=12)
ax2.set_ylim(0, 11)
ax2.legend(loc="upper right", framealpha=0.9)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

for i in range(len(variants)):
    ax2.text(i, pass_counts[i] / 2, f"{pass_counts[i]}/10", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

plt.tight_layout()
plot_path = out_dir / "synthetic_suite_matrix.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"Saved synthetic validation plot to {plot_path}")
