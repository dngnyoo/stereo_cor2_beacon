"""All-frame review of saved P16 spatial metrics (no detector or QC changes).

Usage:
  python -m cme_pipeline.p16_spatial_review \
    --metrics runs/20240510_d4_p16_spatial/p16_spatial_metrics.csv \
    --output-dir runs/20240510_d4_p16_review
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def review(metrics: str | Path, output_dir: str | Path, *, flagged_pairs=(99, 100, 110),
           reference_pairs=(111, 112)):
    df = pd.read_csv(metrics).sort_values("pair_index").copy()
    required = ("pair_index", "detection_status", "p16_weighted_tile_variance",
                "p16_active_fraction", "p16_excess_active_p95",
                "p16_clipped_excess_moran")
    missing = [v for v in required if v not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not df.pair_index.is_unique:
        raise ValueError("Duplicate pair_index")
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)

    df["manual_review"] = "UNREVIEWED"
    df.loc[df.pair_index.isin(flagged_pairs), "manual_review"] = "REVIEW_REJECT"
    df.loc[df.pair_index.isin(reference_pairs), "manual_review"] = "REFERENCE_ONLY"
    # Marks the user's example frames, NOT ground-truth CME labels.
    by_variance = df.sort_values("p16_weighted_tile_variance", na_position="last")
    by_variance.to_csv(out / "frames_ranked_by_weighted_variance.csv", index=False)

    cols = ["pair_index", "detection_status", "manual_review", "p16_active_fraction",
            "p16_weighted_tile_variance", "p16_excess_active_p95",
            "p16_clipped_excess_moran"]
    print("\nAll frames, sorted by ascending weighted grid variance:")
    print(by_variance[cols].round(4).to_string(index=False))

    fig, ax = plt.subplots(3, 1, figsize=(17, 12), sharex=True, constrained_layout=True)
    pairs = df.pair_index.to_numpy()
    var = df.p16_weighted_tile_variance.to_numpy()
    rejected = df.detection_status.eq("GLOBAL_ACTIVATION").to_numpy()

    ax[0].plot(pairs, var, color="#748398", linewidth=1, alpha=.75)
    ax[0].scatter(pairs[~rejected], var[~rejected], color="#356EA3", label="D4 other", s=42)
    ax[0].scatter(pairs[rejected], var[rejected], color="#D77B35", marker="x",
                  label="D4 GLOBAL_ACTIVATION", s=56)
    ax[0].set(ylabel="Weighted grid variance", title="P16 weighted variance, all frames")
    ax[0].legend(loc="upper right")
    ax[1].plot(pairs, df.p16_active_fraction, marker="o", ms=3, color="#557D97")
    ax[1].set(ylabel="P16 active fraction", ylim=(-.05, 1.05))
    ax[2].plot(pairs, df.p16_excess_active_p95, marker="o", ms=3, color="#8A6EAD")
    ax[2].set(ylabel="P16 excess P95", xlabel="Pair index")
    for p in flagged_pairs:
        row = df[df.pair_index.eq(p)]
        if len(row):
            for axis in ax:
                axis.axvline(p, color="#C7483B", ls=":", alpha=.55)
            ax[0].annotate(str(p), (p, float(row.p16_weighted_tile_variance.iloc[0])),
                           xytext=(3, 8), textcoords="offset points", fontsize=9)
    for p in reference_pairs:
        row = df[df.pair_index.eq(p)]
        if len(row):
            ax[0].annotate(str(p), (p, float(row.p16_weighted_tile_variance.iloc[0])),
                           xytext=(3, 8), textcoords="offset points", fontsize=9)
    ax[0].grid(alpha=.2)
    ax[1].grid(alpha=.2)
    ax[2].grid(alpha=.2)
    fig.savefig(out/"all_frame_variance_timeline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for status, group in df.groupby("detection_status"):
        ax[0].scatter(group.p16_active_fraction, group.p16_weighted_tile_variance,
                      label=status, s=40, alpha=.8)
        ax[1].scatter(group.p16_excess_active_p95, group.p16_weighted_tile_variance,
                      label=status, s=40, alpha=.8)
    for p in tuple(flagged_pairs) + tuple(reference_pairs):
        row = df[df.pair_index.eq(p)]
        if row.empty:
            continue
        r = row.iloc[0]
        color = "#C7483B" if p in flagged_pairs else "#367D55"
        for axis,xval in [(ax[0],r.p16_active_fraction),
                          (ax[1],r.p16_excess_active_p95)]:
            axis.scatter([xval], [r.p16_weighted_tile_variance],
                         s=155, facecolors="none", edgecolors=color, linewidths=2)
            axis.annotate(str(p), (xval,r.p16_weighted_tile_variance),
                          xytext=(4,5), textcoords="offset points", fontsize=9)
    ax[0].set(xlabel="P16 active fraction", ylabel="Weighted grid variance")
    ax[1].set(xlabel="P16 excess active P95", ylabel="Weighted grid variance")
    for axis in ax:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(out/"all_frame_variance_scatter.png", dpi=160)
    plt.close(fig)

    # Exploratory threshold sweep. No arbitrary threshold is applied as a QC rule.
    unique = np.sort(df.p16_weighted_tile_variance.dropna().unique())
    cuts = np.r_[0., (unique[:-1] + unique[1:])/2, unique[-1] + 1e-9] if len(unique) else np.array([])
    rows = []
    for cut in cuts:
        selected = df.p16_weighted_tile_variance.le(cut)
        rows.append({
            "variance_cutoff_le":cut,
            "n_flagged":int(selected.sum()),
            "user_review_reject_flagged":int((selected & df.manual_review.eq("REVIEW_REJECT")).sum()),
            "reference_only_flagged":int((selected & df.manual_review.eq("REFERENCE_ONLY")).sum()),
            "existing_global_activation_flagged":int((selected & df.detection_status.eq("GLOBAL_ACTIVATION")).sum()),
            "other_unreviewed_flagged":int((selected & df.manual_review.eq("UNREVIEWED")).sum()),
        })
    pd.DataFrame(rows).to_csv(out/"exploratory_variance_cutoff_sweep.csv", index=False)
    print("\nSaved review to:", out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--flagged-pairs", type=int, nargs="+", default=[99, 100, 110])
    p.add_argument("--reference-pairs", type=int, nargs="+", default=[111, 112])
    a = p.parse_args()
    review(a.metrics, a.output_dir,
           flagged_pairs=a.flagged_pairs, reference_pairs=a.reference_pairs)


if __name__ == "__main__":
    main()
