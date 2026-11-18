"""Compare three P16 tile variance measures across all saved frames.

Input from a fresh p16_spatial_diagnostics run. No QC or tracking changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--highlight-pairs", type=int, nargs="+", default=[99,100,110,111,112,137])
    args = p.parse_args()
    df = pd.read_csv(args.metrics).sort_values("pair_index")
    metrics = [
        ("p16_tile_occupancy_variance", "Unweighted occupancy variance"),
        ("p16_weighted_tile_variance", "Weighted variance (absolute)"),
        ("p16_normalized_weighted_tile_variance", "Normalized weighted variance (CV squared)"),
    ]
    missing = [key for key,_ in metrics if key not in df.columns]
    if missing:
        raise ValueError(f"Missing metrics: {missing}. Re-run p16_spatial_diagnostics to calculate them.")
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    cols = ["pair_index","detection_status","p16_active_fraction"] + [k for k,_ in metrics]
    df[cols].to_csv(out/"three_variances_all_frames.csv",index=False)
    print("\nThree P16 variances, all frames:")
    print(df[cols].round(5).to_string(index=False))
    print("\nHighlighted frames:")
    print(df.loc[df.pair_index.isin(args.highlight_pairs), cols].round(5).to_string(index=False))

    fig, axes = plt.subplots(3,1,figsize=(16,12),sharex=True,constrained_layout=True)
    global_qc = df.detection_status.eq("GLOBAL_ACTIVATION")
    for ax,(key,label) in zip(axes,metrics):
        ax.plot(df.pair_index,df[key],color="#8c95a2",linewidth=1,alpha=.7)
        for mask,name,color,marker in (
            (~global_qc,"Other","#2677a3","o"),
            (global_qc,"GLOBAL_ACTIVATION","#d17d38","x"),
        ):
            ax.scatter(df.loc[mask,"pair_index"],df.loc[mask,key],
                       color=color,label=name,marker=marker,s=38)
        for pair in args.highlight_pairs:
            selected = df.loc[df.pair_index.eq(pair)]
            if selected.empty or not np.isfinite(float(selected.iloc[0][key])):
                continue
            yy = float(selected.iloc[0][key])
            ax.annotate(str(pair),(pair,yy),xytext=(4,5),
                        textcoords="offset points",fontsize=9)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel("Pair index")
    fig.savefig(out/"three_variances_timeline.png",dpi=160)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    for ax,(key,label) in zip(axes,metrics):
        for mask,name,color,marker in (
            (~global_qc,"Other","#2677a3","o"),
            (global_qc,"GLOBAL_ACTIVATION","#d17d38","x"),
        ):
            ax.scatter(df.loc[mask,"p16_active_fraction"],df.loc[mask,key],
                       color=color,label=name,marker=marker,s=38)
        for pair in args.highlight_pairs:
            selected = df.loc[df.pair_index.eq(pair)]
            if selected.empty or not np.isfinite(float(selected.iloc[0][key])):
                continue
            rr = selected.iloc[0]
            ax.annotate(str(pair),(rr.p16_active_fraction,rr[key]),
                        xytext=(4,5),textcoords="offset points",fontsize=9)
        ax.set(xlabel="P16 active fraction",ylabel=label)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(out/"three_variances_vs_active_fraction.png",dpi=160)
    plt.close(fig)
    print("\nSaved:",out)


if __name__ == "__main__":
    main()
