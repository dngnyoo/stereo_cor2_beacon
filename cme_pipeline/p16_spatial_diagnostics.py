"""Read-only P16 spatial QC experiment on saved C3/D4 observations.

P16 score is already P95(abs(pixel Z)); this includes positive and negative
pixel deviations. Compare raw-score weights and positive threshold exceedance,
without changing detection, QC, or tracking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label

from cme_pipeline.live_detection import _scores, _smooth_and_z


def moran_i(values, valid):
    """Unstandardized binary 4-neighbour weights, only valid-valid edges."""
    v = np.asarray(values, dtype=float)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(v)
    x = v[mask]
    if len(x) < 3 or np.var(x) <= 1e-14:
        return np.nan
    centered = np.zeros_like(v)
    centered[mask] = x - x.mean()
    n = len(x)
    num = 0.
    edges = 0
    for sl1, sl2 in ((np.s_[:, :-1], np.s_[:, 1:]),
                     (np.s_[:-1, :], np.s_[1:, :])):
        connected = mask[sl1] & mask[sl2]
        num += float(np.sum((centered[sl1] * centered[sl2])[connected]))
        edges += int(connected.sum())
    if not edges:
        return np.nan
    # Both directions of each undirected edge cancel between numerator/W.
    return float((n / edges) * num / np.sum((x - x.mean()) ** 2))


def spatial_metrics(scores, valid, t24, *, tile_size=4, clip_quantile=95.):
    """Return raw-absolute-Z-score, excess-weight and binary spatial metrics."""
    accepted = np.asarray(valid, bool) & np.isfinite(scores)
    active = accepted & np.isfinite(t24) & (scores > t24)
    n_valid, n_active = int(accepted.sum()), int(active.sum())
    raw = np.where(active, scores, 0.)
    excess = np.where(active, scores - t24, 0.) if np.isfinite(t24) else np.zeros_like(scores)
    excess = np.maximum(excess, 0.)
    positive = excess[active]
    cap = float(np.percentile(positive, clip_quantile)) if len(positive) else 0.
    clipped = np.minimum(excess, cap)
    tiled = []
    occupancy = []
    for y in range(0, 16, tile_size):
        for x in range(0, 16, tile_size):
            region = np.s_[y:y+tile_size, x:x+tile_size]
            vv = accepted[region]
            if vv.sum() < 4:
                continue
            occupancy.append(float(active[region].sum() / vv.sum()))
            tiled.append(float(clipped[region][vv].mean()))
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labs, ncomp = label(active, structure=structure)
    sizes = np.bincount(labs[active], minlength=ncomp+1)[1:]
    return {
        "n_valid_p16": n_valid,
        "n_active_p16": n_active,
        "p16_active_fraction": n_active/n_valid if n_valid else np.nan,
        "p16_score_active_p95": float(np.percentile(raw[active], 95)) if n_active else np.nan,
        "p16_excess_active_p95": float(np.percentile(positive, 95)) if n_active else np.nan,
        "p16_excess_weight_cap": cap,
        "p16_binary_moran": moran_i(active.astype(float), accepted),
        "p16_raw_weighted_moran": moran_i(raw, accepted),
        "p16_excess_moran": moran_i(excess, accepted),
        "p16_clipped_excess_moran": moran_i(clipped, accepted),
        "p16_tile_occupancy_variance": float(np.var(occupancy)) if len(occupancy) > 1 else np.nan,
        "p16_weighted_tile_variance": float(np.var(tiled)) if len(tiled) > 1 else np.nan,
        "p16_largest_component_fraction": float(sizes.max()/n_active) if n_active else np.nan,
    }, accepted, active, raw, clipped


def run(detection_dir, output_dir, *, date=None, focus_pair=99, tile_size=4):
    det, out = Path(detection_dir).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    if tile_size < 1 or 16 % tile_size:
        raise ValueError("tile-size must divide 16")
    meta = json.loads((det/"detection_metadata.json").read_text())
    if meta["detector"] not in ("c3", "d4"):
        raise ValueError("C3/D4 only")
    pre = Path(meta["input_dir"]).resolve()
    summary = pd.read_csv(det/"frame_detection_summary.csv")
    manifest = pd.read_csv(pre/"01_rd_manifest.csv")
    rd = np.load(pre/"north_up_RD_DNs_per_second.npy", mmap_mode="r")
    masks = np.load(pre/"north_up_pair_valid_masks.npy", mmap_mode="r")
    if not (len(summary)==len(manifest)==len(rd)==len(masks)
            and np.array_equal(summary.pair_index.astype(int),manifest.pair_index.astype(int))):
        raise ValueError("Manifest mismatch")
    day = date or str(meta["requested_date"])
    rows, grids = [], {}
    for i, item in manifest.iterrows():
        if pd.to_datetime(item.end_time, utc=True).strftime("%Y%m%d") != day:
            continue
        pixels = np.asarray(rd[i], dtype=float)
        valid = np.asarray(masks[i], bool) & np.isfinite(pixels)
        z, _, _, _ = _smooth_and_z(pixels, valid, denoiser=meta["denoiser"])
        if z is None:
            scores, accepted = np.full((16,16), np.nan), np.zeros((16,16), bool)
        else:
            scores, accepted = _scores(z, valid, 16)
        t24 = float(summary.iloc[i]["T24_p16"])
        metrics, valid16, active, raw, clipped = spatial_metrics(
            scores, accepted, t24, tile_size=tile_size)
        pair = int(item.pair_index)
        rows.append({"pair_index":pair,"end_time":item.end_time,
                     "detection_status":summary.iloc[i]["status"],
                     "active_fraction_pre_p04":summary.iloc[i]["active_fraction_pre"],
                     "T24_p16":t24, **metrics})
        grids[pair] = (valid16, active, raw, clipped)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No frames for {day}")
    out.mkdir(parents=True)
    df.to_csv(out/"p16_spatial_metrics.csv",index=False)
    fig, ax = plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    rejected = df.detection_status.eq("GLOBAL_ACTIVATION")
    for mask,name,marker in ((~rejected,"Other","."),
                             (rejected,"GLOBAL_ACTIVATION","x")):
        d = df.loc[mask]
        ax[0,0].scatter(d.p16_active_fraction,d.p16_clipped_excess_moran,
                        label=name,marker=marker,s=40)
        ax[0,1].scatter(d.p16_active_fraction,d.p16_weighted_tile_variance,
                        label=name,marker=marker,s=40)
    for key,axis in (("p16_binary_moran",ax[1,0]),
                     ("p16_raw_weighted_moran",ax[1,1])):
        axis.plot(df.pair_index,df[key],marker=".",linewidth=1)
        axis.axvline(focus_pair,color="orange",linestyle=":")
        axis.set(xlabel="Pair index",ylabel=key)
    ax[0,0].set(xlabel="P16 active fraction",ylabel="Clipped excess Moran's I")
    ax[0,1].set(xlabel="P16 active fraction",ylabel="Weighted tile variance")
    ax[0,0].legend()
    ax[0,1].legend()
    focus=df[df.pair_index.eq(focus_pair)]
    if not focus.empty:
        ax[0,0].scatter(focus.p16_active_fraction,focus.p16_clipped_excess_moran,
                        s=160,facecolors="none",edgecolors="orange",linewidths=2)
        ax[0,1].scatter(focus.p16_active_fraction,focus.p16_weighted_tile_variance,
                        s=160,facecolors="none",edgecolors="orange",linewidths=2)
    fig.suptitle(f"P16 spatial experiment {day} (raw P95(abs(Z)), threshold excess)")
    fig.savefig(out/"p16_spatial_overview.png",dpi=160)
    plt.close(fig)

    # Save every frame's actual P16 maps, so rejected frames can be inspected too.
    gallery = out/"grids"
    gallery.mkdir()
    for pair,(valid16,active,raw,clipped) in grids.items():
        fig, axes = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        panels=((np.ma.masked_where(~valid16, active.astype(float)),"P16 binary activation"),
                (np.ma.masked_where(~valid16, raw),"P16 active raw score P95(abs(Z))"),
                (np.ma.masked_where(~valid16, clipped),"P16 clipped threshold excess"))
        for a,(v,title) in zip(axes,panels):
            im=a.imshow(v,origin="lower",interpolation="nearest",cmap="viridis")
            a.set_title(title)
            fig.colorbar(im,ax=a,shrink=.75)
        status=df.loc[df.pair_index.eq(pair),"detection_status"].iloc[0]
        fig.suptitle(f"Pair {pair} | {status}")
        fig.savefig(gallery/f"pair_{pair:03d}.png",dpi=120)
        plt.close(fig)
    (out/"experiment_metadata.json").write_text(json.dumps({
        "source_detection":str(det),"date":day,"tile_size":tile_size,
        "score_definition":"P95(abs(Z)), inclusive of both Z signs",
        "active_definition":"P16 score > causal T24 P16",
        "raw_weight":"P16 score on active cells, zero otherwise",
        "excess_weight":"max(0, P16 score - T24 P16)",
        "clipped_excess_weight":"excess clipped at each frame active excess P95",
        "moran_weights":"4 nearest neighbors; valid pairs only",
        "tile_validity":"minimum 4 valid patches",
        "experiment_only":True,
    },indent=2),encoding="utf-8")
    print(f"Saved {len(df)} frames to {out}")
    print(df.loc[df.pair_index.eq(focus_pair)].to_string(index=False))
    print("GLOBAL_ACTIVATION frames:",int(rejected.sum()))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detection-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--date")
    p.add_argument("--focus-pair",type=int,default=99)
    p.add_argument("--tile-size",type=int,default=4)
    a=p.parse_args()
    run(a.detection_dir,a.output_dir,date=a.date,focus_pair=a.focus_pair,tile_size=a.tile_size)


if __name__=="__main__":
    main()
