"""Read-only P16 spatial concentration experiment on saved D4 observations.

Evaluate absolute normalized weighted variance, active-fraction adjustment,
and a count-and-weight-conditioned spatial permutation null.
No detector thresholds, QC status, or tracking arrays are changed.
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

from cme_pipeline.live_detection import _scores, _smooth_and_z
from cme_pipeline.p16_spatial_diagnostics import spatial_metrics


def tile_statistics(weight_vector, tile_ids, tile_counts, used_tiles):
    """Statistics of per-tile mean excess with original valid patch denominators."""
    tile_sums = np.bincount(tile_ids, weights=weight_vector, minlength=16)
    tile_means = tile_sums[used_tiles] / tile_counts[used_tiles]
    mean = float(tile_means.mean())
    variance = float(tile_means.var())
    normalized = variance / mean ** 2 if mean > 1e-10 else np.nan
    return variance, normalized


def evaluate_frame(scores, valid, t24, active_fraction, *,
                   permutations, rng):
    accepted = np.asarray(valid, bool) & np.isfinite(scores)
    active = accepted & np.isfinite(t24) & (scores > t24)
    excess = np.maximum(np.asarray(scores, float)[active] - t24, 0.0)
    cap = float(np.percentile(excess, 95)) if excess.size else 0.0
    weights = np.minimum(excess, cap)
    valid_flat = np.flatnonzero(accepted.ravel())
    # 16x16 grid, four-by-four P16 patches per macro-tile (16 tiles).
    positions = np.arange(256)
    tile_map = ((positions // 16) // 4) * 4 + ((positions % 16) // 4)
    tile_ids = tile_map[valid_flat]
    counts = np.bincount(tile_ids, minlength=16)
    used = counts >= 4
    if used.sum() < 2 or not excess.size:
        return dict(null_mean=np.nan,null_std=np.nan,spatial_z=np.nan,
                    upper_tail_p=np.nan,random_mean_normalized=np.nan,
                    random_sd_normalized=np.nan)
    observed = np.zeros(len(valid_flat), dtype=float)
    # Map weights to actual active locations in the accepted-vector ordering.
    valid_active = active.ravel()[valid_flat]
    observed[valid_active] = weights
    obs_var, obs_normalized = tile_statistics(observed, tile_ids, counts, used)
    random_variances = np.empty(permutations, dtype=float)
    random_normalized = np.empty(permutations, dtype=float)
    # Assign same positive weight multiset randomly to valid P16 positions.
    # This preserves active count, excess magnitudes, clipping, and mask.
    for idx in range(permutations):
        shuffled = rng.permutation(observed)
        vv, nv = tile_statistics(shuffled, tile_ids, counts, used)
        random_variances[idx] = vv
        random_normalized[idx] = nv
    null_avg = float(random_normalized.mean())
    null_sd = float(random_normalized.std(ddof=1))
    z = float((obs_normalized-null_avg)/null_sd) if null_sd > 1e-12 else np.nan
    # Inclusive one-sided empirical tail p, plus-one Monte Carlo correction.
    p = float((np.count_nonzero(random_normalized >= obs_normalized)+1)
              / (permutations+1))
    return {
        "null_mean":null_avg, "null_std":null_sd, "spatial_z":z,
        "upper_tail_p":p, "random_mean_normalized":null_avg,
        "random_sd_normalized":null_sd,
    }


def run(cases, output_dir, *, permutations=1000, seed=20261117, tile_size=4):
    if tile_size != 4:
        raise ValueError("This comparison is fixed to tile_size=4, matching saved metrics")
    if permutations < 30:
        raise ValueError("Use at least 30 permutations")
    if len({case[0] for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation dates")
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    all_rows = []
    for day, detector_dir, metric_csv in cases:
        det = Path(detector_dir).resolve()
        meta = json.loads((det/"detection_metadata.json").read_text())
        if meta["detector"] != "d4":
            raise ValueError(f"D4 required: {det}")
        pre = Path(meta["input_dir"]).resolve()
        manifest = pd.read_csv(pre/"01_rd_manifest.csv")
        summary = pd.read_csv(det/"frame_detection_summary.csv")
        saved = pd.read_csv(metric_csv)
        if saved.pair_index.duplicated().any():
            raise ValueError("Duplicate saved P16 pair indices")
        saved = saved.set_index("pair_index")
        rd = np.load(pre/"north_up_RD_DNs_per_second.npy",mmap_mode="r")
        masks = np.load(pre/"north_up_pair_valid_masks.npy",mmap_mode="r")
        if not (len(manifest)==len(summary)==len(rd)==len(masks)):
            raise ValueError(f"Manifest/array count mismatch: {day}")
        if not np.array_equal(manifest.pair_index.astype(int),summary.pair_index.astype(int)):
            raise ValueError("Detector summary does not match manifest")
        times = pd.to_datetime(manifest.end_time,utc=True)
        selected = np.flatnonzero(times.dt.strftime("%Y%m%d").eq(day).to_numpy())
        if not len(selected) or set(manifest.iloc[selected].pair_index.astype(int)) != set(saved.index.astype(int)):
            raise ValueError(f"Saved metrics not exactly aligned with {day} frames")
        for i in selected:
            pair = int(manifest.iloc[i].pair_index)
            item = saved.loc[pair]
            pixels=np.asarray(rd[i],float)
            mask=np.asarray(masks[i],bool) & np.isfinite(pixels)
            z,_,_,_ = _smooth_and_z(pixels,mask,denoiser=meta["denoiser"])
            if z is None:
                scores, valid16 = np.full((16,16),np.nan), np.zeros((16,16),bool)
            else:
                scores, valid16 = _scores(z,mask,16,score_mode=meta.get("score_mode","abs"))
            t24 = float(summary.iloc[i].T24_p16)
            measure, _, active, _, _ = spatial_metrics(scores,valid16,t24)
            for key in ("p16_active_fraction","p16_weighted_tile_variance",
                        "p16_normalized_weighted_tile_variance"):
                a,b = float(measure[key]),float(item[key])
                if not np.isclose(a,b,rtol=1e-6,atol=1e-8,equal_nan=True):
                    raise ValueError(f"{day} pair {pair}: recomputed {key}={a} != CSV {b}")
            # Deterministic per-frame random generator independent of loop order.
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(day),pair]))
            random = evaluate_frame(scores,valid16,t24,measure["p16_active_fraction"],
                                    permutations=permutations,rng=rng)
            fraction = float(measure["p16_active_fraction"])
            normalized = float(measure["p16_normalized_weighted_tile_variance"])
            all_rows.append({
                "date":day,"pair_index":pair,"end_time":manifest.iloc[i].end_time,
                "d4_status":summary.iloc[i].status,
                "n_valid_p16":int(measure["n_valid_p16"]),
                "n_active_p16":int(measure["n_active_p16"]),
                "active_fraction":fraction,
                "weighted_variance":float(measure["p16_weighted_tile_variance"]),
                "normalized_weighted_variance":normalized,
                "active_fraction_adjusted_variance":fraction*normalized,
                "normalized_rule_lt_1p5":bool(np.isfinite(normalized) and normalized<1.5),
                **random,
            })
        print(f"Completed {day}: {len(selected)} frames",flush=True)

    df=pd.DataFrame(all_rows).sort_values(["date","pair_index"])
    out.mkdir(parents=True)
    df.to_csv(out/"spatial_count_adjustment_all_frames.csv",index=False)
    focuses={"20240503":[130], "20240510":[99,100,110,111,112,137],
             "20240531":[113]}
    focus=df.loc[[int(row.pair_index) in focuses.get(str(row.date),[])
                  for _,row in df.iterrows()]]
    focus.to_csv(out/"focus_frame_comparison.csv",index=False)

    for day,block in df.groupby("date",sort=True):
        fig,axes=plt.subplots(4,1,figsize=(15,12),sharex=True,constrained_layout=True)
        values=[
            ("normalized_weighted_variance","Original normalized weighted variance"),
            ("active_fraction_adjusted_variance","Active fraction x normalized variance"),
            ("spatial_z","Count-and-weight conditioned permutation Z"),
            ("active_fraction","Active P16 fraction"),
        ]
        for ax,(key,title) in zip(axes,values):
            ax.plot(block.pair_index,block[key],marker="o",ms=3,lw=1)
            if key=="normalized_weighted_variance":
                ax.axhline(1.5,color="red",linestyle="--",label="Original QC cutoff 1.5")
                ax.legend(loc="best")
            if key=="spatial_z":
                ax.axhline(0,color="grey",linestyle=":")
            for pair in focuses.get(str(day),[]):
                row=block.loc[block.pair_index.eq(pair)]
                if not row.empty:
                    ax.axvline(pair,color="#ca7b28",alpha=.55,linestyle=":")
                    val=float(row.iloc[0][key])
                    if np.isfinite(val):
                        ax.annotate(str(pair),(pair,val),xytext=(4,5),
                                    textcoords="offset points",fontsize=9)
            ax.set_ylabel(title)
            ax.grid(alpha=.2)
        axes[-1].set_xlabel("Pair index")
        fig.suptitle(f"P16 spatial concentration comparison {day} — {permutations} permutations")
        fig.savefig(out/f"{day}_spatial_count_adjustment.png",dpi=160)
        plt.close(fig)
    (out/"experiment_metadata.json").write_text(json.dumps({
        "cases":cases,"permutations":permutations,"seed":seed,
        "tile_size":4,"null_hypothesis":"Random reassignment of the frame's clipped positive P16 excess weights among valid P16 positions",
        "null_preserves":["valid mask","number of active patches","weight multiset","tile denominators"],
        "spatial_z":"(observed normalized weighted tile variance - permutation mean) / permutation sample SD",
        "upper_tail_p":"(number of shuffled values >= observed + 1)/(permutations + 1)",
        "limitation":"Uniform random-position null is not a realistic physical instrument noise model; descriptive comparison only",
        "no_qc_mutation":True,
    },indent=2),encoding="utf-8")
    print("\nFocus frames:\n",focus[[
        "date","pair_index","active_fraction",
        "normalized_weighted_variance","active_fraction_adjusted_variance",
        "null_mean","null_std","spatial_z","upper_tail_p"]].round(5).to_string(index=False))
    print("\nSaved:",out)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",nargs=3,action="append",required=True,
                   metavar=("DATE","DETECTION_DIR","METRICS_CSV"),
                   help="Repeat once per date")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--permutations",type=int,default=1000)
    p.add_argument("--seed",type=int,default=20261117)
    args=p.parse_args()
    run(args.case,args.output_dir,permutations=args.permutations,seed=args.seed)


if __name__=="__main__":
    main()
