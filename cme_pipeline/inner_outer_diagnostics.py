"""Read-only Inner/Outer P04 diagnostic for C3/D4 saved detections.

Recomputes the same Gaussian/median smoothed Z and P95(abs(Z)) patch scores,
including frames already rejected by GLOBAL_ACTIVATION. No detector changes.
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


def _region_masks(accepted, *, inner_min, boundary, outer_max):
    y, x = np.indices((64, 64), dtype=float)
    radius = np.hypot((x + .5) * 4 - 128., (y + .5) * 4 - 128.)
    return (
        accepted & (radius >= inner_min) & (radius < boundary),
        accepted & (radius >= boundary) & (radius <= outer_max),
    )


def _stats(scores, mask, cutoff):
    vals = np.asarray(scores[mask], dtype=float)
    vals = vals[np.isfinite(vals)]
    return {
        "n": len(vals),
        "p50": float(np.percentile(vals, 50)) if len(vals) else np.nan,
        "p90": float(np.percentile(vals, 90)) if len(vals) else np.nan,
        "p95": float(np.percentile(vals, 95)) if len(vals) else np.nan,
        "tail_fraction": float(np.mean(vals > cutoff)) if len(vals) else np.nan,
    }


def diagnose(detection_dir, output_dir, *, date=None, cutoff=.5,
             inner_min=40., boundary=None, outer_max=None,
             focus_pair=99, min_patches=10):
    detector = Path(detection_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    metadata = json.loads((detector / "detection_metadata.json").read_text())
    if metadata["detector"] not in ("c3", "d4"):
        raise ValueError("Use C3 or D4 detection results")
    pre = Path(metadata["input_dir"]).resolve()
    boundary = float(metadata.get("outer_floor_inner_px") or 88) if boundary is None else float(boundary)
    outer_max = float(metadata.get("outer_floor_outer_px") or 116) if outer_max is None else float(outer_max)
    if not (0 <= inner_min < boundary < outer_max <= 128):
        raise ValueError("Require 0 <= inner_min < boundary < outer_max <= 128")
    if not (np.isfinite(cutoff) and cutoff >= 0 and min_patches > 0):
        raise ValueError("Invalid cutoff or min_patches")
    summary = pd.read_csv(detector / "frame_detection_summary.csv")
    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    if not (len(summary) == len(manifest) == len(rd) == len(valid)
            and np.array_equal(summary.pair_index.astype(int), manifest.pair_index.astype(int))):
        raise ValueError("Detector and preprocessing frames do not align")
    target_date = date or str(metadata["requested_date"])
    selections = [
        i for i, item in manifest.iterrows()
        if pd.to_datetime(item["end_time"], utc=True).strftime("%Y%m%d") == target_date
    ]
    if not selections:
        raise ValueError(f"No frames for {target_date}")
    rows = []
    focus = None
    for i in selections:
        item = manifest.iloc[i]
        obs = summary.iloc[i]
        image = np.asarray(rd[i], dtype=float)
        usable = np.asarray(valid[i], dtype=bool) & np.isfinite(image)
        z, _, _, _ = _smooth_and_z(image, usable, denoiser=metadata["denoiser"])
        if z is None:
            scores = np.full((64, 64), np.nan)
            accepted = np.zeros((64, 64), bool)
        else:
            scores, accepted = _scores(z, usable, 4)
        inner, outer = _region_masks(accepted, inner_min=inner_min,
                                     boundary=boundary, outer_max=outer_max)
        ins, outs = _stats(scores, inner, cutoff), _stats(scores, outer, cutoff)
        enough = ins["n"] >= min_patches and outs["n"] >= min_patches
        ip95, op95 = ins["p95"], outs["p95"]
        ratio = ip95 / op95 if enough and np.isfinite(op95) and op95 > 0 else np.nan
        contrast = (ip95 - op95) / (ip95 + op95) if enough and np.isfinite(ip95 + op95) and ip95 + op95 > 0 else np.nan
        rows.append({
            "pair_index": int(item["pair_index"]),
            "end_time": item["end_time"],
            "detection_status": str(obs["status"]),
            "active_fraction_pre": obs["active_fraction_pre"],
            "n_inner": ins["n"], "n_outer": outs["n"],
            "inner_p50": ins["p50"], "outer_p50": outs["p50"],
            "inner_p90": ins["p90"], "outer_p90": outs["p90"],
            "inner_p95": ip95, "outer_p95": op95,
            "inner_tail_fraction": ins["tail_fraction"],
            "outer_tail_fraction": outs["tail_fraction"],
            "p95_inner_outer_ratio": ratio,
            "p95_symmetric_contrast": contrast,
            # Exploratory flags, NOT CME/no-CME labels:
            "both_p95_above_cutoff": bool(enough and min(ip95, op95) > cutoff),
            "weak_radial_contrast_20pct": bool(enough and np.isfinite(ratio) and
                                                1 / 1.2 <= ratio <= 1.2),
            "insufficient_region_samples": not enough,
        })
        if int(item["pair_index"]) == focus_pair:
            focus = (scores, inner, outer, image, usable)
    df = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=False)
    df.to_csv(output / "inner_outer_frame_metrics.csv", index=False)

    # Scatter: all frames, explicitly show formerly rejected QC frames.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    rejected = df["detection_status"].eq("GLOBAL_ACTIVATION")
    ok = ~rejected
    axes[0, 0].scatter(df.loc[ok, "outer_p95"], df.loc[ok, "inner_p95"],
                       s=24, alpha=.7, label="Other status")
    axes[0, 0].scatter(df.loc[rejected, "outer_p95"], df.loc[rejected, "inner_p95"],
                       s=42, marker="x", label="GLOBAL_ACTIVATION")
    lim = np.nanmax(df[["inner_p95", "outer_p95"]].to_numpy(dtype=float))
    if not np.isfinite(lim):
        lim = 1.
    axes[0, 0].plot([0, lim], [0, lim], linestyle="--", color="gray")
    axes[0, 0].axvline(cutoff, color="red", linestyle=":")
    axes[0, 0].axhline(cutoff, color="red", linestyle=":")
    axes[0, 0].set(xlabel="Outer P95", ylabel="Inner P95", title="P04 radial P95")
    axes[0, 0].legend()
    axes[0, 1].plot(df.pair_index, df.p95_inner_outer_ratio, marker=".", linewidth=1)
    axes[0, 1].axhline(1, color="gray", linestyle="--")
    axes[0, 1].axhline(1.2, color="red", linestyle=":")
    axes[0, 1].axhline(1 / 1.2, color="red", linestyle=":")
    axes[0, 1].set(xlabel="Pair index", ylabel="Inner P95 / Outer P95", title="Radial contrast over time")
    axes[1, 0].plot(df.pair_index, df.inner_tail_fraction, marker=".", label="Inner")
    axes[1, 0].plot(df.pair_index, df.outer_tail_fraction, marker=".", label="Outer")
    axes[1, 0].set(xlabel="Pair index", ylabel=f"Fraction with P04 score > {cutoff:g}",
                   title="Tail fraction in each radial region")
    axes[1, 0].legend()
    axes[1, 1].scatter(df.pair_index, df.active_fraction_pre, s=16, alpha=.8)
    axes[1, 1].scatter(df.loc[rejected, "pair_index"], df.loc[rejected, "active_fraction_pre"],
                       s=36, marker="x", label="GLOBAL_ACTIVATION")
    axes[1, 1].set(xlabel="Pair index", ylabel="Existing pre-QC active fraction",
                   title="Original activation QC diagnostic")
    axes[1, 1].legend()
    target = df.loc[df.pair_index.eq(focus_pair)]
    if len(target):
        for axis in axes.ravel():
            axis.axvline(focus_pair, color="orange", linestyle=":", alpha=.65) if axis is not axes[0, 0] else None
        axes[0, 0].scatter(target.outer_p95, target.inner_p95,
                           facecolors="none", edgecolors="orange", linewidth=2, s=160)
    fig.suptitle(f"{metadata['detector'].upper()} Inner/Outer P04 diagnostics: {target_date}")
    fig.savefig(output / "inner_outer_overview.png", dpi=150)
    plt.close(fig)

    if focus is not None:
        scores, inner, outer, image, usable = focus
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        axes[0].imshow(np.ma.masked_where(~usable, image), origin="lower", cmap="gray",
                       vmin=-600, vmax=600)
        y, x = np.indices((64, 64), dtype=float)
        r = np.hypot((x + .5) * 4 - 128., (y + .5) * 4 - 128.)
        axes[0].contour(r, levels=[inner_min, boundary, outer_max], origin="lower",
                        extent=(-.5, 255.5, -.5, 255.5),
                        colors=["cyan", "yellow", "orange"], linewidths=1.2)
        axes[0].set_title(f"Pair {focus_pair}: radial boundaries")
        iv = np.asarray(scores[inner], float)
        ov = np.asarray(scores[outer], float)
        v = np.concatenate((iv[np.isfinite(iv)], ov[np.isfinite(ov)]))
        high = max(1., float(np.percentile(v, 99)) if len(v) else 1.)
        bins = np.linspace(0, high, 100)
        axes[1].hist(iv, bins=bins, alpha=.55, label=f"Inner (n={len(iv)})")
        axes[1].hist(ov, bins=bins, alpha=.55, label=f"Outer (n={len(ov)})")
        axes[1].axvline(cutoff, color="red", linestyle="--", label=f"Floor {cutoff:g}")
        axes[1].set(xlabel="P04 P95(abs(Z)) score", ylabel="Patch count",
                    title=f"Pair {focus_pair}: regional score histograms")
        axes[1].legend()
        fig.savefig(output / f"pair_{focus_pair}_inner_outer.png", dpi=160)
        plt.close(fig)
    (output / "experiment_metadata.json").write_text(json.dumps({
        "source_detection": str(detector), "source_preprocessing": str(pre),
        "date": target_date, "cutoff": cutoff,
        "inner_radius_min_px": inner_min, "outer_floor_inner_px": boundary,
        "outer_floor_outer_px": outer_max,
        "min_patches_per_region": min_patches,
        "note": "Diagnostic only; no automatic QC decisions. P95 comparison can miss angularly localized CME.",
    }, indent=2))
    print(f"Saved {len(df)} frames in {output}")
    print(df.loc[df.pair_index.eq(focus_pair)].to_string(index=False))
    print("QC rejected:", int(rejected.sum()),
          "| weak contrast & both above cutoff:",
          int((df.weak_radial_contrast_20pct & df.both_p95_above_cutoff).sum()))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--date")
    parser.add_argument("--cutoff", type=float, default=.5)
    parser.add_argument("--inner-min-radius-px", type=float, default=40.)
    parser.add_argument("--boundary-radius-px", type=float)
    parser.add_argument("--outer-max-radius-px", type=float)
    parser.add_argument("--focus-pair", type=int, default=99)
    parser.add_argument("--min-patches", type=int, default=10)
    args = parser.parse_args()
    diagnose(args.detection_dir, args.output_dir, date=args.date,
             cutoff=args.cutoff, inner_min=args.inner_min_radius_px,
             boundary=args.boundary_radius_px, outer_max=args.outer_max_radius_px,
             focus_pair=args.focus_pair, min_patches=args.min_patches)


if __name__ == "__main__":
    main()
