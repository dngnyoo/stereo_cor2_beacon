"""Read-only D4 normalized weighted-variance QC candidate and RD image review.

Uses precomputed P16 metrics, original RD and saved D4 P04 states.
Does NOT modify detection, tracking, or any existing QC outputs.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from cme_pipeline.stages.segmentation.visualization import draw_rd_background
import numpy as np
import pandas as pd

from cme_pipeline.live_detection import _scores, _smooth_and_z
from cme_pipeline.p16_spatial_diagnostics import spatial_metrics


def make_review(detection_dir, metrics_csv, output_dir, *,
                normalized_max=1.5, date=None):
    detector = Path(detection_dir).resolve()
    metrics_path = Path(metrics_csv).resolve()
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    if not (np.isfinite(normalized_max) and normalized_max > 0):
        raise ValueError("normalized_max must be finite and positive")

    meta = json.loads((detector / "detection_metadata.json").read_text())
    if meta["detector"] != "d4":
        raise ValueError("This review expects saved D4 results")
    pre = Path(meta["input_dir"]).resolve()
    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    det_summary = pd.read_csv(detector / "frame_detection_summary.csv")
    metrics = pd.read_csv(metrics_path)
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    p04 = np.load(detector / "patch_state_p04.npy", mmap_mode="r")
    if not (len(manifest) == len(det_summary) == len(rd) == len(valid) == len(p04)):
        raise ValueError("Detector/preprocessing frame count mismatch")
    if not np.array_equal(manifest.pair_index.astype(int), det_summary.pair_index.astype(int)):
        raise ValueError("Detector and preprocessing pair indices differ")
    required_metrics = {"p16_normalized_weighted_tile_variance", "n_active_p16",
                        "p16_active_fraction", "p16_weighted_tile_variance"}
    if not required_metrics.issubset(metrics.columns):
        raise ValueError("Missing normalized metrics; run p16_spatial_diagnostics v2 first")
    if metrics.pair_index.duplicated().any():
        raise ValueError("Duplicate pairs in spatial metrics")
    day = date or str(meta["requested_date"])
    indices = [i for i, r in manifest.iterrows()
               if pd.to_datetime(r.end_time, utc=True).strftime("%Y%m%d") == day]
    if not indices:
        raise ValueError(f"No frames found for date {day}")
    day_pairs = set(manifest.iloc[indices].pair_index.astype(int))
    measured_pairs = set(metrics.pair_index.astype(int))
    if day_pairs != measured_pairs:
        raise ValueError(f"Metrics must cover exactly the date frames; missing={sorted(day_pairs-measured_pairs)}, extra={sorted(measured_pairs-day_pairs)}")
    metrics = metrics.set_index("pair_index", drop=False)
    frame_rows = []
    for i in indices:
        pair = int(manifest.iloc[i].pair_index)
        m = metrics.loc[pair]
        d = det_summary.iloc[i]
        if str(m.detection_status) != str(d.status):
            raise ValueError(f"Mismatch in saved QC status for pair {pair}")
        p16_n = float(m.p16_active_fraction)
        variance = float(m.p16_weighted_tile_variance)
        normalized = float(m.p16_normalized_weighted_tile_variance)
        spatial = bool(np.isfinite(normalized) and normalized < normalized_max)
        old_reject = d.status == "GLOBAL_ACTIVATION"
        invalid = d.status != "VALID" and not old_reject
        # Independent rule is evaluated on historical GLOBAL_ACTIVATION frames too.
        # Combined QC is a conservative extension: old QC + new valid-frame failures.
        combined = old_reject or (d.status == "VALID" and spatial)
        category = ("BOTH_QC_REJECT" if old_reject and spatial else
                    "EXISTING_QC_ONLY" if old_reject else
                    "NEW_SPATIAL_ONLY" if d.status == "VALID" and spatial else
                    "OTHER_UNUSABLE" if invalid else "RETAINED")
        frame_rows.append({
            "pair_index": pair, "end_time": manifest.iloc[i].end_time,
            "old_status": d.status, "old_qc_reject": bool(old_reject),
            "normalized_qc_reject_independent": spatial if not invalid else False,
            "candidate_status": category, "candidate_reject": combined,
            "p16_active_fraction": p16_n,
            "p16_weighted_tile_variance": variance,
            "p16_normalized_weighted_tile_variance": normalized,
            "p16_excess_active_p95": float(m.p16_excess_active_p95),
            "p04_detected_patches": int(np.count_nonzero(p04[i] == 1)),
            "manual_cme_review": "", "manual_notes": "",
        })
    table = pd.DataFrame(frame_rows).sort_values("pair_index")
    output_frames = out / "frames"
    output_frames.mkdir(parents=True)
    table.to_csv(out / "normalized_qc_full_day_decisions.csv", index=False)
    table.loc[table.candidate_reject].to_csv(out / "qc_candidate_review_sheet.csv", index=False)
    table.loc[table.old_qc_reject != table.normalized_qc_reject_independent].to_csv(
        out / "old_vs_normalized_disagreements.csv", index=False)
    # The independently-applied normalized rule does not imply that existing QC
    # rejection is safe to reverse; the combined outcome always retains old QC.
    counts = {
        "total": int(len(table)),
        "old_qc_rejected": int(table.old_qc_reject.sum()),
        "normalized_rule_rejected": int(table.normalized_qc_reject_independent.sum()),
        "both_rejected": int((table.old_qc_reject & table.normalized_qc_reject_independent).sum()),
        "new_spatial_only": int((~table.old_qc_reject & table.normalized_qc_reject_independent).sum()),
        "existing_qc_only": int((table.old_qc_reject & ~table.normalized_qc_reject_independent).sum()),
        "combined_rejected": int(table.candidate_reject.sum()),
        "combined_retained": int(table.candidate_status.eq("RETAINED").sum()),
    }
    (out / "qc_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.plot(table.pair_index, table.p16_normalized_weighted_tile_variance,
            ".-", color="#526F98", label="Normalized weighted variance")
    ax.axhline(normalized_max, color="#BC4839", ls="--",
               label=f"QC cutoff < {normalized_max:g}")
    for _, rr in table.loc[table.candidate_reject].iterrows():
        ax.axvspan(rr.pair_index-.4, rr.pair_index+.4, color="#C67A45", alpha=.12)
    ax.set(xlabel="Pair index", ylabel="Weighted tile variance / mean(tile excess)^2",
           title="D4 full-day normalized spatial QC evaluation")
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(out / "normalized_qc_timeline.png", dpi=160)
    plt.close(fig)

    for i in indices:
        pair = int(manifest.iloc[i].pair_index)
        row = table[table.pair_index.eq(pair)].iloc[0]
        image = np.asarray(rd[i], dtype=float)
        mask = np.asarray(valid[i], dtype=bool) & np.isfinite(image)
        z, _, _, _ = _smooth_and_z(image, mask, denoiser=meta["denoiser"])
        if z is None:
            scores16, accepted16 = np.full((16, 16), np.nan), np.zeros((16,16), bool)
        else:
            scores16, accepted16 = _scores(z, mask, 16)
        t24 = float(det_summary.iloc[i].T24_p16)
        _, v16, a16, raw16, clipped16 = spatial_metrics(scores16, accepted16, t24)
        if int(a16.sum()) != int(metrics.loc[pair].n_active_p16):
            raise ValueError(f"P16 recomputation does not match saved metrics for pair {pair}")
        # Project standard: signed North-up RD; coolwarm +/-600 DN/s;
        # black invalid pixels; white figure background; solar reference circles.
        fig, axes = plt.subplots(1, 4, figsize=(19, 6), constrained_layout=True)
        fig.patch.set_facecolor("white")
        for a in axes:
            a.set_facecolor("white")
            a.set_box_aspect(1)
        for a in axes[:2]:
            rd_artist = draw_rd_background(a, image, mask)
            a.set_xlim(0, 256)
            a.set_ylim(0, 256)
            a.set_xlabel("X (pixel)")
            a.set_ylabel("Y (pixel)")
            a.set_aspect("equal")
            # Solar geometry follows the previous fixed project gallery convention.
            sx, sy = 126.618, 127.503
            for rr, color, style, width in (
                (22., "white", "--", 1.0),
                (8.163, "#FFD700", "--", 1.0),
                (5*8.163, "white", ":", .65),
                (10*8.163, "white", ":", .65),
                (15*8.163, "white", ":", .65),
            ):
                a.add_patch(Circle((sx, sy), rr, fill=False,
                                   edgecolor=color, linestyle=style,
                                   linewidth=width, alpha=.65))
        fig.colorbar(rd_artist, ax=axes[:2], orientation="vertical",
                     fraction=.018, pad=.015, label="RD (DN/s)", shrink=.75)
        axes[0].set_title("Signed North-up RD (DN/s)")
        axes[1].set_title("RD + saved D4 P04 detections")
        yy,xx = np.where(np.asarray(p04[i]) == 1)
        for y,x in zip(yy,xx):
            axes[1].add_patch(Rectangle((x*4,y*4),4,4,
                                        facecolor="none",edgecolor="#FF00FF",
                                        linewidth=.65,alpha=.9))
        # Original QC-rejected frames have -1 sentinels, not P04 detections.
        axes[2].imshow(np.ma.masked_where(~v16,a16.astype(float)),
                       origin="lower",interpolation="nearest",vmin=0,vmax=1,cmap="viridis")
        axes[2].set_title("P16 binary activation")
        im = axes[3].imshow(np.ma.masked_where(~v16,clipped16),
                            origin="lower",interpolation="nearest",cmap="viridis",
                            vmin=0,vmax=max(0.1,float(np.nanmax(clipped16))))
        axes[3].set_title("P16 clipped excess")
        fig.colorbar(im,ax=axes[3],shrink=.75, label="P16 excess above T24")
        for a in axes[2:]:
            a.set_xlabel("P16 grid X")
            a.set_ylabel("P16 grid Y")
        fig.suptitle(
            f"Pair {pair} | {pd.to_datetime(row['end_time'], utc=True).strftime('%Y-%m-%d %H:%M:%S UTC')} | {row['candidate_status']}\n"
            f"Normalized variance={row['p16_normalized_weighted_tile_variance']:.3f} "
            f"(reject if < {normalized_max:.2f}); "
            f"P16 active={row['p16_active_fraction']:.3f}",
            fontsize=11,
        )
        fig.savefig(output_frames / f"pair_{pair:03d}.png",dpi=140)
        plt.close(fig)

    # Offline HTML gallery: no scripts, no external dependencies.
    entries = []
    for _,r in table.iterrows():
        pair = int(r.pair_index)
        category = html.escape(str(r.candidate_status))
        entries.append(
            f'<section id="pair-{pair}"><h2>Pair {pair} — {category}</h2>'
            f'<p>{html.escape(str(r.end_time))} | P16 active '
            f'{r.p16_active_fraction:.3f} | normalized variance '
            f'{r.p16_normalized_weighted_tile_variance:.3f} | '
            f'old QC: {html.escape(str(r.old_status))}</p>'
            f'<img src="frames/pair_{pair:03d}.png" alt="Pair {pair} RD and spatial maps" loading="lazy"></section>'
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>D4 spatial QC review</title>'
        '<style>body{font:15px system-ui;margin:2em auto;max-width:1500px;padding:0 1em}'
        'section{border-top:1px solid #aaa;padding:1em 0}img{width:100%;height:auto}'
        'a{margin-right:.6em}nav{line-height:2.5}</style>'
        f'<h1>D4 P16 normalized spatial QC review — {html.escape(day)}</h1>'
        f'<p>Exploratory independent rule: normalized weighted variance &lt; {normalized_max:g}. '
        'Existing GLOBAL_ACTIVATION is preserved in the combined proposal. '
        'Rejected frames are candidates for visual review, not ground-truth CME labels.</p>'
        '<nav>'+''.join(f'<a href="#pair-{int(p)}">{int(p)}</a>'
                          for p in table.loc[table.candidate_reject,"pair_index"])+
        '</nav>'+''.join(entries)+'</html>'
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "review_metadata.json").write_text(json.dumps({
        "detector_dir":str(detector),"metrics_csv":str(metrics_path),
        "date":day,"normalized_max":normalized_max,"counts":counts,
        "original_d4_unchanged":True,
        "manual_review_required":True,
    },indent=2))
    print("\nOld vs normalized rule summary:")
    print(json.dumps(counts, indent=2))
    print("\nCandidate counts:")
    print(table.candidate_status.value_counts().to_string())
    print("\nCandidate pair indices:", table.loc[table.candidate_reject,"pair_index"].tolist())
    print("New spatial-only pair indices:", table.loc[table.candidate_status.eq("NEW_SPATIAL_ONLY"),"pair_index"].tolist())
    print("Review gallery:", out/"index.html")
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detection-dir",required=True)
    p.add_argument("--metrics",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--date")
    p.add_argument("--normalized-max",type=float,default=1.5)
    a=p.parse_args()
    make_review(a.detection_dir,a.metrics,a.output_dir,date=a.date,
                normalized_max=a.normalized_max)


if __name__=="__main__":
    main()
