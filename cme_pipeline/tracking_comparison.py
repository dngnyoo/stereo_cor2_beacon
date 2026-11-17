"""Read-only full-day side-by-side A5b tracking diagnostics."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def compare_tracking(
    baseline_run: str | Path, experiment_run: str | Path, *,
    date: str, output_dir: str | Path | None = None,
) -> Path:
    """Compare P04 detection outlines and track ownership IDs on identical RD."""
    if len(date) != 8 or not date.isascii() or not date.isdigit():
        raise ValueError("date must be YYYYMMDD")
    base, exp = Path(baseline_run).resolve(), Path(experiment_run).resolve()
    btrack = json.loads((base / "tracking_metadata.json").read_text())
    etrack = json.loads((exp / "tracking_metadata.json").read_text())
    if btrack["method"] != etrack["method"] or btrack["method"] != "a5b":
        raise ValueError("Both tracking runs must use A5b")
    bdet, edet = Path(btrack["detection_dir"]), Path(etrack["detection_dir"])
    bm = json.loads((bdet / "detection_metadata.json").read_text())
    em = json.loads((edet / "detection_metadata.json").read_text())
    pre = Path(bm["input_dir"]).resolve()
    if pre != Path(em["input_dir"]).resolve():
        raise ValueError("Both tracking runs must refer to the same preprocessed RD")
    if btrack["requested_date"] != date or etrack["requested_date"] != date:
        raise ValueError("Tracking run dates differ from requested date")

    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    bs = pd.read_csv(bdet / "frame_detection_summary.csv")
    es = pd.read_csv(edet / "frame_detection_summary.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    bstate = np.load(bdet / "patch_state_p04.npy", mmap_mode="r")
    estate = np.load(edet / "patch_state_p04.npy", mmap_mode="r")
    bown = np.load(base / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    eown = np.load(exp / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    n = len(manifest)
    if (rd.shape != valid.shape or rd.shape != (n, 256, 256)
        or any(a.shape != (n, 64, 64) for a in (bstate, estate, bown, eown))
        or len(bs) != n or len(es) != n):
        raise ValueError("RD, detection and tracking frame dimensions must match")
    for summary in (bs, es):
        if not np.array_equal(summary["pair_index"].astype(int), manifest["pair_index"].astype(int)):
            raise ValueError("Frame pair index mismatch")
        if not np.array_equal(
            pd.to_datetime(summary["end_time"], utc=True).to_numpy(),
            pd.to_datetime(manifest["end_time"], utc=True).to_numpy()
        ):
            raise ValueError("Frame timestamp mismatch")
    times = pd.to_datetime(manifest["end_time"], utc=True)
    indices = np.flatnonzero(times.dt.strftime("%Y%m%d").eq(date).to_numpy())
    if len(indices) == 0:
        raise ValueError(f"No frames on {date}")
    out = Path(output_dir).resolve() if output_dir else exp / "comparison_tracking_full_day"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite comparison directory: {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    from cme_pipeline.stages.segmentation.visualization import get_rd_cmap

    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True)
    rows = []
    for i0 in indices:
        i = int(i0)
        image = np.asarray(rd[i], dtype=float)
        usable = np.asarray(valid[i], dtype=bool) & np.isfinite(image)
        image = np.ma.masked_where(~usable, image)
        pair = int(manifest.iloc[i]["pair_index"])
        frame_rows = []
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        for ax, label, det, own, summary in (
            (axes[0], "Baseline + A5b", bstate[i], bown[i], bs.iloc[i]),
            (axes[1], "Outer P95 + A5b", estate[i], eown[i], es.iloc[i])
        ):
            det = np.asarray(det) == 1
            own = np.asarray(own, dtype=np.int32)
            if np.any((own > 0) & ~det):
                raise ValueError(f"Track-owned patch absent from detection: pair {pair}")
            im = ax.imshow(image, origin="lower", cmap=get_rd_cmap(),
                           vmin=-600, vmax=600, interpolation="nearest")
            yy, xx = np.nonzero(det)
            if len(yy):
                ax.add_collection(PatchCollection([
                    Rectangle((int(x) * 4 - .5, int(y) * 4 - .5), 4, 4)
                    for y, x in zip(yy, xx)
                ], facecolor="none", edgecolor="red", linewidth=.6, alpha=.9))
            # Colored ownership centers; color determined by numeric track ID.
            yo, xo = np.nonzero(own > 0)
            if len(yo):
                ax.scatter(xo * 4 + 1.5, yo * 4 + 1.5, c=own[yo, xo],
                           s=9, cmap="tab20", vmin=0.5, vmax=20.5,
                           marker="s", edgecolors="none", alpha=.95)
            tids = np.unique(own[own > 0])
            ax.set_title(
                f"{label} | {summary['status']}\n"
                f"Active: {int(det.sum())}, owned: {len(yo)}, IDs: {len(tids)}",
                fontsize=10
            )
            ax.set_xlim(-.5, 255.5)
            ax.set_ylim(-.5, 255.5)
            ax.set_aspect("equal")
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
            frame_rows.append((int(det.sum()), len(yo), len(tids)))
        fig.colorbar(im, ax=axes, label="RD (DN/s)", shrink=.75)
        fig.suptitle(f"2024-05-10 A5b | Pair {pair} | {times.iloc[i].strftime('%H:%M UTC')}")
        fig.savefig(frame_dir / f"pair_{pair:03d}_{times.iloc[i].strftime('%Y%m%dT%H%M%SZ')}.png",
                    dpi=140)
        plt.close(fig)
        rows.append({
            "pair_index": pair,
            "end_time": manifest.iloc[i]["end_time"],
            "baseline_status": bs.iloc[i]["status"],
            "outer_status": es.iloc[i]["status"],
            "baseline_active": frame_rows[0][0],
            "outer_active": frame_rows[1][0],
            "baseline_owned": frame_rows[0][1],
            "outer_owned": frame_rows[1][1],
            "baseline_track_ids": frame_rows[0][2],
            "outer_track_ids": frame_rows[1][2],
        })
    pd.DataFrame(rows).to_csv(out / "comparison_summary.csv", index=False)
    print(f"Compared {len(rows)} frames | Baseline total tracks={btrack['total_tracks']} "
          f"| Outer P95 total tracks={etrack['total_tracks']}")
    return out
