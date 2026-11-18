"""Canonical single-run A5b full-day visualizations from saved tracking outputs.

Track-level annotations use the same WCS and geometry definitions as the
two-run comparison: two PA edges, tracked CPA and front arc.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from cme_pipeline.tracking_comparison import _frame_tracks


def save_tracking_diagnostics(
    tracking_run: str | Path, *, date: str | None = None,
    output_dir: str | Path | None = None,
    included_track_ids: set[int] | None = None,
    track_family_ids: dict[int, str] | None = None,
    allowed_family_pairs: dict[str, set[int]] | None = None,
) -> Path:
    root = Path(tracking_run).resolve()
    meta = json.loads((root / "tracking_metadata.json").read_text())
    if meta["method"] != "a5b":
        raise ValueError("Canonical tracking visualization currently supports A5b")
    wanted = str(date or meta["requested_date"])
    det = Path(meta["detection_dir"]).resolve()
    detmeta = json.loads((det / "detection_metadata.json").read_text())
    pre = Path(detmeta["input_dir"]).resolve()
    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    summary = pd.read_csv(det / "frame_detection_summary.csv")
    states = pd.read_csv(root / "tables" / "live_tracking_states.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    owners = np.load(root / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    n = len(manifest)
    if rd.shape != valid.shape or rd.shape != (n, 256, 256) or owners.shape != (n, 64, 64):
        raise ValueError("RD and saved ownership arrays are not aligned")
    if not np.array_equal(summary["pair_index"].astype(int), manifest["pair_index"].astype(int)):
        raise ValueError("Detection/manifest ordering mismatch")
    times = pd.to_datetime(manifest["end_time"], utc=True)
    indices = np.flatnonzero(times.dt.strftime("%Y%m%d").eq(wanted).to_numpy())
    if not len(indices):
        raise ValueError(f"No evaluation frames for {wanted}")
    from astropy.io import fits
    from astropy.wcs import WCS
    from cme_pipeline.utils.geometry import compute_wcs_rasters

    wcs = WCS(fits.Header.fromstring(
        (pre / "target_wcs_header.txt").read_text(encoding="utf-8"), sep="\n"))
    sx, sy = wcs.world_to_pixel_values(0., 0.)
    if not np.isfinite([sx, sy]).all():
        raise ValueError("Invalid solar center WCS")
    rasters = compute_wcs_rasters(wcs=wcs, center_xy=(sx, sy), patch_size=4)
    out = (Path(output_dir).resolve() if output_dir else
           root / "visualizations" / "tracking")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite visualization: {out}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cme_pipeline.family_colors import track_color
    from matplotlib import patheffects
    from cme_pipeline.stages.segmentation.visualization import get_rd_cmap

    out.mkdir(parents=True)
    records = []
    stroke = [patheffects.withStroke(linewidth=2.2, foreground="black")]

    def xy(rad, pa):
        theta = np.deg2rad(pa)
        return sx - rad * np.sin(theta), sy + rad * np.cos(theta)

    for idx in indices:
        i = int(idx)
        pair = int(manifest.iloc[i]["pair_index"])
        image = np.asarray(rd[i], float)
        image = np.ma.masked_where(~np.asarray(valid[i], bool) | ~np.isfinite(image), image)
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        im = ax.imshow(image, origin="lower", cmap=get_rd_cmap(), vmin=-600, vmax=600)
        owner_grid = np.asarray(owners[i])
        if included_track_ids is not None:
            owner_grid = np.where(np.isin(owner_grid, list(included_track_ids)), owner_grid, 0)
        if allowed_family_pairs is not None:
            if track_family_ids is None:
                raise ValueError('allowed_family_pairs requires track_family_ids')
            keep_ids = [tid for tid, fid in track_family_ids.items()
                        if pair in allowed_family_pairs.get(fid, set())]
            owner_grid = np.where(np.isin(owner_grid, keep_ids), owner_grid, 0)
        tracks = _frame_tracks(owner_grid, rasters["patch_pa"], rasters["patch_rad"])
        observed = states[(states["pair_index"] == pair) & (states["state"] == "OBSERVED")]
        by_id = {int(r["track_id"]): r for _, r in observed.iterrows()}
        for t in tracks:
            tid = t["track_id"]
            colour = track_color(tid)
            ax.contour(np.arange(64) * 4 + 1.5, np.arange(64) * 4 + 1.5,
                       t["mask"].astype(float), levels=[.5],
                       colors=[colour], linewidths=1.6)
            rad = t["observed_front_radius_px"]
            start = t["pa_arc_start_deg"]
            end = t["pa_arc_end_deg"]
            arc = start + np.linspace(0., t["pa_span_deg"], max(20, int(t["pa_span_deg"] * 2) + 1))
            rec = by_id.get(tid)
            cpa = (float(rec["tracked_cpa"]) if rec is not None and pd.notna(rec["tracked_cpa"])
                   else t["observed_cpa_deg"])
            for angle in (start, end):
                ex, ey = xy(rad, angle)
                ax.plot([sx, ex], [sy, ey], color=colour, lw=1.5)
                tx, ty = xy(max(12., rad * (.52 if angle == start else .60)), angle)
                ax.text(tx, ty, f"PA {angle:.0f}°", fontsize=7, color=colour,
                        ha="center", va="center", path_effects=stroke)
            cx, cy = xy(rad, cpa)
            ax.plot([sx, cx], [sy, cy], color=colour, lw=1.5, ls="--")
            cpx, cpy = xy(max(12., rad * .82), cpa)
            ax.text(cpx, cpy, f"CPA {cpa:.0f}°", fontsize=7, color=colour,
                    ha="center", va="center", path_effects=stroke)
            fx, fy = xy(rad, arc)
            ax.plot(fx, fy, color=colour, lw=2.3)
            ax.scatter(t["front_x"], t["front_y"], marker="*", s=65,
                       color="yellow", edgecolors="black", linewidths=.5, zorder=5)
            midx, midy = xy(min(123., rad + 3.), start + t["pa_span_deg"] / 2.)
            family_label = track_family_ids.get(tid, f"T{tid}") if track_family_ids else f"T{tid}"
            ax.text(midx, midy, f"{family_label} / T{tid} F {rad:.0f}px" if track_family_ids else f"T{tid} F {rad:.0f}px", fontsize=7,
                    color=colour, ha="center", va="center", path_effects=stroke)
            records.append({
                "pair_index": pair, "end_time": manifest.iloc[i]["end_time"],
                "track_id": tid, "pa_start": start, "pa_end": end,
                "tracked_cpa": cpa, "observed_front_radius_px": rad,
                "a5b_front_radius_px": float(rec["front_radius"]) if rec is not None else np.nan,
                "leading_front_state": str(rec["leading_front_state"]) if rec is not None else "UNAVAILABLE",
            })
        ax.set(xlim=(-.5,255.5), ylim=(-.5,255.5), xlabel="X (pixel)", ylabel="Y (pixel)")
        ax.set_aspect("equal")
        ax.set_title(f"{'Admitted Families' if included_track_ids is not None else 'A5b'} | Pair {pair} | {times.iloc[i].strftime('%Y-%m-%d %H:%M UTC')}\n"
                     f"{summary.iloc[i]['status']} | Active Tracks={len(tracks)}")
        fig.colorbar(im, ax=ax, shrink=.8, label="RD (DN/s)")
        fig.savefig(out / f"pair_{pair:03d}_{times.iloc[i].strftime('%Y%m%dT%H%M%SZ')}.png",
                    dpi=145)
        plt.close(fig)
    pd.DataFrame(records).to_csv(out / "track_geometry.csv", index=False)
    print(f"Tracking visualization: {len(indices)} frames in {out}")
    return out
