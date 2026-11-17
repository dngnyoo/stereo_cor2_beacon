"""Full-day A5b tracking diagnostic: one geometry per owned Track ID.

All disconnected P04 regions belonging to a track share a single PA/CPA/front
representation. No component-based geometry is overlaid on the image.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.utils.geometry import circular_mean_deg, minimal_circular_enclosing_arc


def _frame_tracks(owners: np.ndarray, patch_pa: np.ndarray,
                  patch_rad: np.ndarray):
    """One observed geometry per Track ID, aggregated over all owned P04 patches.

    PA is the outermost owned patch's position angle, CPA is the circular mean
    of all owned patch PAs; the minimal circular enclosing arc defines width.
    These are observed geometries, not the possibly predicted A5b track front.
    """
    tracks = []
    for tid in np.unique(owners):
        if tid <= 0:
            continue
        mask = owners == tid
        yy, xx = np.where(mask)
        pas = np.asarray(patch_pa[mask], dtype=float)
        radii = np.asarray(patch_rad[mask], dtype=float)
        if not len(pas):
            continue
        front_idx = int(np.nanargmax(radii))
        arc = minimal_circular_enclosing_arc(pas)
        tracks.append({
            "track_id": int(tid),
            "mask": mask,
            "n_patches": int(mask.sum()),
            "observed_cpa_deg": float(circular_mean_deg(pas)),
            "front_pa_deg": float(pas[front_idx]),
            "pa_span_deg": float(arc["arc_span"]),
            "pa_arc_start_deg": float(arc["arc_start"]),
            "observed_front_radius_px": float(radii[front_idx]),
            "front_x": float(xx[front_idx] * 4 + 1.5),
            "front_y": float(yy[front_idx] * 4 + 1.5),
        })
    return tracks


def compare_tracking(
    baseline_run: str | Path, experiment_run: str | Path, *,
    date: str, output_dir: str | Path | None = None,
) -> Path:
    """Render every date frame with exactly one PA/CPA/front set per owned track."""
    if len(date) != 8 or not date.isascii() or not date.isdigit():
        raise ValueError("date must be YYYYMMDD")
    base, exp = Path(baseline_run).resolve(), Path(experiment_run).resolve()
    btrack = json.loads((base / "tracking_metadata.json").read_text())
    etrack = json.loads((exp / "tracking_metadata.json").read_text())
    if btrack["method"] != etrack["method"] or btrack["method"] != "a5b":
        raise ValueError("Both runs must use A5b")
    bdet, edet = Path(btrack["detection_dir"]), Path(etrack["detection_dir"])
    bm = json.loads((bdet / "detection_metadata.json").read_text())
    em = json.loads((edet / "detection_metadata.json").read_text())
    pre = Path(bm["input_dir"]).resolve()
    if pre != Path(em["input_dir"]).resolve():
        raise ValueError("Both runs must use the same preprocessed RD")
    if btrack["requested_date"] != date or etrack["requested_date"] != date:
        raise ValueError("Tracking dates do not match")

    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    bs = pd.read_csv(bdet / "frame_detection_summary.csv")
    es = pd.read_csv(edet / "frame_detection_summary.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    bstate = np.load(bdet / "patch_state_p04.npy", mmap_mode="r")
    estate = np.load(edet / "patch_state_p04.npy", mmap_mode="r")
    bown = np.load(base / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    eown = np.load(exp / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    bt = pd.read_csv(base / "tables" / "live_tracking_states.csv")
    et = pd.read_csv(exp / "tables" / "live_tracking_states.csv")
    n = len(manifest)
    if (rd.shape != valid.shape or rd.shape != (n, 256, 256)
        or any(a.shape != (n, 64, 64) for a in (bstate, estate, bown, eown))
        or len(bs) != n or len(es) != n):
        raise ValueError("Input/detection/tracking frame shape mismatch")
    for summary in (bs, es):
        if not np.array_equal(summary["pair_index"].astype(int), manifest["pair_index"].astype(int)):
            raise ValueError("Pair indices do not align")
        if not np.array_equal(
            pd.to_datetime(summary["end_time"], utc=True).to_numpy(),
            pd.to_datetime(manifest["end_time"], utc=True).to_numpy()
        ):
            raise ValueError("Timestamps do not align")
    times = pd.to_datetime(manifest["end_time"], utc=True)
    indices = np.flatnonzero(times.dt.strftime("%Y%m%d").eq(date).to_numpy())
    if len(indices) == 0:
        raise ValueError(f"No frames ending on {date}")

    from astropy.io import fits
    from astropy.wcs import WCS
    from cme_pipeline.utils.geometry import compute_wcs_rasters

    hdr = fits.Header.fromstring(
        (pre / "target_wcs_header.txt").read_text(encoding="utf-8"), sep="\n"
    )
    wcs = WCS(hdr)
    sx, sy = wcs.world_to_pixel_values(0.0, 0.0)
    if not np.isfinite([sx, sy]).all():
        raise ValueError("Invalid WCS solar center")
    rasters = compute_wcs_rasters(wcs=wcs, center_xy=(sx, sy), patch_size=4)
    patch_pa, patch_rad = rasters["patch_pa"], rasters["patch_rad"]

    out = (Path(output_dir).resolve() if output_dir else
           exp / "visualizations" / "comparison_tracking_full_day")
    if out.exists():
        raise FileExistsError(
            f"Existing comparison at {out}; choose --output-dir or rename it before regeneration"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import to_hex
    from cme_pipeline.stages.segmentation.visualization import get_rd_cmap

    def pa_xy(radius_px, pa_deg):
        """North-up WCS convention: PA=0 north, 90 west, measured clockwise from N."""
        angle = np.deg2rad(pa_deg)
        return sx - radius_px * np.sin(angle), sy + radius_px * np.cos(angle)

    stroke = [patheffects.withStroke(linewidth=2.2, foreground="black")]

    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=False)
    track_rows, summaries = [], []
    for idx in indices:
        i = int(idx)
        pair = int(manifest.iloc[i]["pair_index"])
        raw = np.asarray(rd[i], dtype=float)
        usable = np.asarray(valid[i], dtype=bool) & np.isfinite(raw)
        image = np.ma.masked_where(~usable, raw)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        details = []
        for ax, run_name, det, owners, summary, track_df in (
            (axes[0], "Baseline + A5b", bstate[i], bown[i], bs.iloc[i], bt),
            (axes[1], "Outer P95 + A5b", estate[i], eown[i], es.iloc[i], et)
        ):
            grid = np.asarray(det) == 1
            ownership = np.asarray(owners, dtype=np.int32)
            if np.any((ownership > 0) & ~grid):
                raise ValueError(f"Owned patch not detected: pair {pair} ({run_name})")
            im = ax.imshow(image, origin="lower", cmap=get_rd_cmap(),
                           vmin=-600, vmax=600, interpolation="nearest")
            tracks = _frame_tracks(ownership, patch_pa, patch_rad)
            matched = track_df.loc[
                (track_df["pair_index"] == pair) & (track_df["state"] == "OBSERVED")
            ]
            by_track = {int(r["track_id"]): r for _, r in matched.iterrows()}
            for t in tracks:
                tid = t["track_id"]
                colour = to_hex(plt.get_cmap("tab20")(tid % 20))
                # Multiple disconnected regions may have one track ID; contours
                # still outline ALL owned regions, but geometry is drawn ONCE.
                ax.contour(np.arange(64) * 4 + 1.5, np.arange(64) * 4 + 1.5,
                           t["mask"].astype(float), levels=[0.5],
                           colors=[colour], linewidths=1.6)
                front_r = t["observed_front_radius_px"]
                front_pa = t["front_pa_deg"]
                state = by_track.get(tid)
                # Prefer actual A5b tracked CPA; preserve observed CPA separately.
                cpa = (float(state["tracked_cpa"]) if state is not None
                       and pd.notna(state["tracked_cpa"]) else t["observed_cpa_deg"])
                t["tracked_cpa_deg"] = cpa
                t["a5b_front_radius_px"] = (
                    float(state["front_radius"]) if state is not None else float("nan")
                )
                t["leading_front_state"] = (
                    str(state["leading_front_state"]) if state is not None else "UNAVAILABLE"
                )
                t["pair_index"] = pair
                t["run"] = run_name
                track_rows.append({k: v for k, v in t.items() if k != "mask"})

                arc_angles = t["pa_arc_start_deg"] + np.linspace(
                    0., t["pa_span_deg"], max(20, int(t["pa_span_deg"] * 2) + 1)
                )
                arc_x, arc_y = pa_xy(front_r, arc_angles)
                pa_x, pa_y = pa_xy(front_r, front_pa)
                cpa_x, cpa_y = pa_xy(front_r, cpa)
                ax.plot([sx, pa_x], [sy, pa_y], color=colour,
                        linewidth=1.5, linestyle="-", alpha=.9, zorder=4)
                ax.plot([sx, cpa_x], [sy, cpa_y], color=colour,
                        linewidth=1.5, linestyle="--", alpha=.9, zorder=4)
                ax.plot(arc_x, arc_y, color=colour,
                        linewidth=2.3, alpha=.95, zorder=5)
                ax.scatter([t["front_x"]], [t["front_y"]], marker="*",
                           s=65, facecolor="yellow", edgecolor="black",
                           linewidth=.5, zorder=6)
                # Only one compact label set per Track ID (not per island).
                id_x, id_y = pa_xy(min(123., front_r + 6.), cpa)
                ax.text(id_x, id_y, f"T{tid}", fontsize=8, fontweight="bold",
                        color=colour, ha="center", va="center",
                        path_effects=stroke, zorder=7)
                pa_lx, pa_ly = pa_xy(max(12., front_r * .52), front_pa)
                cpa_lx, cpa_ly = pa_xy(max(12., front_r * .81), cpa)
                f_ang = t["pa_arc_start_deg"] + t["pa_span_deg"] / 2.
                f_lx, f_ly = pa_xy(min(123., front_r + 3.), f_ang)
                for tx, ty, label_text in (
                    (pa_lx, pa_ly, f"PA {front_pa:.0f}°"),
                    (cpa_lx, cpa_ly, f"CPA {cpa:.0f}°"),
                    (f_lx, f_ly, f"F {front_r:.0f}px"),
                ):
                    ax.text(tx, ty, label_text, fontsize=7, color=colour,
                            ha="center", va="center", path_effects=stroke,
                            zorder=7)
            active_ids = np.unique(ownership[ownership > 0])
            ax.set_title(
                f"{run_name} | {summary['status']}\\n"
                f"Detection={int(grid.sum())}  Owned={int((ownership > 0).sum())} "
                f"Tracks={len(active_ids)}",
                fontsize=10
            )
            ax.set_xlim(-.5, 255.5)
            ax.set_ylim(-.5, 255.5)
            ax.set_aspect("equal")
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
            details.append((int(grid.sum()), int((ownership > 0).sum()),
                            len(active_ids)))
        fig.colorbar(im, ax=axes, label="RD (DN/s)", shrink=.77)
        fig.suptitle(
            f"A5b track geometry | Pair {pair} | "
            f"{times.iloc[i].strftime('%Y-%m-%d %H:%M UTC')}\n"
            "One geometry per track | Solid: front PA | Dashed: A5b tracked CPA "
            "| Arc: observed front radius over track PA span",
            fontsize=10
        )
        fig.savefig(frames / f"pair_{pair:03d}_{times.iloc[i].strftime('%Y%m%dT%H%M%SZ')}.png",
                    dpi=145)
        plt.close(fig)
        summaries.append({
            "pair_index": pair, "end_time": manifest.iloc[i]["end_time"],
            "baseline_status": bs.iloc[i]["status"],
            "outer_status": es.iloc[i]["status"],
            "baseline_active": details[0][0], "outer_active": details[1][0],
            "baseline_owned": details[0][1], "outer_owned": details[1][1],
            "baseline_track_ids": details[0][2], "outer_track_ids": details[1][2],
                    })
    pd.DataFrame(summaries).to_csv(out / "comparison_summary.csv", index=False)
    pd.DataFrame(track_rows).to_csv(out / "track_geometry.csv", index=False)
    print(f"Rendered {len(summaries)} frames. Baseline tracks={btrack['total_tracks']}; "
          f"Outer P95 tracks={etrack['total_tracks']}.")
    return out
