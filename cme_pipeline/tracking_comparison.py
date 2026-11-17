"""Full-day read-only A5b tracking diagnostics with observed track/component geometry.

A component here means an 8-connected region of P04 patches *owned by one track*.
This is not necessarily the original C3 raw component after contested allocation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

from cme_pipeline.utils.geometry import circular_mean_deg, minimal_circular_enclosing_arc


def _frame_components(owners: np.ndarray, patch_pa: np.ndarray,
                      patch_rad: np.ndarray, sx: float, sy: float):
    """Compute visually connected components of individual ownership masks."""
    components = []
    for tid in np.unique(owners):
        if tid <= 0:
            continue
        labels, n = label(owners == tid, structure=np.ones((3, 3), bool))
        for cid in range(1, n + 1):
            mask = labels == cid
            ys, xs = np.where(mask)
            pa = np.asarray(patch_pa[mask], dtype=float)
            radial = np.asarray(patch_rad[mask], dtype=float)
            if not len(pa):
                continue
            front_i = int(np.nanargmax(radial))
            center_pa = float(circular_mean_deg(pa))
            arc = minimal_circular_enclosing_arc(pa)
            front_y = float(ys[front_i] * 4 + 1.5)
            front_x = float(xs[front_i] * 4 + 1.5)
            components.append({
                "track_id": int(tid),
                "component_index": cid,
                "mask": mask,
                "n_patches": int(len(xs)),
                "center_x": float(np.mean(xs) * 4 + 1.5),
                "center_y": float(np.mean(ys) * 4 + 1.5),
                "cpa_deg": center_pa,
                "front_pa_deg": float(pa[front_i]),
                "pa_span_deg": float(arc["arc_span"]),
                "pa_arc_start_deg": float(arc["arc_start"]),
                "front_radius_px": float(radial[front_i]),
                "front_x": front_x,
                "front_y": front_y,
            })
    return components


def compare_tracking(
    baseline_run: str | Path, experiment_run: str | Path, *,
    date: str, output_dir: str | Path | None = None,
) -> Path:
    """Render every date frame, with track/component labels and measured fronts."""
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
    brawcomp = np.load(bdet / "patch_component_labels.npy", mmap_mode="r")
    erawcomp = np.load(edet / "patch_component_labels.npy", mmap_mode="r")
    estate = np.load(edet / "patch_state_p04.npy", mmap_mode="r")
    bown = np.load(base / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    eown = np.load(exp / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    bt = pd.read_csv(base / "tables" / "live_tracking_states.csv")
    et = pd.read_csv(exp / "tables" / "live_tracking_states.csv")
    n = len(manifest)
    if (rd.shape != valid.shape or rd.shape != (n, 256, 256)
        or any(a.shape != (n, 64, 64) for a in (bstate, estate, bown, eown, brawcomp, erawcomp))
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
    component_rows, summaries = [], []
    for idx in indices:
        i = int(idx)
        pair = int(manifest.iloc[i]["pair_index"])
        raw = np.asarray(rd[i], dtype=float)
        usable = np.asarray(valid[i], dtype=bool) & np.isfinite(raw)
        image = np.ma.masked_where(~usable, raw)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        details = []
        for ax, run_name, det, owners, raw_components, summary, track_df in (
            (axes[0], "Baseline + A5b", bstate[i], bown[i], brawcomp[i], bs.iloc[i], bt),
            (axes[1], "Outer P95 + A5b", estate[i], eown[i], erawcomp[i], es.iloc[i], et)
        ):
            grid = np.asarray(det) == 1
            ownership = np.asarray(owners, dtype=np.int32)
            if np.any((ownership > 0) & ~grid):
                raise ValueError(f"Owned patch not detected: pair {pair} ({run_name})")
            im = ax.imshow(image, origin="lower", cmap=get_rd_cmap(),
                           vmin=-600, vmax=600, interpolation="nearest")
            components = _frame_components(ownership, patch_pa, patch_rad, sx, sy)
            matched = track_df.loc[
                (track_df["pair_index"] == pair) & (track_df["state"] == "OBSERVED")
            ]
            by_track = {
                int(r["track_id"]): r for _, r in matched.iterrows()
            }
            for c in components:
                tid = c["track_id"]
                # Connected contours separate islands even within one track.
                colour = to_hex(plt.get_cmap("tab20")(int(tid) % 20))
                mask = c["mask"].astype(float)
                ax.contour(np.arange(64) * 4 + 1.5, np.arange(64) * 4 + 1.5,
                           mask, levels=[0.5], colors=[colour], linewidths=1.6)
                ax.scatter([c["front_x"]], [c["front_y"]], marker="*",
                           s=60, facecolor="yellow", edgecolor="black", linewidth=.5,
                           zorder=5)
                raw_ids = np.asarray(raw_components)[c["mask"]]
                raw_ids = raw_ids[raw_ids > 0]
                raw_component_id = int(np.bincount(raw_ids.astype(int)).argmax()) if len(raw_ids) else -1
                c["raw_component_id"] = raw_component_id
                c["run"] = run_name
                c["pair_index"] = pair
                state = by_track.get(tid)
                c["tracked_cpa_deg"] = (
                    float(state["tracked_cpa"]) if state is not None else float("nan")
                )
                c["track_front_radius_px"] = (
                    float(state["front_radius"]) if state is not None else float("nan")
                )
                component_rows.append({k: v for k, v in c.items() if k != "mask"})
                # Geometry drawn as lines, not opaque text boxes obscuring the image.
                # Arc starts at the actual minimal circular enclosing PA interval,
                # not at mean CPA +/- span/2 (those are not generally identical).
                front_r = c["front_radius_px"]
                front_pa = c["front_pa_deg"]
                cpa = c["cpa_deg"]
                arc_angles = c["pa_arc_start_deg"] + np.linspace(
                    0., c["pa_span_deg"], max(20, int(c["pa_span_deg"] * 2) + 1)
                )
                arc_x, arc_y = pa_xy(front_r, arc_angles)
                pa_x, pa_y = pa_xy(front_r, front_pa)
                cpa_x, cpa_y = pa_xy(front_r, cpa)
                ax.plot([sx, pa_x], [sy, pa_y], color=colour,
                        linewidth=1.25, linestyle="-", alpha=.85, zorder=4)
                ax.plot([sx, cpa_x], [sy, cpa_y], color=colour,
                        linewidth=1.25, linestyle="--", alpha=.95, zorder=4)
                ax.plot(arc_x, arc_y, color=colour,
                        linewidth=2.4, alpha=.95, zorder=5)
                # Component identity is kept near its boundary, rather than
                # covering the component with a multiline callout.
                id_x, id_y = pa_xy(min(123., front_r + 7.), cpa)
                ax.text(id_x, id_y, f"T{tid}/C{raw_component_id}.{c['component_index']}",
                        fontsize=7, color=colour, fontweight="bold",
                        ha="center", va="center", path_effects=stroke, zorder=6)
                # Very small islands keep only the contour and ID; showing
                # three angular labels on a tiny component would be illegible.
                if c["n_patches"] >= 4:
                    pa_label_x, pa_label_y = pa_xy(max(12., front_r * .56), front_pa)
                    cpa_label_x, cpa_label_y = pa_xy(max(12., front_r * .82), cpa)
                    front_label_angle = c["pa_arc_start_deg"] + c["pa_span_deg"] / 2.
                    f_label_x, f_label_y = pa_xy(min(123., front_r + 4.), front_label_angle)
                    for x, y, text_value in (
                        (pa_label_x, pa_label_y, f"PA {front_pa:.0f}°"),
                        (cpa_label_x, cpa_label_y, f"CPA {cpa:.0f}°"),
                        (f_label_x, f_label_y, f"F {front_r:.0f}px"),
                    ):
                        ax.text(x, y, text_value, fontsize=7, color=colour,
                                ha="center", va="center", path_effects=stroke,
                                zorder=6)
            active_ids = np.unique(ownership[ownership > 0])
            # A5b track-level CPA and front are the actual stored tracking states.
            lines = []
            for tid in active_ids:
                record = by_track.get(int(tid))
                if record is not None:
                    lines.append(
                        f"T{tid} CPA={float(record['tracked_cpa']):.1f}° "
                        f"front={float(record['front_radius']):.1f}px "
                        f"({record['leading_front_state']})"
                    )
            ax.set_title(
                f"{run_name} | {summary['status']}\n"
                f"Detection={int(grid.sum())}  Owned={int((ownership > 0).sum())} "
                f"Tracks={len(active_ids)}  Components={len(components)}",
                fontsize=10
            )
            # All per-track numerical state is preserved in the CSV rather
            # than an opaque legend covering the lower-left image region.
            ax.set_xlim(-.5, 255.5)
            ax.set_ylim(-.5, 255.5)
            ax.set_aspect("equal")
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
            details.append((int(grid.sum()), int((ownership > 0).sum()),
                            len(active_ids), len(components)))
        fig.colorbar(im, ax=axes, label="RD (DN/s)", shrink=.77)
        fig.suptitle(
            f"A5b track/component geometry | Pair {pair} | "
            f"{times.iloc[i].strftime('%Y-%m-%d %H:%M UTC')}\n"
            "Contours: track-owned components | Solid: PA | Dashed: CPA | Arc: outer front "
            "| Labels: front PA, component CPA, front radius (px)",
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
            "baseline_components": details[0][3], "outer_components": details[1][3],
        })
    pd.DataFrame(summaries).to_csv(out / "comparison_summary.csv", index=False)
    pd.DataFrame(component_rows).to_csv(out / "component_geometry.csv", index=False)
    print(f"Rendered {len(summaries)} frames. Baseline tracks={btrack['total_tracks']}; "
          f"Outer P95 tracks={etrack['total_tracks']}.")
    return out
