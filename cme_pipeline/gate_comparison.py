"""Side-by-side full-day A5b recovery / identity / birth-size gate diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from cme_pipeline.tracking_comparison import _frame_tracks


def compare_gates(
    baseline_run: str | Path, experiment_run: str | Path, *,
    date: str, output_dir: str | Path | None = None,
) -> Path:
    if len(date) != 8 or not date.isascii() or not date.isdigit():
        raise ValueError("date must be YYYYMMDD")
    base, exp = Path(baseline_run).resolve(), Path(experiment_run).resolve()
    bm = json.loads((base / "gates_summary.json").read_text())
    em = json.loads((exp / "gates_summary.json").read_text())
    if bm["date"] != em["date"] or bm["date"] != date:
        raise ValueError("Date mismatch")
    if bm["recovery_inner_radius_px"] != em["recovery_inner_radius_px"]:
        raise ValueError("Recovery inner radius differs between experiments")
    if (bm["birth_max_radius_px"] != em["birth_max_radius_px"]
            or bm["min_detected_patches"] != em["min_detected_patches"]
            or bm.get("min_observed_pairs", 1) != em.get("min_observed_pairs", 1)):
        raise ValueError("Birth/size gate parameters differ")
    bd = json.loads((Path(bm["source_detection_dir"]) / "detection_metadata.json").read_text())
    ed = json.loads((Path(em["source_detection_dir"]) / "detection_metadata.json").read_text())
    pre = Path(bd["input_dir"]).resolve()
    if Path(ed["input_dir"]).resolve() != pre:
        raise ValueError("Comparison requires identical RD inputs")
    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    n = len(manifest)
    if rd.shape != valid.shape or rd.shape != (n, 256, 256):
        raise ValueError("Invalid RD shape")
    times = pd.to_datetime(manifest["end_time"], utc=True)
    indices = np.flatnonzero(times.dt.strftime("%Y%m%d").eq(date).to_numpy())
    if not len(indices):
        raise ValueError("No frame on date")

    # Family geometry uses the same WCS and patch centers as tracking plots.
    from astropy.io import fits
    from astropy.wcs import WCS
    from cme_pipeline.utils.geometry import compute_wcs_rasters
    header = fits.Header.fromstring(
        (pre / "target_wcs_header.txt").read_text(encoding="utf-8"), sep="\\n"
    )
    wcs = WCS(header)
    sx, sy = wcs.world_to_pixel_values(0.0, 0.0)
    if not np.isfinite([sx, sy]).all():
        raise ValueError("Invalid WCS solar center")
    rasters = compute_wcs_rasters(wcs=wcs, center_xy=(sx, sy), patch_size=4)
    patch_pa, patch_rad = rasters["patch_pa"], rasters["patch_rad"]

    def pa_xy(radius, pa_deg):
        theta = np.deg2rad(pa_deg)
        return sx - radius * np.sin(theta), sy + radius * np.cos(theta)

    data = []
    for run, meta in ((base, bm), (exp, em)):
        summary = pd.read_csv(run / "stages" / "size_gate" / "family_summary.csv")
        ownership = np.load(run / "arrays" / "saved_track_ownership.npy", mmap_mode="r")
        recovery = np.load(run / "arrays" / "recovered_patch_masks_pipeline_b_full_day.npy", mmap_mode="r")
        detection = np.load(run / "arrays" / "saved_detection_p04.npy", mmap_mode="r")
        if ownership.shape != recovery.shape or recovery.shape != detection.shape or ownership.shape != (n,64,64):
            raise ValueError("Gate input arrays mismatch")
        admitted = set()
        rejected = set()
        family_ids = {}
        for _, row in summary.iterrows():
            fid = str(row["family_id"])
            if not fid.startswith("FAM_T"):
                raise ValueError(f"Unexpected family mapping: {fid}")
            tid = int(fid.removeprefix("FAM_T"))
            family_ids[tid] = fid
            if row["status"] == "ADMITTED":
                admitted.add(tid)
            else:
                rejected.add(tid)
        data.append((meta, ownership, recovery, detection, admitted, rejected, family_ids))

    output = (Path(output_dir).resolve() if output_dir else
              exp / "visualizations" / "comparison_gates_full_day")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cme_pipeline.stages.segmentation.visualization import get_rd_cmap
    frames = output / "frames"
    frames.mkdir(parents=True)
    summaries = []
    geometry_rows = []
    for idx in indices:
        idx = int(idx)
        pair = int(manifest.iloc[idx]["pair_index"])
        image = np.asarray(rd[idx], dtype=float)
        image = np.ma.masked_where(~np.asarray(valid[idx], dtype=bool) | ~np.isfinite(image), image)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        row = {"pair_index": pair, "end_time": manifest.iloc[idx]["end_time"]}
        for ax, group, name in zip(axes, data, ("baseline", "outer_p95")):
            meta, ownership_all, recovery_all, detection_all, admitted, rejected, family_ids = group
            owners = np.asarray(ownership_all[idx])
            recovery = np.asarray(recovery_all[idx], bool)
            detection = np.asarray(detection_all[idx], bool)
            selected = np.isin(owners, list(admitted)) & (owners > 0)
            excluded = np.isin(owners, list(rejected)) & (owners > 0)
            im = ax.imshow(image, origin="lower", cmap=get_rd_cmap(),
                           vmin=-600, vmax=600, interpolation="nearest")
            def contour(mask, color, width):
                if np.any(mask) and not np.all(mask):
                    ax.contour(np.arange(64)*4+1.5, np.arange(64)*4+1.5,
                               mask.astype(float), levels=[.5],
                               colors=[color], linewidths=width)
            contour(detection, "#aaaaaa", .5)
            contour(selected, "#20c997", 1.9)
            contour(excluded, "#ff4757", 1.5)
            contour(recovery, "#00ccff", 1.1)
            # Identity Association: one A5b Track = one Family. Aggregate
            # all disconnected P04 components, drawing geometry once per Family.
            for family in _frame_tracks(owners, patch_pa, patch_rad):
                tid = family["track_id"]
                fid = family_ids.get(tid)
                if fid is None:
                    raise ValueError(f"Missing Family for Track {tid}")
                admitted_family = tid in admitted
                color = "#20c997" if admitted_family else "#ff4757"
                radius = family["observed_front_radius_px"]
                start = family["pa_arc_start_deg"]
                end = family["pa_arc_end_deg"]
                cpa = family["observed_cpa_deg"]
                angles = start + np.linspace(
                    0.0, family["pa_span_deg"],
                    max(20, int(family["pa_span_deg"] * 2) + 1),
                )
                ax.plot(*pa_xy(radius, angles), color=color,
                        linewidth=2.0, zorder=5)
                for bound in (start, end):
                    bx, by = pa_xy(radius, bound)
                    ax.plot([sx, bx], [sy, by], color=color,
                            linestyle="-", linewidth=1.2, zorder=4)
                cx, cy = pa_xy(radius, cpa)
                ax.plot([sx, cx], [sy, cy], color=color,
                        linestyle="--", linewidth=1.4, zorder=4)
                ax.scatter([family["front_x"]], [family["front_y"]],
                           marker="*", s=50, c=[color], edgecolors="black",
                           linewidths=.5, zorder=6)
                tx, ty = pa_xy(min(123., radius + 7.), cpa)
                ax.text(tx, ty, fid, fontsize=7, ha="center", va="center",
                        color="white", bbox=dict(facecolor=color, alpha=.8, pad=1),
                        zorder=7)
                geometry_rows.append({
                    "run": name, "pair_index": pair, "family_id": fid,
                    "gate_status": "ADMITTED" if admitted_family else "REJECTED",
                    "observed_patches": family["n_patches"],
                    "pa_start_deg": start, "pa_end_deg": end,
                    "pa_span_deg": family["pa_span_deg"],
                    "observed_cpa_deg": cpa,
                    "observed_front_radius_px": radius,
                })
            ax.set_title(
                f"{name} | admitted families {meta['size_gate_admitted']}\n"
                f"Detected {int(detection.sum())} | owned {int((owners>0).sum())} "
                f"| recovered {int(recovery.sum())}", fontsize=10
            )
            ax.set_xlim(-.5,255.5)
            ax.set_ylim(-.5,255.5)
            ax.set_aspect("equal")
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
            row[f"{name}_owned"] = int((owners > 0).sum())
            row[f"{name}_admitted_owned"] = int(selected.sum())
            row[f"{name}_rejected_owned"] = int(excluded.sum())
            row[f"{name}_recovered"] = int(recovery.sum())
        fig.colorbar(im, ax=axes, shrink=.75, label="RD (DN/s)")
        fig.suptitle(
            f"Recovery + Identity + Birth/Size Gates | Pair {pair} | "
            f"{times.iloc[idx].strftime('%Y-%m-%d %H:%M UTC')}\n"
            "Gray: detection | Green: admitted Family | Red: rejected Family | Cyan: recovery\\n"
            "Solid: PA start/end | Dashed: observed CPA | Arc: observed front",
            fontsize=10,
        )
        fig.savefig(frames / f"pair_{pair:03d}_{times.iloc[idx].strftime('%Y%m%dT%H%M%SZ')}.png", dpi=140)
        plt.close(fig)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output / "comparison_summary.csv", index=False)
    pd.DataFrame(geometry_rows, columns=[
        "run", "pair_index", "family_id", "gate_status", "observed_patches",
        "pa_start_deg", "pa_end_deg", "pa_span_deg", "observed_cpa_deg",
        "observed_front_radius_px",
    ]).to_csv(output / "family_geometry.csv", index=False)
    (output / "gate_counts.json").write_text(json.dumps({
        name: {key: meta[key] for key in ("tracks", "candidate_families",
                "birth_gate_pass", "size_gate_admitted", "total_recovered_patches_all_frames")}
        for name, (meta, *_rest) in zip(("baseline", "outer_p95"), data)
    }, indent=2))
    print(f"Rendered {len(summaries)} full-day frames to {output}")
    return output
