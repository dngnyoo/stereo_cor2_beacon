"""Read-only family-resolved Gate/Recovery/Segmentation diagnostics.

Every frame uses the saved A5b ownership, original north-up RD, target WCS
and identical track-ID palette. Family NPZ files are authoritative; never
reconstruct recovered/segmented membership from the union mask.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.run_layout import verify_stage
from cme_pipeline.tracking_comparison import _frame_tracks
from cme_pipeline.stages.segmentation.visualization import get_rd_cmap


def _saved(root, stage, key):
    meta = verify_stage(root, stage)
    return root / meta["outputs"][key]["path"]


def _family_id(tid):
    return f"FAM_T{int(tid):02d}"


def _palette(fid):
    """Match the current A5b tracking renderer's tab20(track_id % 20)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    if not fid.startswith("FAM_T") or not fid[5:].isdigit():
        raise ValueError(f"Unexpected identity Family ID: {fid!r}")
    return to_hex(plt.get_cmap("tab20")(int(fid[5:]) % 20))


def _load(run, stage):
    runs = run.parent
    first = runs / "01_02_preprocessing_detection"
    pre = first / "01_preprocessing/data/acquisition/preprocessed"
    if not pre.is_dir():
        pre = first / "01_preprocessing/data"
    track = runs / "03_tracking"
    gates = runs / "05_gates"
    recovery = runs / "06_recovery"
    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    owners = np.load(_saved(track, "tracking", "track_ownership"), mmap_mode="r")
    states = pd.read_csv(_saved(track, "tracking", "track_states"))
    decisions = pd.read_csv(_saved(gates, "gates", "gate_decisions"))
    if not {"pair_index", "track_id", "state"}.issubset(states):
        raise ValueError("Tracking states are missing required columns")
    if not {"family_id", "status", "max_detected_patches"}.issubset(decisions):
        raise ValueError("Gate decisions are missing required columns")
    if len(decisions) != decisions.family_id.nunique():
        raise ValueError("Duplicate Gate Family IDs")
    if (len(manifest) != len(rd) or rd.shape != valid.shape or
            rd.shape[1:] != (256, 256) or owners.shape != (len(rd), 64, 64)):
        raise ValueError("Saved input shapes disagree")
    from astropy.io import fits
    from astropy.wcs import WCS
    from cme_pipeline.utils.geometry import compute_wcs_rasters
    wcs = WCS(fits.Header.fromstring(
        (pre / "target_wcs_header.txt").read_text(encoding="utf-8"), sep="\n"))
    sx, sy = wcs.world_to_pixel_values(0., 0.)
    if not np.isfinite([sx, sy]).all():
        raise ValueError("Invalid solar center from saved WCS")
    rasters = compute_wcs_rasters(wcs=wcs, center_xy=(sx, sy), patch_size=4)
    family = {}
    if stage == "recovery":
        with np.load(_saved(recovery, "recovery", "family_recovered_patch_masks"),
                     allow_pickle=False) as npz:
            family = {key.removeprefix("rec_"): np.asarray(npz[key])
                      for key in npz.files if key.startswith("rec_")}
        target_shape = (len(rd), 64, 64)
    elif stage == "segmentation":
        with np.load(_saved(run, "segmentation", "family_pixel_masks"),
                     allow_pickle=False) as npz:
            family = {key.removeprefix("mask_"): np.asarray(npz[key])
                      for key in npz.files if key.startswith("mask_")}
        target_shape = (len(rd), 256, 256)
    else:
        target_shape = None
    if stage in ("recovery", "segmentation"):
        admitted = set(decisions.loc[decisions.status == "ADMITTED", "family_id"].astype(str))
        if set(family) != admitted:
            raise ValueError("Saved Family masks and Gate admission disagree")
        if any(value.shape != target_shape or value.dtype != np.bool_
               for value in family.values()):
            raise ValueError(f"Invalid {stage} Family mask geometry")
    return manifest, rd, valid, owners, states, decisions, rasters, (float(sx), float(sy)), family


def _xy(rad, pa, center):
    angle = np.deg2rad(pa)
    return center[0] - rad * np.sin(angle), center[1] + rad * np.cos(angle)


def _contour(ax, mask, color, *, pixel=False, lw=1.35, ls="solid"):
    if not np.any(mask):
        return
    if pixel:
        ax.contour(mask.astype(float), levels=[.5], origin="lower",
                   extent=(0, 256, 0, 256), colors=[color], linewidths=lw,
                   linestyles=ls, zorder=8)
    else:
        ax.contour(np.arange(64)*4+1.5, np.arange(64)*4+1.5,
                   mask.astype(float), levels=[.5], colors=[color],
                   linewidths=lw, linestyles=ls, zorder=8)


def _annotate(ax, fid, track, center, row, *, detailed=True, color=None):
    """PA boundaries, tracked CPA, front arc and ID: A5b plot semantics."""
    from matplotlib import patheffects
    color = color or _palette(fid)
    rad = float(track["observed_front_radius_px"])
    start = float(track["pa_arc_start_deg"])
    end = float(track["pa_arc_end_deg"])
    span = float(track["pa_span_deg"])
    cpa = (float(row["tracked_cpa"]) if row is not None and
           pd.notna(row.get("tracked_cpa")) else float(track["observed_cpa_deg"]))
    if not np.isfinite([rad, start, end, span, cpa]).all():
        return
    if detailed:
        for angle in (start, end):
            ex, ey = _xy(rad, angle, center)
            ax.plot([center[0], ex], [center[1], ey], color=color, lw=1.25)
        cx, cy = _xy(rad, cpa, center)
        ax.plot([center[0], cx], [center[1], cy], color=color, lw=1.1, ls="--")
        arc = start + np.linspace(0, span, max(20, int(span*2)+1))
        ax.plot(*_xy(rad, arc, center), color=color, lw=1.8)
        ax.scatter(track["front_x"], track["front_y"], marker="*",
                   color="yellow", edgecolors="black", linewidths=.5, s=40, zorder=12)
    x, y = _xy(min(121., rad+4.), start+span/2., center)
    ax.text(x, y, fid, color=color, fontsize=8, ha="center", va="center",
            path_effects=[patheffects.withStroke(linewidth=2.4, foreground="black")],
            zorder=13)


def _base(ax, rd, valid, title):
    from matplotlib.patches import Patch
    image = np.ma.masked_where(~valid.astype(bool) | ~np.isfinite(rd), rd)
    ax.set_facecolor("black")
    artist = ax.imshow(image, origin="lower", cmap=get_rd_cmap(),
                       vmin=-600, vmax=600, interpolation="nearest")
    ax.set(xlim=(-.5, 255.5), ylim=(-.5, 255.5), xlabel="X (pixel)",
           ylabel="Y (pixel)", title=title, aspect="equal")
    return artist


def _indices(manifest, date, mode, scores):
    dates = pd.to_datetime(manifest["end_time"], utc=True).dt.strftime("%Y%m%d")
    candidates = np.flatnonzero(dates.eq(str(date)).to_numpy())
    if not len(candidates):
        raise ValueError(f"No evaluation-date frames for {date}")
    if mode == "all":
        return candidates
    # First/last, strongest and most informative overlap frame, without
    # depending on manual event labels or plotting every warmup frame.
    ranked = sorted(candidates, key=lambda i: (float(scores[int(i)]), -int(i)), reverse=True)
    picks = {int(candidates[0]), int(candidates[-1])}
    picks.update(map(int, ranked[:4]))
    return np.array(sorted(picks), dtype=int)


def render_stage(config, stage, run, destination, mode):
    """Render Family overlays + framewise CSV, with no algorithm recomputation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if stage not in ("gates", "recovery", "segmentation"):
        raise ValueError(stage)
    manifest, rd, valid, owners, states, decisions, rasters, center, families = _load(run, stage)
    by_family = decisions.set_index("family_id")
    admitted = set(decisions.loc[decisions.status == "ADMITTED", "family_id"].astype(str))
    n = len(rd)
    scores = np.zeros(n, dtype=float)
    rows = []
    for i in range(n):
        owned = owners[i]
        observed_ids = set(np.unique(owned[owned>0]).astype(int))
        active = {_family_id(t) for t in observed_ids}
        counts = {"observed_patches": int((owned>0).sum()),
                  "admitted_observed_patches": int(np.isin(owned,
                      [int(fid[5:]) for fid in admitted]).sum()),
                  "active_families": len(active),
                  "active_admitted_families": len(active & admitted)}
        if stage != "gates":
            frame_masks = {fid: data[i] for fid, data in families.items()}
            occupied = np.zeros_like(next(iter(frame_masks.values()))) if frame_masks else (
                np.zeros((64,64), bool) if stage == "recovery" else np.zeros((256,256), bool))
            overlap = np.zeros_like(occupied, dtype=np.uint16)
            for mask in frame_masks.values():
                occupied |= mask
                overlap += mask.astype(np.uint16)
            counts["union_pixels_or_patches"] = int(occupied.sum())
            counts["overlap_pixels_or_patches"] = int((overlap > 1).sum())
            if stage == "recovery":
                new = np.zeros((64,64), bool)
                for fid, mask in frame_masks.items():
                    new |= mask & ~(owned == int(fid[5:]))
                counts["recovered_only_patches"] = int(new.sum())
            scores[i] = counts["union_pixels_or_patches"] + 2*counts["overlap_pixels_or_patches"]
        else:
            scores[i] = counts["active_admitted_families"] + counts["observed_patches"]/4096.
        rows.append({"pair_index": int(manifest.iloc[i]["pair_index"]),
                     "end_time": str(manifest.iloc[i]["end_time"]), **counts})
    chosen = _indices(manifest, config.data["experiment"]["date"], mode, scores)
    destination.mkdir(parents=True, exist_ok=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(destination / "frame_metrics.csv", index=False)
    if stage == "gates":
        decisions.to_csv(destination / "family_gate_decisions.csv", index=False)
    (destination / "family_palette.json").write_text(json.dumps({
        fid: _palette(fid) for fid in sorted(by_family.index.astype(str))}, indent=2) + "\n")
    output = []
    for idx in chosen:
        i = int(idx)
        pair = int(manifest.iloc[i]["pair_index"])
        observed = owners[i]
        tracks = _frame_tracks(observed, rasters["patch_pa"], rasters["patch_rad"])
        observed_states = states[(states.pair_index == pair) & states.state.eq("OBSERVED")]
        state_lookup = {int(r.track_id): r for _, r in observed_states.iterrows()}
        panels = ("candidates", "admitted") if stage == "gates" else (
            ("observed", "observed_plus_recovered") if stage == "recovery"
            else ("union", "families"))
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        for ax, panel in zip(axes, panels):
            im = _base(ax, rd[i], valid[i], panel.replace("_", " ").title())
            if stage == "gates":
                for t in tracks:
                    fid = _family_id(t["track_id"])
                    if fid not in by_family.index:
                        raise ValueError(f"Ownership track {fid} missing from Gate table")
                    accepted = fid in admitted
                    if panel == "admitted" and not accepted:
                        continue
                    color = _palette(fid) if accepted else "#777777"
                    qualified_text = str(by_family.loc[fid].get("size_qualified_pairs", "[]"))
                    qualified = set(ast.literal_eval(qualified_text))
                    size_qualified_now = pair in qualified
                    _contour(ax, t["mask"], color,
                             lw=1.9 if size_qualified_now and accepted else 1.1,
                             ls="solid" if size_qualified_now else "dashed")
                    _annotate(ax, fid, t, center, state_lookup.get(t["track_id"]),
                              detailed=accepted, color=color)
                ax.legend(handles=[Patch(edgecolor="#00ffdd", facecolor="none",label="Admitted (family color)"),
                                   Patch(edgecolor="#777777",facecolor="none",label="Rejected"),
                                   Patch(edgecolor="white",facecolor="none",label="Solid: size qualified"),
                                   Patch(edgecolor="white",facecolor="none",label="Dashed: not qualified")],
                          loc="lower right", fontsize=8)
            elif stage == "recovery":
                overlap = np.zeros((64,64), dtype=np.uint16)
                for mask_stack in families.values():
                    overlap += mask_stack[i].astype(np.uint16)
                for fid, mask_stack in sorted(families.items()):
                    tid = int(fid[5:])
                    observed_family = observed == tid
                    color = _palette(fid)
                    if panel == "observed_plus_recovered":
                        extra = mask_stack[i] & ~observed_family
                        if extra.any():
                            from matplotlib.colors import to_rgba
                            rgba = np.zeros((64,64,4))
                            rgba[extra] = to_rgba(color, alpha=.28)
                            ax.imshow(rgba, origin="lower", extent=(-.5,255.5,-.5,255.5),
                                      interpolation="nearest", zorder=4)
                            _contour(ax, extra, color, ls="dashed", lw=.9)
                    _contour(ax, observed_family, color, lw=1.5)
                    record = next((t for t in tracks if int(t["track_id"]) == tid), None)
                    if record is not None:
                        _annotate(ax, fid, record, center, state_lookup.get(tid),
                                  detailed=(panel=="observed_plus_recovered"), color=color)
                if panel == "observed_plus_recovered":
                    _contour(ax, overlap > 1, "#ff00ff", lw=2.0, ls="dashed")
            else:
                overlaps = np.zeros((256,256), dtype=np.uint16)
                for fid, stack in sorted(families.items()):
                    mask = stack[i]
                    overlaps += mask.astype(np.uint16)
                    if panel == "families":
                        color = _palette(fid)
                        if mask.any():
                            from matplotlib.colors import to_rgba
                            rgba = np.zeros((256,256,4))
                            rgba[mask] = to_rgba(color, alpha=.32)
                            ax.imshow(rgba, origin="lower", extent=(0,256,0,256),
                                      interpolation="nearest", zorder=3)
                            _contour(ax, mask, color, pixel=True, lw=1.2)
                            yy, xx = np.nonzero(mask)
                            ax.text(float(xx.mean())+.5,float(yy.mean())+.5,fid,
                                    color=color, fontsize=8, ha="center", va="center",
                                    bbox={"facecolor":"black","alpha":.5,"edgecolor":"none"})
                if panel == "union":
                    combined = overlaps > 0
                    _contour(ax, combined, "#00FF66", pixel=True, lw=1.45)
                if panel == "families":
                    # Outline the saved Recovery union as a thin dotted
                    # comparison guide, never as a segmentation family mask.
                    recovery_root = run.parent / "06_recovery"
                    support = np.load(
                        _saved(recovery_root, "recovery", "recovered_patch_masks"),
                        mmap_mode="r")[i]
                    if support.shape != (64,64):
                        raise ValueError("Saved Recovery support must be 64x64")
                    _contour(ax, support, "#ffffff", lw=.65, ls="dotted")
                    _contour(ax, overlaps > 1, "#ff00ff", pixel=True, lw=1.3, ls="dashed")
            ax.set_aspect("equal")
        counts = rows[i]
        fig.suptitle(f"{stage.capitalize()} | Pair {pair} | "
                     f"{pd.to_datetime(manifest.iloc[i]['end_time'],utc=True).strftime('%Y-%m-%d %H:%M UTC')}\n"
                     f"Admitted families={len(admitted)} | " +
                     (f"Overlap={counts.get('overlap_pixels_or_patches',0)}" if stage != "gates"
                      else f"Active candidates={counts['active_families']}"), fontsize=12)
        fig.colorbar(im, ax=axes, shrink=.68, label="RD (DN/s)")
        filename = destination / f"pair_{pair:03d}_{stage}.png"
        fig.savefig(filename, dpi=135)
        plt.close(fig)
        output.append(filename)
    return output
