"""Read-only C3/C0 detection diagnostics using stored detector products."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cme_pipeline.live_detection import _scores, _smooth_and_z


def save_detection_diagnostics(
    detection_dir: str | Path, *,
    date: str | None = None,
    all_frames: bool = False,
    max_frames: int = 6,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Render RD, P04 scores and activated patch overlays; never rerun detection."""
    root = Path(detection_dir).expanduser().resolve()
    meta = json.loads((root / "detection_metadata.json").read_text())
    pre = Path(meta["input_dir"])
    if not pre.is_dir():
        raise FileNotFoundError(f"Preprocessed input no longer exists: {pre}")
    summary = pd.read_csv(root / "frame_detection_summary.csv")
    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    states = np.load(root / "patch_state_p04.npy", mmap_mode="r")
    n = len(manifest)
    if (len(summary) != n or rd.shape != (n, 256, 256)
            or valid.shape != rd.shape or states.shape != (n, 64, 64)):
        raise ValueError("Detection, manifest, RD and mask dimensions disagree")
    if not np.array_equal(
        summary["pair_index"].astype(int).to_numpy(),
        manifest["pair_index"].astype(int).to_numpy(),
    ):
        raise ValueError("Detection summary and RD manifest have different pair ordering")
    times = pd.to_datetime(manifest["end_time"], utc=True, errors="raise")
    wanted = str(date or meta["requested_date"])
    if len(wanted) != 8 or not wanted.isdigit():
        raise ValueError("date must be YYYYMMDD")
    candidates = np.flatnonzero(times.dt.strftime("%Y%m%d").eq(wanted).to_numpy())
    if not len(candidates):
        raise ValueError(f"No frames on {wanted}")
    if not all_frames:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        candidates = candidates[np.unique(np.linspace(
            0, len(candidates) - 1, min(max_frames, len(candidates)), dtype=int
        ))]
    dest = (Path(output_dir).expanduser().resolve() if output_dir else
            root / "visualizations" / "detection")
    paths = [
        dest / f"pair_{int(manifest.iloc[int(i)]['pair_index']):03d}_"
        f"{times.iloc[int(i)].strftime('%Y%m%dT%H%M%SZ')}_detection.png"
        for i in candidates
    ]
    if not overwrite:
        for path in paths:
            if path.exists():
                raise FileExistsError(f"Diagnostic already exists: {path}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.collections import PatchCollection
    from cme_pipeline.stages.segmentation.visualization import get_rd_cmap

    dest.mkdir(parents=True, exist_ok=True)
    for i, path in zip(candidates, paths):
        i = int(i)
        row = summary.iloc[i]
        raw = np.asarray(rd[i], dtype=float)
        usable = np.asarray(valid[i], dtype=bool) & np.isfinite(raw)
        masked = np.ma.masked_where(~usable, raw)
        z, _, _, _ = _smooth_and_z(raw, usable, denoiser=meta["denoiser"])
        if z is None:
            score = np.full((64, 64), np.nan)
            accepted = np.zeros((64, 64), bool)
        else:
            score, accepted = _scores(z, usable, 4)
        state = np.asarray(states[i])
        active = state == 1
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
        for ax in axes:
            ax.set_xlim(-0.5, 255.5)
            ax.set_ylim(-0.5, 255.5)
            ax.set_aspect("equal")
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
        im = axes[0].imshow(masked, origin="lower", cmap=get_rd_cmap(),
                            vmin=-600, vmax=600)
        fig.colorbar(im, ax=axes[0], shrink=.75, label="RD (DN/s)")
        axes[0].set_title("North-up RD")
        score_map = np.ma.masked_where(~accepted, score)
        sm = axes[1].imshow(score_map, origin="lower", extent=(-.5, 255.5, -.5, 255.5),
                             interpolation="nearest", cmap="viridis", vmin=0)
        fig.colorbar(sm, ax=axes[1], shrink=.75, label="P04 |Z| p95")
        axes[1].set_title("P04 patch activity score")
        axes[2].imshow(masked, origin="lower", cmap=get_rd_cmap(),
                       vmin=-600, vmax=600)
        yy, xx = np.nonzero(active)
        if len(yy):
            axes[2].add_collection(PatchCollection(
                [Rectangle((int(x) * 4 - .5, int(y) * 4 - .5), 4, 4)
                 for y, x in zip(yy, xx)],
                facecolor="none", edgecolor="red",
                linewidth=.65, alpha=.9,
            ))
        axes[2].set_title(f"Activated P04: {int(active.sum())} patches")
        if meta.get("floor_mode") in ("outer_background", "outer_background_mad", "outer_p95"):
            inner = meta.get("outer_floor_inner_px")
            outer = meta.get("outer_floor_outer_px")
            if inner is not None and outer is not None:
                for ax in axes:
                    for radius in (float(inner), float(outer)):
                        ax.add_patch(Circle((127.5, 127.5), radius, fill=False,
                                            linestyle="--", linewidth=1.2,
                                            edgecolor="cyan"))
        def fmt(value):
            try:
                x = float(value)
                return f"{x:.3f}" if np.isfinite(x) else "N/A"
            except (TypeError, ValueError):
                return "N/A"
        fig.suptitle(
            f"Pair {int(row['pair_index'])} | {times.iloc[i].strftime('%Y-%m-%d %H:%M UTC')} "
            f"| {meta['detector']} / {meta.get('floor_mode', 'legacy')}"
            f" (k={meta.get('outer_floor_k')}) | {row['status']}\n"
            f"T24={fmt(row['T24_p04'])} | Floor={fmt(row['floor_p04'])} "
            f"| Effective={fmt(row['effective_threshold_p04'])} "
            f"| Floor bound={row['floor_bound_p04']}",
            fontsize=11,
        )
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return paths
