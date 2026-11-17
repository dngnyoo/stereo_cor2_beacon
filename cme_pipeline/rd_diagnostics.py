"""Optional, read-only RD diagnostic PNGs for an existing preprocessing run.

Reuses the segmentation visualization color scale and valid-pixel treatment.
Never recomputes or modifies RD, masks, manifests, or detection products.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cme_pipeline.stages.segmentation.visualization import draw_rd_background


def save_rd_diagnostics(
    preprocessed_dir: str | Path,
    *,
    date: str | None = None,
    all_frames: bool = False,
    max_frames: int = 3,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Render sampled (or all) RD frames for the requested UTC date.

    By default, evenly sample three evaluation-date frames, rather than
    silently processing all preceding warmup frames.
    """
    root = Path(preprocessed_dir).expanduser().resolve()
    meta = json.loads((root / "preprocessing_metadata.json").read_text())
    if meta.get("unit") != "DN/s":
        raise ValueError("RD diagnostics require DN/s preprocessing products")
    target_date = str(date or meta["requested_date"])
    if len(target_date) != 8 or not target_date.isascii() or not target_date.isdigit():
        raise ValueError("date must be YYYYMMDD")
    manifest = pd.read_csv(root / "01_rd_manifest.csv")
    rd = np.load(root / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(root / "north_up_pair_valid_masks.npy", mmap_mode="r")
    if (rd.shape != valid.shape or rd.ndim != 3 or
            rd.shape[1:] != (256, 256) or len(manifest) != len(rd)):
        raise ValueError("Inconsistent RD, valid mask or manifest dimensions")
    times = pd.to_datetime(manifest["end_time"], utc=True, errors="raise")
    matching = np.flatnonzero(times.dt.strftime("%Y%m%d").eq(target_date).to_numpy())
    if not matching.size:
        raise ValueError(f"No RD pairs ending on {target_date}")
    if not all_frames:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        indices = matching[np.unique(np.linspace(
            0, len(matching) - 1, min(max_frames, len(matching)), dtype=int))]
    else:
        indices = matching
    target = (Path(output_dir).expanduser().resolve() if output_dir is not None else
              root.parent / "visualizations" / "rd")
    outputs = []
    for i in indices:
        stamp = times.iloc[int(i)].strftime("%Y%m%dT%H%M%SZ")
        pair = int(manifest.iloc[int(i)]["pair_index"])
        outputs.append(target / f"pair_{pair:03d}_{stamp}_rd.png")
    # Do not silently overwrite existing diagnostic images.
    existing = [p for p in outputs if p.exists()]
    if existing:
        raise FileExistsError(f"RD diagnostic already exists: {existing[0]}")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    target.mkdir(parents=True, exist_ok=True)
    for i, path in zip(indices, outputs):
        i = int(i)
        fig = Figure(figsize=(7.2, 7.0), dpi=120, facecolor="#0e0e0e")
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=.08, right=.94, bottom=.15, top=.88)
        im = draw_rd_background(ax, rd[i], valid[i])
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.set_xlabel("X (pixel)", color="white")
        ax.set_ylabel("Y (pixel)", color="white")
        ax.tick_params(colors="white", labelsize=8)
        fig.suptitle(
            f"COR2-A RD | Pair {int(manifest.iloc[i]['pair_index'])} | "
            f"{times.iloc[i].strftime('%Y-%m-%d %H:%M:%S')} UTC",
            color="white", fontsize=10, y=.96,
        )
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          pad=.10, fraction=.045)
        cb.set_label("RD (DN/s) | Fixed [-600, +600]", color="white", fontsize=8)
        cb.ax.tick_params(colors="white", labelsize=7)
        fig.savefig(path, facecolor=fig.get_facecolor())
        fig.clear()
    return outputs
