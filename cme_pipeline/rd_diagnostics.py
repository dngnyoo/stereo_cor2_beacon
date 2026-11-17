"""Optional, read-only RD diagnostic PNGs for an existing preprocessing run.

Reuses the segmentation visualization color scale and valid-pixel treatment.
Never recomputes or modifies RD, masks, manifests, or detection products.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cme_pipeline.stages.segmentation.visualization import get_rd_cmap


def save_rd_diagnostics(
    preprocessed_dir: str | Path,
    *,
    date: str | None = None,
    all_frames: bool = False,
    max_frames: int = 3,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
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
    if existing and not overwrite:
        raise FileExistsError(f"RD diagnostic already exists: {existing[0]}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target.mkdir(parents=True, exist_ok=True)
    for i, path in zip(indices, outputs):
        i = int(i)
        pair = manifest.iloc[i]
        image = np.ma.masked_where(
            ~np.asarray(valid[i], dtype=bool) | ~np.isfinite(rd[i]), rd[i]
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(
            image,
            origin="lower",
            cmap=get_rd_cmap(),
            vmin=-600,
            vmax=600,
            interpolation="nearest",
        )
        ax.set_title(f"Pair {pair['pair_index']} | {pair['end_time']} UTC")
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        fig.colorbar(im, ax=ax, label="RD (DN/s)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return outputs
