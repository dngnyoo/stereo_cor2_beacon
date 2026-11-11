"""Generate scientifically labeled diagnostic figures from frozen COR2 masks.

All figures are derived from pipeline segmentation and measurement outputs.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def create_visualizations(segmentation, frames: pd.DataFrame, output_dir: Path,
                          make_animation: bool = True) -> list[Path]:
    """Render RD overlays, family snapshots, height-time plots, and optional GIF."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output_dir = Path(output_dir)
    overlay_dir = output_dir / "segmentation_overlays"
    family_dir = output_dir / "family_tracking"
    height_dir = output_dir / "height_time"
    for directory in (overlay_dir, family_dir, height_dir):
        directory.mkdir(parents=True, exist_ok=True)

    track = (segmentation.size_gate_output.birth_gate_output
             .association_output.tracking_output)
    frame_cache = track.frame_cache
    union = segmentation.final_pixel_segmentation_union
    geom = track.geometry
    paths: list[Path] = []

    for i, mask in enumerate(union):
        rd = np.asarray(frame_cache[i]["rd"], dtype=float)
        valid = np.asarray(frame_cache[i]["val"], dtype=bool)
        if rd.shape != mask.shape:
            raise ValueError(f"RD and segmentation shape mismatch at frame {i}")
        values = rd[valid & np.isfinite(rd)]
        limit = float(np.percentile(np.abs(values), 98)) if values.size else 1.0
        limit = max(limit, 1e-6)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.imshow(rd, origin="lower", cmap="gray", vmin=-limit, vmax=limit)
        colored = np.ma.masked_where(~mask, mask.astype(float))
        ax.imshow(colored, origin="lower", cmap="autumn", alpha=.55, vmin=0, vmax=1)
        ax.contour(mask.astype(float), levels=[.5], colors=["yellow"],
                   linewidths=.65, origin="lower") if mask.any() else None
        ax.set_title(f"COR2 RD + segmented candidates | frame {i:03d}")
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        fig.tight_layout()
        target = overlay_dir / f"frame_{i:03d}.png"
        fig.savefig(target)
        plt.close(fig)
        paths.append(target)

    for fid, mask_stack in sorted(segmentation.family_pixel_masks.items()):
        counts = np.sum(mask_stack, axis=(1, 2))
        if not np.any(counts):
            continue
        i = int(np.argmax(counts))
        rd = np.asarray(frame_cache[i]["rd"], dtype=float)
        valid = np.asarray(frame_cache[i]["val"], dtype=bool)
        values = rd[valid & np.isfinite(rd)]
        limit = max(float(np.percentile(np.abs(values), 98)), 1e-6) if values.size else 1.0
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.imshow(rd, origin="lower", cmap="gray", vmin=-limit, vmax=limit)
        pixels = mask_stack[i].astype(bool)
        ax.imshow(np.ma.masked_where(~pixels, pixels.astype(float)),
                  origin="lower", cmap="autumn", alpha=.6, vmin=0, vmax=1)
        ax.set_title(f"{fid} | peak mask frame {i:03d} | {int(counts[i])} pixels")
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        fig.tight_layout()
        target = family_dir / f"{_safe_name(fid)}.png"
        fig.savefig(target)
        plt.close(fig)
        paths.append(target)

        selected = frames.loc[frames["family_id"] == fid].copy()
        if selected.empty:
            continue
        selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True)
        selected.sort_values("timestamp", inplace=True)
        x = (selected["timestamp"] - selected["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 3600
        y = selected["front_rsun"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
        ax.plot(x, y, "o-", label="Maximum segmented-mask radius")
        if len(np.unique(x)) >= 3:
            fitted = np.polyfit(x, y, 1)
            ax.plot(x, np.polyval(fitted, x), "--", label="Linear fit (diagnostic)")
        ax.set_xlabel("Hours since first segmented observation")
        ax.set_ylabel("Front radius (projected solar radii)")
        ax.set_title(f"{fid} | mask-based front height")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        target = height_dir / f"{_safe_name(fid)}.png"
        fig.savefig(target)
        plt.close(fig)
        paths.append(target)

    if make_animation and len(paths) > 0:
        from PIL import Image
        overlay_files = sorted(overlay_dir.glob("frame_*.png"))
        if overlay_files:
            destination = output_dir / "cme_animation.gif"
            with Image.open(overlay_files[0]) as first:
                frames_rgb = []
                for file in overlay_files:
                    with Image.open(file) as img:
                        frames_rgb.append(img.convert("RGB").resize((480, 480)))
                frames_rgb[0].save(destination, save_all=True,
                                   append_images=frames_rgb[1:], duration=300, loop=0)
            paths.append(destination)
    return paths
