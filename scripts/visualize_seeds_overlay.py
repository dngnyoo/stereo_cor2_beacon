"""Visual-only SEEDS sectors over already-rendered CME segmentation PNGs.

No pipeline recomputation or input modification. SEEDS angular/height sector
is a geometric approximation, NOT a SEEDS binary pixel mask.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scripts.validate_seeds import DEFAULT_SEEDS

# Pixel layout fixed by _save_science_frame() in visualization.py:
# 7.2 x 7.0 inches * 120 dpi; subplot bounds .03,.97,.10,.88;
# equal-aspect 256x256 raster, centered horizontally.
EXPECTED_SIZE = (864, 840)
FILE_PATTERN = re.compile(r"^pair_(\d+)_(\d{8}T\d{6}Z)_families\.png$")


def axes_bounds(size):
    if tuple(size) != EXPECTED_SIZE:
        raise ValueError(f"Unknown segmentation PNG layout {size}; expected {EXPECTED_SIZE}")
    w, h = size
    side = h * (.88 - .10)
    return (w - side) / 2, h * (1 - .88), side


def png_xy(x, y, bounds):
    left, top, side = bounds
    return left + x * side / 256, top + (256 - y) * side / 256


def sector_points(bpa, width, r_inner, r_outer, sx, sy, bounds):
    if not (np.isfinite(bpa) and np.isfinite(width) and 0 < width <= 360):
        raise ValueError("Invalid SEEDS angular extent")
    if not (np.isfinite(r_outer) and r_outer > r_inner >= 0):
        raise ValueError("Invalid SEEDS radius")
    angles = np.deg2rad(bpa - width / 2 + np.linspace(0, width, max(24, int(width))))
    def arc(radius, theta):
        return [png_xy(sx - radius * np.sin(a), sy + radius * np.cos(a), bounds)
                for a in theta]
    return arc(r_outer, angles) + arc(r_inner, angles[::-1])

def sector_points_from_bounds(pa_start, pa_end, r_inner, r_outer, sx, sy, bounds):
    """Use SEEDS reported PASTART and PAEND, not BPA +/- width/2."""
    pa_start, pa_end = float(pa_start), float(pa_end)
    if not np.isfinite([pa_start, pa_end]).all():
        raise ValueError("Non-finite PA boundaries")
    span = (pa_end - pa_start) % 360.0
    if span <= 0:
        raise ValueError("Zero or ambiguous SEEDS angular sector")
    return sector_points((pa_start + span / 2) % 360, span,
                         r_inner, r_outer, sx, sy, bounds)


def select_nearest_observations(seeds, timestamp, max_delta):
    """At most one observation per parent event, always nearest in time."""
    candidates = seeds.copy()
    candidates["delta_sec"] = (candidates["time"] - timestamp).abs().dt.total_seconds()
    candidates = candidates[candidates["delta_sec"] <= max_delta]
    return (candidates.sort_values(["delta_sec", "parent_event_id"])
            .drop_duplicates("parent_event_id", keep="first"))


def load_seeds(path):
    df = pd.read_csv(path)
    detail_keys = {"DATE", "TIME", "BPA", "PASTART", "PAEND",
                   "HMHEIGHT", "parent_event_id"}
    if detail_keys.issubset(df.columns):
        starts = pd.to_numeric(df["PASTART"], errors="raise")
        ends = pd.to_numeric(df["PAEND"], errors="raise")
        widths = (ends - starts) % 360.0
        if (widths <= 0).any():
            raise ValueError("Invalid SEEDS PASTART/PAEND interval")
        df = pd.DataFrame({
            "parent_event_id": df["parent_event_id"],
            "seeds_timestamp": df["DATE"].astype(str) + " " + df["TIME"].astype(str),
            "is_evaluable": True,
            "seeds_bpa": df["BPA"],
            "seeds_width": widths,
            "seeds_pa_start": starts,
            "seeds_pa_end": ends,
            "seeds_height": df["HMHEIGHT"],
        })
    keys = {"parent_event_id", "seeds_timestamp", "is_evaluable", "seeds_bpa",
            "seeds_width", "seeds_height"}
    if not keys.issubset(df.columns):
        raise ValueError(f"SEEDS reference missing columns: {sorted(keys - set(df.columns))}")
    df = df[df["is_evaluable"].astype(str).str.lower().eq("true")].copy()
    df["time"] = pd.to_datetime(df["seeds_timestamp"], utc=True, errors="raise")
    for field in ("seeds_bpa", "seeds_width", "seeds_height"):
        df[field] = pd.to_numeric(df[field], errors="raise")
    return df


def overlay_frame(source, destination, seeds, *, max_delta=900.,
                  sx=126.618, sy=127.503, rsun_px=8.163,
                  inner_px=22., outer_limit_px=120.):
    name = FILE_PATTERN.fullmatch(source.name)
    if not name:
        raise ValueError(f"Unexpected PNG name: {source.name}")
    stamp = pd.to_datetime(name.group(2), utc=True, format="%Y%m%dT%H%M%SZ")
    active = select_nearest_observations(seeds, stamp, max_delta)
    with Image.open(source) as original:
        base = original.convert("RGBA")
    bounds = axes_bounds(base.size)
    layer = Image.new("RGBA", base.size)
    painter = ImageDraw.Draw(layer, "RGBA")
    count = 0
    for _, ev in active.iterrows():
        radius = min(float(ev["seeds_height"]) * rsun_px, outer_limit_px)
        if not np.isfinite(radius) or radius <= inner_px:
            continue
        bpa, width = float(ev["seeds_bpa"]), float(ev["seeds_width"])
        if {"seeds_pa_start", "seeds_pa_end"}.issubset(active.columns):
            vertices = sector_points_from_bounds(ev["seeds_pa_start"], ev["seeds_pa_end"],
                                                 inner_px, radius, sx, sy, bounds)
        else:
            vertices = sector_points(bpa, width, inner_px, radius, sx, sy, bounds)
        painter.polygon(vertices, fill=(224, 64, 251, 38))
        painter.line(vertices + [vertices[0]], fill=(240, 70, 255, 225), width=3)
        a = np.deg2rad(bpa)
        p0 = png_xy(sx - inner_px * np.sin(a), sy + inner_px * np.cos(a), bounds)
        p1 = png_xy(sx - radius * np.sin(a), sy + radius * np.cos(a), bounds)
        painter.line([p0, p1], fill=(250, 130, 255, 230), width=2)
        count += 1
    result = Image.alpha_composite(base, layer).convert("RGB")
    # Caption in the bottom margin (away from the data coordinate axes).
    ImageDraw.Draw(result).text(
        (16, 813), f"SEEDS approximate sectors | {count} active | visual review only",
        fill=(255, 158, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    result.close()
    layer.close()
    base.close()
    return count


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seeds-csv", type=Path, default=DEFAULT_SEEDS)
    ap.add_argument("--max-time-delta-sec", type=float, default=60.)
    ap.add_argument("--sx", type=float, default=126.618)
    ap.add_argument("--sy", type=float, default=127.503)
    ap.add_argument("--rsun-px", type=float, default=8.163)
    ap.add_argument("--inner-px", type=float, default=22.)
    ap.add_argument("--outer-limit-px", type=float, default=120.)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args(argv)
    if args.max_time_delta_sec < 0 or args.rsun_px <= 0 or args.inner_px < 0:
        ap.error("Invalid time or geometry parameter")
    root = args.run_dir.expanduser().resolve()
    images_dir = root / "visualizations/segmentation_overlays"
    sources = sorted(p for p in images_dir.glob("pair_*_families.png")
                     if FILE_PATTERN.fullmatch(p.name))
    if not sources:
        raise FileNotFoundError(f"No current segmentation PNGs in {images_dir}")
    seeds = load_seeds(args.seeds_csv.expanduser().resolve())
    output = root / "validation/seeds/frames"
    frames = []
    for source in sources:
        target = output / source.name.replace("_families.png", "_seeds.png")
        n = overlay_frame(
            source, target, seeds, max_delta=args.max_time_delta_sec,
            sx=args.sx, sy=args.sy, rsun_px=args.rsun_px,
            inner_px=args.inner_px, outer_limit_px=args.outer_limit_px,
        )
        print(f"{target.name}: {n} SEEDS sectors")
        frames.append(target)
    if args.gif:
        images = []
        for frame in frames:
            with Image.open(frame) as im:
                images.append(im.convert("RGB"))
        gif = root / "validation/seeds/seeds_overlay.gif"
        images[0].save(gif, save_all=True, append_images=images[1:],
                       duration=500, loop=0)
        for im in images:
            im.close()
        print(f"GIF: {gif}")
    print(f"Wrote {len(frames)} PNGs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
