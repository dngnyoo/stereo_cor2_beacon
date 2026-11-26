"""Draw CACTus liftoff-window angular sectors on existing segmentation PNGs.

The display radii are graphical guides, NOT measured CME heights.
All active reference events are shown, independently of matching.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scripts.visualize_seeds_overlay import (
    FILE_PATTERN, axes_bounds, load_target_geometry, png_xy, sector_points,
)
from scripts.validate_cactus import _prepare


def load_cactus(path):
    df = pd.read_csv(path, dtype={"catalog_event_id": str})
    if "event_type" not in df or not df.event_type.isin(["CME", "Flow"]).all():
        raise ValueError("Invalid CACTus event_type")
    df = _prepare(df.loc[df.event_type.eq("CME")], "catalog_event_id",
                  "onset_utc", "CACTus")
    if df.catalog_event_id.duplicated().any():
        raise ValueError("Duplicate CACTus IDs")
    if (pd.to_numeric(df.angular_width_deg) <= 0).any():
        raise ValueError("CACTus width must be positive")
    duration = pd.to_numeric(df.liftoff_duration_hours, errors="raise")
    if not np.isfinite(duration).all() or (duration < 0).any():
        raise ValueError("Invalid liftoff duration")
    df["liftoff_end"] = df.onset_utc + pd.to_timedelta(duration, unit="h")
    return df


def active_events(df, stamp):
    return df.loc[(df.onset_utc <= stamp) & (stamp <= df.liftoff_end)]


def image_time(path):
    match = FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected PNG filename: {path}")
    return pd.to_datetime(match.group(2), format="%Y%m%dT%H%M%SZ", utc=True)


def overlay_frame(source, destination, reference, *, sx, sy,
                  inner_px=22., outer_px=120.):
    if source.resolve() == destination.resolve():
        raise ValueError("Output must not replace source")
    if not all(map(math.isfinite, (sx, sy, inner_px, outer_px))) or not (
        0 <= inner_px < outer_px <= min(sx, sy, 256-sx, 256-sy)
    ):
        raise ValueError("Invalid display geometry")
    active = active_events(reference, image_time(source))
    with Image.open(source) as original:
        base = original.convert("RGBA")
    bounds = axes_bounds(base.size)
    layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(layer)
    for row in active.itertuples(index=False):
        vertices = sector_points(float(row.cpa_deg), float(row.angular_width_deg),
                                 inner_px, outer_px, sx, sy, bounds)
        draw.polygon(vertices, fill=(224, 64, 251, 22))
        draw.line(vertices + [vertices[0]], fill=(255, 85, 245, 190), width=2)
        angle = math.radians(float(row.cpa_deg))
        p0 = png_xy(sx-inner_px*math.sin(angle), sy+inner_px*math.cos(angle), bounds)
        p1 = png_xy(sx-outer_px*math.sin(angle), sy+outer_px*math.cos(angle), bounds)
        draw.line([p0, p1], fill=(255, 85, 245, 210), width=2)
        x, y = png_xy(sx-(outer_px-10)*math.sin(angle),
                      sy+(outer_px-10)*math.cos(angle), bounds)
        draw.text((x, y), row.catalog_event_id.rsplit("_", 1)[-1],
                  fill=(255, 220, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))
    result = Image.alpha_composite(base, layer).convert("RGB")
    painter = ImageDraw.Draw(result)
    painter.rectangle((0, 788, result.width, result.height), fill=(20, 20, 20))
    painter.text((14, 792), f"CACTus liftoff window | {len(active)} active CME(s) | magenta = angular extent",
                 fill=(255, 220, 255))
    painter.text((14, 810), "Display radii only; no measured height or pixel mask. All active references.",
                 fill=(245, 245, 245))
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    result.close()
    layer.close()
    base.close()
    return len(active)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--geometry-run-dir", required=True, type=Path,
                    help="Run containing acquisition/preprocessed/target_wcs_header.txt")
    ap.add_argument("--cactus-csv", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--inner-px", type=float, default=22.)
    ap.add_argument("--outer-px", type=float, default=120.)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args(argv)
    sources = sorted((p for p in args.images_dir.glob("pair_*_families.png")
                      if FILE_PATTERN.fullmatch(p.name)), key=lambda p: (image_time(p), p.name))
    if not sources:
        raise FileNotFoundError(f"No compatible segmentation PNGs in {args.images_dir}")
    ref = load_cactus(args.cactus_csv)
    sx, sy, _ = load_target_geometry(args.geometry_run_dir)
    for source in sources:
        with Image.open(source) as im:
            axes_bounds(im.size)
    if not all(map(math.isfinite, (sx, sy, args.inner_px, args.outer_px))) or not (
        0 <= args.inner_px < args.outer_px <= min(sx, sy, 256-sx, 256-sy)
    ):
        raise ValueError("Invalid display geometry")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames = []
    for source in sources:
        target = args.output_dir / "frames" / source.name.replace("_families.png", "_cactus.png")
        count = overlay_frame(source, target, ref, sx=sx, sy=sy,
                              inner_px=args.inner_px, outer_px=args.outer_px)
        frames.append(target)
        print(f"{target.name}: {count} active CACTus events")
    if args.gif:
        images = []
        try:
            for frame in frames:
                with Image.open(frame) as im:
                    images.append(im.convert("RGB"))
            images[0].save(args.output_dir / "cactus_overlay.gif", save_all=True,
                           append_images=images[1:], duration=500, loop=0)
        finally:
            for im in images:
                im.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
