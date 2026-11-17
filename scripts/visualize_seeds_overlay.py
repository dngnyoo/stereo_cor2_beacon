"""Overlay SEEDS frame geometry on EXISTING segmentation PNGs.

The SEEDS source describes PA boundaries and three height measurements; it
does not provide a pixel segmentation mask. No CME scoring or assignment.
Solar center and target plate scale come from the run WCS, while per-frame
apparent solar radius comes from the closest raw COR2-A FITS header.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scripts.validate_seeds import DEFAULT_SEEDS

EXPECTED_SIZE = (864, 840)
FILE_PATTERN = re.compile(r"^pair_(\d+)_(\d{8}T\d{6}Z)_families\.png$")
FITS_PATTERN = re.compile(r"^(\d{8})_(\d{6})_.*\.(?:fts|fits)$", re.IGNORECASE)
ARC_COLORS = {
    "seeds_lead": (255, 205, 60, 255),    # yellow: leading edge
    "seeds_height": (255, 85, 245, 255),  # magenta: measured height
    "seeds_follow": (70, 220, 255, 255),  # cyan: following edge
}


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
    """Actual angular boundaries; BPA need not bisect them."""
    pa_start, pa_end = float(pa_start), float(pa_end)
    if not np.isfinite([pa_start, pa_end]).all():
        raise ValueError("Non-finite PA boundaries")
    span = (pa_end - pa_start) % 360.0
    if span <= 0:
        raise ValueError("Zero or ambiguous SEEDS angular sector")
    return sector_points((pa_start + span / 2) % 360, span,
                         r_inner, r_outer, sx, sy, bounds)


def arc_points(pa_start, pa_end, radius, sx, sy, bounds):
    span = (float(pa_end) - float(pa_start)) % 360.0
    if span <= 0 or radius <= 0:
        raise ValueError("Invalid SEEDS arc")
    angles = np.deg2rad(float(pa_start) + np.linspace(0, span, max(24, int(span))))
    return [png_xy(sx - radius * np.sin(a), sy + radius * np.cos(a), bounds)
            for a in angles]


def select_nearest_observations(seeds, timestamp, max_delta):
    """One source measurement per SEEDS event; no temporal interpolation."""
    candidates = seeds.copy()
    candidates["delta_sec"] = (candidates["time"] - timestamp).abs().dt.total_seconds()
    candidates = candidates[candidates["delta_sec"] <= max_delta]
    return (candidates.sort_values(["delta_sec", "parent_event_id"])
            .drop_duplicates("parent_event_id", keep="first"))


def load_seeds(path):
    df = pd.read_csv(path)
    detail_keys = {"DATE", "TIME", "BPA", "PASTART", "PAEND", "HMHEIGHT",
                   "HMLEAD", "HMFOLLOW", "parent_event_id"}
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
            "seeds_lead": df["HMLEAD"],
            "seeds_follow": df["HMFOLLOW"],
        })
    keys = {"parent_event_id", "seeds_timestamp", "is_evaluable",
            "seeds_bpa", "seeds_width", "seeds_height"}
    if not keys.issubset(df.columns):
        raise ValueError(f"SEEDS reference missing columns: {sorted(keys - set(df.columns))}")
    df = df[df["is_evaluable"].astype(str).str.lower().eq("true")].copy()
    df["time"] = pd.to_datetime(df["seeds_timestamp"], utc=True, errors="raise")
    for field in ("seeds_bpa", "seeds_width", "seeds_height", "seeds_lead", "seeds_follow",
                  "seeds_pa_start", "seeds_pa_end"):
        if field in df:
            df[field] = pd.to_numeric(df[field], errors="raise")
    return df


def load_target_geometry(run_dir):
    """Read solar center and plate scale from THIS run's 256px target WCS."""
    from astropy.io import fits
    from astropy.wcs import WCS
    header_path = Path(run_dir) / "acquisition/preprocessed/target_wcs_header.txt"
    header = fits.Header.fromstring(header_path.read_text(), sep="\n")
    wcs = WCS(header)
    center = wcs.world_to_pixel_values(0., 0.)
    scales = np.asarray([scale.to_value("arcsec") for scale in wcs.proj_plane_pixel_scales()], dtype=float)
    if not np.isfinite([*center, *scales]).all() or min(scales) <= 0:
        raise ValueError("Invalid target WCS geometry")
    if not np.isclose(scales[0], scales[1], rtol=1e-4):
        raise ValueError("Non-isotropic target plate scale not supported")
    # WCS PA convention must correspond to current north-up renderer.
    dx0, dy0 = np.asarray(wcs.world_to_pixel_values(0., .1)) - center
    dx90, dy90 = np.asarray(wcs.world_to_pixel_values(-.1, 0.)) - center
    if not (dy0 > 0 and abs(dx0) < abs(dy0) * .01
            and dx90 < 0 and abs(dy90) < abs(dx90) * .01):
        raise ValueError("Target WCS is not solar north-up / east-left")
    return float(center[0]), float(center[1]), float(np.mean(scales))


def nearest_fits_radius(frame_timestamp, raw_fits_root, arcsec_per_px,
                        max_delta_sec=1200.):
    """Solar radius in target-image pixels from nearest source FITS RSUN.

    Do not silently fall back to historical 8.163 px/Rsun.
    """
    from astropy.io import fits
    stamp = pd.Timestamp(frame_timestamp)
    day = stamp.strftime("%Y%m%d")
    folder = Path(raw_fits_root) / day
    matches = []
    for file in folder.iterdir() if folder.is_dir() else ():
        match = FITS_PATTERN.fullmatch(file.name)
        if match is None or match.group(1) != day:
            continue
        t = pd.to_datetime(match.group(1) + match.group(2), format="%Y%m%d%H%M%S", utc=True)
        delta = abs((t - stamp).total_seconds())
        if delta <= max_delta_sec:
            matches.append((delta, str(file), file))
    if not matches:
        raise FileNotFoundError(
            f"No COR2 FITS file within {max_delta_sec:g}s of {stamp} in {folder}"
        )
    matches.sort()
    path = matches[0][2]
    header = fits.getheader(path, 0)
    if str(header.get("DETECTOR", "")).upper() != "COR2" or str(
            header.get("OBSRVTRY", "")).upper() != "STEREO_A":
        raise ValueError(f"Not a STEREO-A COR2 FITS: {path}")
    rsun_arcsec = float(header.get("RSUN", np.nan))
    if not np.isfinite(rsun_arcsec) or rsun_arcsec <= 0:
        raise ValueError(f"Missing or invalid RSUN in {path}")
    return rsun_arcsec / arcsec_per_px, path


def overlay_frame(source, destination, seeds, *, max_delta=60.,
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
        r_height = min(float(ev["seeds_height"]) * rsun_px, outer_limit_px)
        if not np.isfinite(r_height) or r_height <= inner_px:
            continue
        bpa, width = float(ev["seeds_bpa"]), float(ev["seeds_width"])
        has_bounds = {"seeds_pa_start", "seeds_pa_end"}.issubset(active.columns)
        if has_bounds:
            start, end = float(ev["seeds_pa_start"]), float(ev["seeds_pa_end"])
            vertices = sector_points_from_bounds(start, end, inner_px,
                                                  r_height, sx, sy, bounds)
        else:
            start, end = bpa - width / 2, bpa + width / 2
            vertices = sector_points(bpa, width, inner_px, r_height, sx, sy, bounds)
        painter.polygon(vertices, fill=(224, 64, 251, 30))
        painter.line(vertices + [vertices[0]], fill=(255, 85, 245, 180), width=2)
        # Three independent source heights, clipped to the plotted COR2 FOV.
        for key, color in ARC_COLORS.items():
            if key not in ev or pd.isna(ev[key]):
                continue
            r = float(ev[key]) * rsun_px
            if not np.isfinite(r) or r <= inner_px:
                continue
            painter.line(arc_points(start, end, min(r, outer_limit_px),
                                    sx, sy, bounds), fill=color, width=3)
        # BPA is a direction, not the midpoint of the angular interval.
        a = np.deg2rad(bpa)
        p0 = png_xy(sx - inner_px * np.sin(a), sy + inner_px * np.cos(a), bounds)
        p1 = png_xy(sx - r_height * np.sin(a), sy + r_height * np.cos(a), bounds)
        painter.line([p0, p1], fill=(255, 255, 255, 215), width=2)
        count += 1
    result = Image.alpha_composite(base, layer).convert("RGB")
    ImageDraw.Draw(result).text(
        (14, 812),
        f"SEEDS {count} | lead=yellow  height=magenta  follow=cyan  BPA=white | "
        f"Rsun={rsun_px:.3f}px | geometric reference only",
        fill=(245, 245, 245))
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
    ap.add_argument("--raw-fits-root", type=Path, default=Path("data/raw/cor2/beacon"))
    ap.add_argument("--max-time-delta-sec", type=float, default=60.)
    ap.add_argument("--max-fits-delta-sec", type=float, default=1200.)
    ap.add_argument("--inner-px", type=float, default=22.)
    ap.add_argument("--outer-limit-px", type=float, default=120.)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args(argv)
    if min(args.max_time_delta_sec, args.max_fits_delta_sec, args.inner_px) < 0:
        ap.error("Invalid time or radius parameter")
    root = args.run_dir.expanduser().resolve()
    images_dir = root / "visualizations/segmentation_overlays"
    sources = sorted(p for p in images_dir.glob("pair_*_families.png")
                     if FILE_PATTERN.fullmatch(p.name))
    if not sources:
        raise FileNotFoundError(f"No current segmentation PNGs in {images_dir}")
    seeds = load_seeds(args.seeds_csv.expanduser().resolve())
    sx, sy, scale_arcsec = load_target_geometry(root)
    # Preflight geometry for all source images before overwriting any output.
    radii = {}
    for source in sources:
        stamp = pd.to_datetime(FILE_PATTERN.fullmatch(source.name).group(2),
                               utc=True, format="%Y%m%dT%H%M%SZ")
        active = select_nearest_observations(seeds, stamp, args.max_time_delta_sec)
        if not active.empty:
            radii[source] = nearest_fits_radius(
                stamp, args.raw_fits_root, scale_arcsec,
                max_delta_sec=args.max_fits_delta_sec,
            )[0]
    output = root / "validation/seeds/frames"
    frames = []
    for source in sources:
        target = output / source.name.replace("_families.png", "_seeds.png")
        n = overlay_frame(
            source, target, seeds, max_delta=args.max_time_delta_sec,
            sx=sx, sy=sy, rsun_px=radii.get(source, np.nan),
            inner_px=args.inner_px, outer_limit_px=args.outer_limit_px,
        )
        print(f"{target.name}: {n} SEEDS sectors"
              + (f" | Rsun={radii[source]:.4f}px" if source in radii else ""))
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
