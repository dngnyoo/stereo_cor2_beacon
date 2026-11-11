"""COR2 2010-12-12 scientific diagnostic visualization.

Rendering conventions follow the existing research scripts
cme_pipeline/experiments/generate_study_visualizations.py and
cme_pipeline/experiments/run_full_day_corrected_visualization.py:
north-up, coolwarm, fixed RD [-600,+600] DN, invalid pixels black,
solar geometry, tracking identifiers and UTC/cadence annotation.
Only the segmentation layer and candidate-catalog plots are additions.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

VMIN_RD = -600.0
VMAX_RD = 600.0
R_BIRTH_GATE_PX = 30.0
GIF_DURATION_MS = 500

# Preserve the historical track palette where explicit IDs were assigned.
TRACK_COLORS = {
    3: "#00e5ff", 4: "#ffd600", 5: "#76ff03", 6: "#ff9100",
    7: "#ff4081", 8: "#f48fb1", 9: "#ce93d8", 10: "#80deea",
    11: "#a7ffeb",
}
TRACK_PALETTE = [
    "#e040fb", "#7c4dff", "#536dfe", "#448aff", "#18ffff",
    "#64ffda", "#b2ff59", "#eeff41", "#ffff00", "#ffd740", "#ff6e40",
]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def get_track_color(tid: int) -> str:
    return TRACK_COLORS.get(int(tid), TRACK_PALETTE[int(tid) % len(TRACK_PALETTE)])


def get_rd_cmap():
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("black")
    return cmap


def prepare_rd(rd, valid) -> np.ndarray:
    """Never display invalid/nonfinite RD pixels as observations."""
    rd = np.asarray(rd, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if rd.shape != valid.shape:
        raise ValueError("RD / validity shapes disagree")
    return np.where(valid & np.isfinite(rd), rd, np.nan)


def draw_rd_background(ax, rd, valid):
    """Shared fixed scientific color scale for every frame and family."""
    from matplotlib.colors import Normalize
    image = prepare_rd(rd, valid)
    return ax.imshow(
        image, cmap=get_rd_cmap(),
        norm=Normalize(vmin=VMIN_RD, vmax=VMAX_RD),
        origin="lower", extent=[0, image.shape[1], 0, image.shape[0]],
        interpolation="nearest", zorder=0,
    )


def draw_solar_references(ax, geom):
    """Geometry and PA orientation from the pipeline's own WCS context."""
    import matplotlib.patheffects as pe

    sx, sy = float(geom.sx), float(geom.sy)
    theta = np.linspace(0, 2 * np.pi, 300)
    for radius, color, linestyle, width, alpha in (
        (float(geom.occulter_r_min_px), "white", "--", 0.9, 0.75),
        (float(geom.fov_r_max_px), "white", "--", 0.9, 0.75),
        (R_BIRTH_GATE_PX, "#FFD700", ":", 1.2, 0.85),
    ):
        ax.plot(sx + radius * np.cos(theta), sy + radius * np.sin(theta),
                color=color, linestyle=linestyle,
                linewidth=width, alpha=alpha, zorder=7)
    ax.plot([sx - 3.5, sx + 3.5], [sy, sy], color="white",
            linewidth=.8, alpha=.7, zorder=8)
    ax.plot([sx, sx], [sy - 3.5, sy + 3.5], color="white",
            linewidth=.8, alpha=.7, zorder=8)
    angle = np.deg2rad(225.0)
    label_radius = R_BIRTH_GATE_PX + 4.5
    lx, ly = sx - label_radius * np.sin(angle), sy + label_radius * np.cos(angle)
    ax.text(lx, ly, "Birth Gate: 30 px", color="#FFD700", fontsize=7,
            fontweight="bold", ha="center", va="center", zorder=12,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])


def draw_mask_outline(ax, mask, color="#00FF66", linewidth=1.1):
    """Overlay segmented pixels without replacing the fixed-scale RD background."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    ax.contour(
        mask.astype(float), levels=[.5], origin="lower",
        extent=[0, mask.shape[1], 0, mask.shape[0]],
        colors=[color], linewidths=linewidth, zorder=10,
    )


def draw_track_annotations(ax, track, index: int):
    """Observed ownership and distinct predicted-only states, when available."""
    import matplotlib.patheffects as pe

    owners = getattr(track, "track_ownership_masks", None)
    states = getattr(track, "track_states_df", None)
    if owners is None or states is None or "state" not in states.columns:
        return
    owners = np.asarray(owners)
    if owners.ndim != 3 or index >= owners.shape[0]:
        return
    if not {"pair_index", "track_id"}.issubset(states.columns):
        return
    pair = int(track.manifest_df.iloc[index]["pair_index"])
    rows = states.loc[states["pair_index"] == pair]
    patch_y, patch_x = owners.shape[1:]
    full_y, full_x = track.geometry.pa_raster.shape
    scale_y, scale_x = full_y / patch_y, full_x / patch_x

    observed = rows.loc[rows["state"] == "OBSERVED"]
    for tid in observed["track_id"].dropna().astype(int).unique():
        own = owners[index] == tid
        if not own.any():
            continue
        yy, xx = np.nonzero(own)
        color = get_track_color(tid)
        # Rectangle patch outlines retain the original spatial semantics.
        from matplotlib.patches import Rectangle
        for y, x in zip(yy, xx):
            ax.add_patch(Rectangle(
                (x * scale_x, y * scale_y), scale_x, scale_y,
                facecolor="none", edgecolor=color, linewidth=.8,
                alpha=.95, zorder=8,
            ))
        cx, cy = (xx.mean() + .5) * scale_x, (yy.mean() + .5) * scale_y
        ax.text(cx, cy, f"T{tid} ({len(xx)}p)", color=color,
                fontsize=7.0, weight="bold", ha="center", va="center",
                zorder=12, path_effects=[pe.withStroke(linewidth=2.3, foreground="black")])

    predicted = rows.loc[rows["state"] == "PREDICTED_ONLY"]
    for tid in predicted["track_id"].dropna().astype(int).unique():
        # Only prior observed owners are valid location anchors.
        previous = None
        for prev in range(index - 1, -1, -1):
            pixels = owners[prev] == tid
            if pixels.any():
                previous = np.nonzero(pixels)
                break
        if previous is None:
            continue
        yy, xx = previous
        cx, cy = (xx.mean() + .5) * scale_x, (yy.mean() + .5) * scale_y
        color = get_track_color(tid)
        ax.scatter([cx], [cy], marker="x", s=45, color=color,
                   linewidths=1.3, zorder=11)
        ax.text(cx + 3, cy - 3, f"T{tid} [PRED]", fontsize=7, color=color,
                zorder=12, path_effects=[pe.withStroke(linewidth=2, foreground="black")])


def _frame_banner(track, i: int) -> tuple[str, str]:
    row = track.manifest_df.iloc[i]
    pair = int(row["pair_index"]) if "pair_index" in row else i
    timestamp = str(row.get("end_time", ""))[:19]
    rd_id = str(row.get("rd_pair_id", ""))
    interval = row.get("delta_t_min", None)
    cadence = f" | dt={float(interval):.1f} min" if interval is not None and pd.notna(interval) else ""
    gap = bool(row.get("is_large_gap", False))
    if "is_large_gap" in track.frame_cache[i]:
        gap = bool(track.frame_cache[i]["is_large_gap"])
    banner = f"Pair {pair:03d} ({rd_id}) | {timestamp} UTC{cadence}"
    if gap:
        return banner + " | CADENCE GAP", "#ffb300"
    return banner, "white"


def _style_image_axis(ax, geom, shape):
    ax.set_facecolor("black")
    draw_solar_references(ax, geom)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.set_aspect("equal")
    ax.axis("off")


def _save_science_frame(target, track, idx, mask, title, annotate_tracks=False):
    import matplotlib.pyplot as plt
    fc = track.frame_cache[idx]
    fig, ax = plt.subplots(figsize=(7.2, 7.0), dpi=120, facecolor="#0e0e0e")
    fig.subplots_adjust(left=.03, right=.97, bottom=.10, top=.88)
    image = draw_rd_background(ax, fc["rd"], fc["val"])
    draw_mask_outline(ax, mask)
    if annotate_tracks:
        draw_track_annotations(ax, track, idx)
    _style_image_axis(ax, track.geometry, np.asarray(mask).shape)
    banner, color = _frame_banner(track, idx)
    fig.suptitle(f"{banner}\n{title}", color=color, fontsize=9.5, weight="bold", y=.965)
    bar = fig.add_axes([.20, .06, .6, .018])
    cb = fig.colorbar(image, cax=bar, orientation="horizontal")
    cb.set_label("RD (DN) | Fixed [-600, +600]", color="white", fontsize=8)
    cb.ax.tick_params(colors="white", labelsize=7)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, facecolor=fig.get_facecolor())
    plt.close(fig)


def create_visualizations(segmentation, frames: pd.DataFrame, output_dir: Path,
                          make_animation: bool = True) -> list[Path]:
    """Generate consistent scientific RD overlays, family diagnostics and GIF."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    overlay_dir = output_dir / "segmentation_overlays"
    family_dir = output_dir / "family_tracking"
    height_dir = output_dir / "height_time"
    for directory in (overlay_dir, family_dir, height_dir):
        directory.mkdir(parents=True, exist_ok=True)

    track = (segmentation.size_gate_output.birth_gate_output
             .association_output.tracking_output)
    union = np.asarray(segmentation.final_pixel_segmentation_union, dtype=bool)
    if len(union) != len(track.frame_cache):
        raise ValueError("Frame-cache length does not match segmentation")
    paths: list[Path] = []
    overlay_paths: list[Path] = []

    for i, mask in enumerate(union):
        target = overlay_dir / f"frame_{i:03d}.png"
        _save_science_frame(target, track, i, mask,
                            "Union segmented candidates + observed tracks",
                            annotate_tracks=True)
        paths.append(target)
        overlay_paths.append(target)

    for fid, mask_stack in sorted(segmentation.family_pixel_masks.items()):
        counts = np.asarray(mask_stack, dtype=bool).sum(axis=(1, 2))
        if not np.any(counts):
            continue
        i = int(np.argmax(counts))
        target = family_dir / f"{_safe_name(fid)}.png"
        _save_science_frame(target, track, i, mask_stack[i],
                            f"{fid} | peak frame | {int(counts[i])} segmented pixels")
        paths.append(target)

        selected = frames.loc[frames["family_id"] == fid].copy()
        if selected.empty:
            continue
        selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True)
        selected.sort_values("timestamp", inplace=True)
        x = (selected["timestamp"] - selected["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 3600
        y = selected["front_rsun"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=120, facecolor="#121212")
        ax.set_facecolor("#1a1a1a")
        ax.plot(x, y, marker="o", color="#00FFCC", linewidth=1.8,
                label="Outermost segmented pixel")
        if len(np.unique(x)) >= 3:
            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(x, slope * x + intercept, "--", color="#FF00FF",
                    label="Exploratory linear fit")
        ax.set_xlabel("Hours since first segmented observation", color="white")
        ax.set_ylabel("Maximum radius (projected R_sun)", color="white")
        ax.set_title(f"{fid} | Segmented-mask radius vs time", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle=":", color="gray", alpha=.3)
        ax.legend(facecolor="#262626", labelcolor="white", fontsize=8)
        fig.tight_layout()
        target = height_dir / f"{_safe_name(fid)}.png"
        fig.savefig(target, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(target)

    if make_animation and overlay_paths:
        from PIL import Image
        destination = output_dir / "cme_animation.gif"
        frames_rgb = []
        for file in overlay_paths:
            with Image.open(file) as img:
                frames_rgb.append(img.convert("RGB"))
        frames_rgb[0].save(destination, save_all=True,
                           append_images=frames_rgb[1:],
                           duration=GIF_DURATION_MS, loop=0)
        for item in frames_rgb:
            item.close()
        paths.append(destination)
    return paths
