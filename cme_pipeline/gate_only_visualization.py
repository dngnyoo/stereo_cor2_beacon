"""Observed-only, per-Family-color visualization for two saved A5b Gate runs.

This module ONLY reads saved A5b ownership, gate decisions and RD.
It does not run Recovery, Segmentation, or recompute any Gate.
"""
from __future__ import annotations

import colorsys
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _palette(index: int) -> tuple[float, float, float]:
    """Deterministic hues, spaced across the wheel for adjacent Family IDs."""
    return colorsys.hsv_to_rgb((index * 0.618033988749895) % 1.0, 0.79, 0.98)


def _load_gate_run(folder: Path):
    info = json.loads((folder / "gates_only_summary.json").read_text())
    track_dir = Path(info["source_tracking_run"]).resolve()
    own = np.load(track_dir / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    table = pd.read_csv(folder / "family_gate_decisions.csv")
    if table["track_id"].duplicated().any():
        raise ValueError(f"Duplicate Track IDs in {folder}")
    passed = set(table.loc[table["status"].eq("ADMITTED"), "track_id"].astype(int))
    return info, own, passed, table


def _match_ids(base: np.ndarray, exp: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    """Assign stable color indices; use overlap across all frames to match tracks."""
    a_ids = [int(v) for v in np.unique(base) if v > 0]
    b_ids = [int(v) for v in np.unique(exp) if v > 0]
    a_idx = {v: k for k, v in enumerate(a_ids)}
    b_idx = {v: k for k, v in enumerate(b_ids)}
    overlap = np.zeros((len(a_ids), len(b_ids)), dtype=np.int64)
    for k in range(len(base)):
        a, b = np.asarray(base[k]), np.asarray(exp[k])
        shared = (a > 0) & (b > 0)
        if shared.any():
            pairs, counts = np.unique(np.stack([a[shared], b[shared]], axis=1),
                                      axis=0, return_counts=True)
            for (tid_a, tid_b), n in zip(pairs, counts):
                overlap[a_idx[int(tid_a)], b_idx[int(tid_b)]] += int(n)
    base_colors = {v: i for i, v in enumerate(a_ids)}
    exp_colors = {}
    if overlap.size:
        rr, cc = linear_sum_assignment(-overlap)
        for i, j in zip(rr, cc):
            if overlap[i, j] > 0:
                exp_colors[b_ids[j]] = base_colors[a_ids[i]]
    next_idx = len(base_colors)
    for tid in b_ids:
        if tid not in exp_colors:
            exp_colors[tid] = next_idx
            next_idx += 1
    return base_colors, exp_colors


def render_gate_only_comparison(
    baseline: str | Path, experiment: str | Path, *,
    output_dir: str | Path,
) -> Path:
    """Create one side-by-side PNG per requested-date frame plus a simple gallery."""
    baseline, experiment, output = (Path(p).resolve() for p in
                                    (baseline, experiment, output_dir))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite visualization: {output}")
    ai, aown, apass, atbl = _load_gate_run(baseline)
    bi, bown, bpass, btbl = _load_gate_run(experiment)
    if (ai["source_detection_dir"] != bi["source_detection_dir"] or
            ai["date"] != bi["date"] or aown.shape != bown.shape):
        raise ValueError("Gate comparison requires same date, detection and frame shape")
    detection_dir = Path(ai["source_detection_dir"]).resolve()
    detector_meta = json.loads((detection_dir / "detection_metadata.json").read_text())
    rd_dir = Path(detector_meta["input_dir"]).resolve()
    manifest = pd.read_csv(rd_dir / "01_rd_manifest.csv")
    rd = np.load(rd_dir / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(rd_dir / "north_up_pair_valid_masks.npy", mmap_mode="r")
    if (len(manifest) != len(aown) or rd.shape != valid.shape or
            rd.shape != (len(aown), 256, 256)):
        raise ValueError("RD and A5b ownership arrays are not aligned")
    if "pair_index" not in manifest or "end_time" not in manifest:
        raise ValueError("RD manifest lacks pair_index/end_time")
    for folder in (baseline, experiment):
        run = Path(json.loads((folder / "gates_only_summary.json").read_text())
                   ["source_tracking_run"]).resolve()
        saved = pd.read_csv(run / "tables" / "live_observation_manifest.csv")
        if not np.array_equal(manifest["pair_index"].to_numpy(),
                              saved["pair_index"].to_numpy()):
            raise ValueError("Frame order differs between tracking and RD")

    dates = pd.to_datetime(manifest["end_time"], utc=True)
    indices = np.flatnonzero(dates.dt.strftime("%Y%m%d").eq(str(ai["date"])).to_numpy())
    if not len(indices):
        raise ValueError("No requested-date observations")
    acmap, bcmap = _match_ids(aown, bown)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from cme_pipeline.stages.segmentation.visualization import get_rd_cmap

    frames = output / "frames"
    frames.mkdir(parents=True)
    color_rows = []
    for name, mapping, passed in (("baseline", acmap, apass),
                                  ("experiment", bcmap, bpass)):
        for tid, color_index in sorted(mapping.items()):
            rgb = _palette(color_index)
            color_rows.append({
                "run": name, "track_id": tid, "family_id": f"FAM_T{tid:02d}",
                "color_group": color_index, "color_hex": "#%02x%02x%02x" %
                tuple(round(c * 255) for c in rgb),
                "admitted": tid in passed,
            })
    pd.DataFrame(color_rows).to_csv(output / "family_color_key.csv", index=False)

    report_rows = []
    gallery_items = []
    for idx in indices:
        idx = int(idx)
        pair = int(manifest.iloc[idx]["pair_index"])
        image = np.asarray(rd[idx], dtype=float)
        good = np.asarray(valid[idx], dtype=bool) & np.isfinite(image)
        masked = np.ma.masked_where(~good, image)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        figures = (("Baseline", axes[0], aown, apass, acmap, ai),
                   ("Experiment", axes[1], bown, bpass, bcmap, bi))
        row = {"pair_index": pair, "end_time": str(manifest.iloc[idx]["end_time"])}
        for label, ax, owner_stack, passed, palette_map, info in figures:
            own = np.asarray(owner_stack[idx])
            selected = np.isin(own, list(passed)) & (own > 0)
            rejected = (own > 0) & ~selected
            ax.imshow(masked, origin="lower", cmap=get_rd_cmap(),
                      vmin=-600, vmax=600, interpolation="nearest")
            rgba = np.zeros((64, 64, 4), dtype=float)
            rgba[rejected, :3] = (0.65, 0.65, 0.65)
            rgba[rejected, 3] = 0.18
            active_ids = [int(v) for v in np.unique(own[selected])]
            sizes = {}
            for tid in active_ids:
                current = (own == tid)
                rgb = _palette(palette_map[tid])
                rgba[current, :3] = rgb
                rgba[current, 3] = 0.72
                sizes[tid] = int(current.sum())
            ax.imshow(rgba, origin="lower", interpolation="nearest",
                      extent=(-0.5, 255.5, -0.5, 255.5))
            # Annotate the largest 12 admitted Families in this frame.
            for tid in sorted(active_ids, key=lambda t: sizes[t], reverse=True)[:12]:
                rr, cc = np.where(own == tid)
                if len(rr) < 2:
                    continue
                rgb = _palette(palette_map[tid])
                ax.text(float(np.mean(cc)) * 4 + 1.5, float(np.mean(rr)) * 4 + 1.5,
                        f"F{tid}", fontsize=8, fontweight="bold", color="white",
                        ha="center", va="center",
                        bbox=dict(facecolor=rgb, alpha=0.92,
                                  edgecolor="black", boxstyle="round,pad=0.15"))
            handles = [Patch(facecolor=_palette(palette_map[tid]),
                             label=f"F{tid} ({sizes[tid]} patches)")
                       for tid in sorted(active_ids, key=lambda t: sizes[t], reverse=True)[:9]]
            if handles:
                ax.legend(handles=handles, loc="lower left", fontsize=7,
                          framealpha=0.83, title="Admitted Families",
                          title_fontsize=8, ncol=2)
            ax.set_xlim(-0.5, 255.5)
            ax.set_ylim(-0.5, 255.5)
            ax.set_aspect("equal")
            ax.set_title(
                f"{label}: {info['admitted_families']} admitted Families\n"
                f"Size >= {info['min_detected_patches']} | "
                f"Persistence >= {info['min_observed_pairs']}\n"
                f"Observed admitted {int(selected.sum())} patches | "
                f"Rejected {int(rejected.sum())} patches",
                fontsize=10
            )
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
            key = label.lower()
            row[f"{key}_admitted_patches"] = int(selected.sum())
            row[f"{key}_rejected_patches"] = int(rejected.sum())
            row[f"{key}_families_in_frame"] = len(active_ids)
        fig.suptitle(
            f"A5b observed-only Family Gates | Pair {pair} | "
            f"{dates.iloc[idx].strftime('%Y-%m-%d %H:%M UTC')}\n"
            "One color per admitted Family; gray = rejected ownership; no Recovery",
            fontsize=12
        )
        filename = f"pair_{pair:03d}.png"
        fig.savefig(frames / filename, dpi=135)
        plt.close(fig)
        report_rows.append(row)
        gallery_items.append((pair, filename))
    pd.DataFrame(report_rows).to_csv(output / "visualization_frame_summary.csv", index=False)
    cards = "\n".join(
        f'<a href="frames/{html.escape(file)}"><img loading="lazy" '
        f'src="frames/{html.escape(file)}" alt="Pair {pair}"><span>Pair {pair}</span></a>'
        for pair, file in gallery_items
    )
    (output / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>A5b Family Gate Comparison</title>'
        '<style>body{font:15px system-ui;background:#16191e;color:#eee;margin:24px}'
        'main{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px}'
        'a{color:#ddd;text-decoration:none}img{width:100%;border-radius:6px}'
        'span{display:block;padding:6px}</style></head><body>'
        '<h1>Observed-only Family Gate comparison</h1>'
        '<p>Colors identify admitted Families. Gray indicates rejected tracked patches. '
        'No Recovery or Segmentation. Color groups are aligned across runs by maximum overlap.</p>'
        f'<main>{cards}</main></body></html>', encoding="utf-8"
    )
    print(f"Rendered {len(indices)} observed-only frame comparisons: {output}")
    print(f"Gallery: {output / 'index.html'}")
    return output
