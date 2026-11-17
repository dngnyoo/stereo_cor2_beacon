"""Full-day P04 detection comparison: red patch outlines over identical RD backgrounds."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def compare_detection(
    baseline_dir: str | Path, experiment_dir: str | Path, *,
    date: str, output_dir: str | Path | None = None,
) -> Path:
    if len(date) != 8 or not date.isascii() or not date.isdigit():
        raise ValueError("date must be YYYYMMDD")
    base, exp = Path(baseline_dir).resolve(), Path(experiment_dir).resolve()
    bmeta = json.loads((base / "detection_metadata.json").read_text())
    emeta = json.loads((exp / "detection_metadata.json").read_text())
    same_detector = bmeta["detector"] == emeta["detector"]
    c3_d4 = {bmeta["detector"], emeta["detector"]} == {"c3", "d4"}
    if (not (same_detector or c3_d4) or
            bmeta["denoiser"] != emeta["denoiser"]):
        raise ValueError("Comparison requires same denoiser and same detector or C3/D4")
    # Older C3 outputs may record different or incomplete configuration metadata.
    # Verify numerical thresholds from the saved summaries instead.
    if Path(bmeta["input_dir"]).resolve() != Path(emeta["input_dir"]).resolve():
        raise ValueError("Both detectors must use exactly the same preprocessed directory")
    pre = Path(bmeta["input_dir"]).resolve()
    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    bs = pd.read_csv(base / "frame_detection_summary.csv")
    es = pd.read_csv(exp / "frame_detection_summary.csv")
    bm = np.load(base / "patch_state_p04.npy", mmap_mode="r")
    em = np.load(exp / "patch_state_p04.npy", mmap_mode="r")
    n = len(manifest)
    if not (rd.shape == valid.shape == (n, 256, 256)
            and bm.shape == em.shape == (n, 64, 64)
            and len(bs) == len(es) == n):
        raise ValueError("Input and detector array shape mismatch")
    for summary in (bs, es):
        if not np.array_equal(
            summary["pair_index"].astype(int), manifest["pair_index"].astype(int)
        ) or not np.array_equal(
            pd.to_datetime(summary["end_time"], utc=True).to_numpy(),
            pd.to_datetime(manifest["end_time"], utc=True).to_numpy()
        ):
            raise ValueError("Manifest and detection summary are not frame-aligned")
    if c3_d4:
        for col in ("T24_p16", "T24_p08", "T24_p04", "floor_p04"):
            if col not in bs.columns or col not in es.columns:
                raise ValueError(f"Cannot verify controlled C3/D4 comparison: missing {col}")
            va = pd.to_numeric(bs[col], errors="coerce").to_numpy(float)
            vb = pd.to_numeric(es[col], errors="coerce").to_numpy(float)
            consistent = np.isclose(va, vb, rtol=1e-8, atol=1e-10, equal_nan=True)
            if not bool(np.all(consistent)):
                first = int(np.flatnonzero(~consistent)[0])
                raise ValueError(
                    f"C3/D4 {col} differs at pair {int(manifest.iloc[first]['pair_index'])}: "
                    f"C3={va[first]:.8g}, D4={vb[first]:.8g}. "
                    "Check the actual detector settings; these are not controlled runs."
                )
        bs_status = bs["status"].astype(str).to_numpy()
        es_status = es["status"].astype(str).to_numpy()
        if not np.array_equal(bs_status, es_status):
            first = int(np.flatnonzero(bs_status != es_status)[0])
            raise ValueError(
                f"C3/D4 observation status differs at pair {int(manifest.iloc[first]['pair_index'])}: "
                f"C3={bs_status[first]}, D4={es_status[first]}; comparison is not controlled."
            )
    times = pd.to_datetime(manifest["end_time"], utc=True, errors="raise")
    indices = np.flatnonzero(times.dt.strftime("%Y%m%d").eq(date).to_numpy())
    if not indices.size:
        raise ValueError(f"No evaluation frames ending on {date}")
    out = Path(output_dir).resolve() if output_dir else exp / "comparison_full_day"
    frames = out / "frames"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing comparison: {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from cme_pipeline.stages.segmentation.visualization import get_rd_cmap

    frames.mkdir(parents=True, exist_ok=False)
    rows = []
    baseline_label = f"{bmeta['detector'].upper()} Baseline (T24 {bmeta.get('t24_mad_multiplier', 3):g}MAD)"
    experiment_label = (f"{emeta['detector'].upper()} Experiment "
                        f"(T24 {emeta.get('t24_mad_multiplier', 3):g}MAD, "
                        f"abs floor {emeta.get('absolute_p04_floor', 'off')})")

    def fmt(val):
        return "N/A" if pd.isna(val) else f"{float(val):.3f}"

    for i in indices:
        i = int(i)
        b, e = bs.iloc[i], es.iloc[i]
        bmask, emask = np.asarray(bm[i]) == 1, np.asarray(em[i]) == 1
        nbase, nexp = int(bmask.sum()), int(emask.sum())
        rows.append({
            "pair_index": int(manifest.iloc[i]["pair_index"]),
            "end_time": manifest.iloc[i]["end_time"],
            "status_base": b["status"],
            "status_experiment": e["status"],
            "n_active_base": nbase,
            "n_active_experiment": nexp,
            "removed": int(np.count_nonzero(bmask & ~emask)),
            "added": int(np.count_nonzero(emask & ~bmask)),
            "delta_experiment_minus_base": nexp - nbase,
            "T24_p04_base": b["T24_p04"],
            "T24_p04_experiment": e["T24_p04"],
            "floor_p04_base": b["floor_p04"],
            "effective_threshold_p04_base": b["effective_threshold_p04"],
            "T24_p04": e["T24_p04"],
            "floor_p04_experiment": e["floor_p04"],
            "effective_threshold_p04_experiment": e["effective_threshold_p04"],
        })
        raw = np.asarray(rd[i], dtype=float)
        usable = np.asarray(valid[i], dtype=bool) & np.isfinite(raw)
        image = np.ma.masked_where(~usable, raw)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.7), constrained_layout=True)
        for ax, mask, label, summary, count in (
            (axes[0], bmask, baseline_label, b, nbase),
            (axes[1], emask, experiment_label, e, nexp),
        ):
            im = ax.imshow(image, origin="lower", cmap=get_rd_cmap(),
                           vmin=-600, vmax=600, interpolation="nearest")
            yy, xx = np.where(mask)
            rectangles = [
                Rectangle((int(x) * 4 - .5, int(y) * 4 - .5), 4, 4)
                for y, x in zip(yy, xx)
            ]
            if rectangles:
                ax.add_collection(PatchCollection(
                    rectangles, match_original=False, facecolor="none",
                    edgecolor="red", linewidth=.65, alpha=.9
                ))
            ax.set_aspect("equal")
            ax.set_xlim(-.5, 255.5)
            ax.set_ylim(-.5, 255.5)
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
            ax.set_title(
                f"{label} | {summary['status']} | active={count}\n"
                f"T24={fmt(summary['T24_p04'])} "
                f"floor={fmt(summary['floor_p04'])} "
                f"effective={fmt(summary['effective_threshold_p04'])}",
                fontsize=10
            )
        fig.colorbar(im, ax=axes, label="RD (DN/s)", shrink=.78)
        pair = int(manifest.iloc[i]["pair_index"])
        fig.suptitle(f"Pair {pair} | {times.iloc[i].strftime('%Y-%m-%d %H:%M UTC')}")
        fig.savefig(frames / f"pair_{pair:03d}_{times.iloc[i].strftime('%Y%m%dT%H%M%SZ')}.png",
                    dpi=140)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison_summary.csv", index=False)
    print(f"Frames: {len(df)}; baseline active: {df.n_active_base.sum()}; "
          f"experiment active: {df.n_active_experiment.sum()}; "
          f"removed patches: {df.removed.sum()}; added patches: {df.added.sum()}")
    print(df.sort_values("delta_experiment_minus_base").head(10).to_string(index=False))
    return out
