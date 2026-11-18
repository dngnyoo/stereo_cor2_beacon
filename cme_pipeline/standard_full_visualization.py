"""Read-only Stage 01–09 diagnostic visualizations from saved artifacts.

No scientific stage code runs during rendering. Each supported stage has its
own concrete plot(s); no empty placeholder images are generated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.run_layout import verify_stage
from cme_pipeline.core.standard_stages import STAGE_BY_NAME


def _artifact(run: Path, metadata: dict, key: str) -> Path:
    return run / metadata["outputs"][key]["path"]


def _plot(stage: str, run: Path, destination: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = verify_stage(run, stage)
    output = destination / f"{stage}_summary.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    if stage == "association":
        df = pd.read_csv(_artifact(run, meta, "family_membership"))
        if df.empty or not {"pair_index", "track_id"}.issubset(df):
            raise ValueError("Association membership cannot be visualized")
        ax.scatter(df["pair_index"], df["track_id"], s=8, alpha=.65)
        ax.set(xlabel="RD pair index", ylabel="Track ID",
               title="Association — observed Track / Family membership")
    elif stage == "gates":
        df = pd.read_csv(_artifact(run, meta, "gate_decisions"))
        if df.empty or not {"family_id", "status", "max_detected_patches"}.issubset(df):
            raise ValueError("Gate decision table cannot be visualized")
        df = df.sort_values("max_detected_patches", ascending=False)
        ax.bar(df["family_id"].astype(str), df["max_detected_patches"].astype(float),
               color=["tab:blue" if s == "ADMITTED" else "tab:gray" for s in df["status"]])
        ax.tick_params(axis="x", rotation=75)
        ax.set(ylabel="Maximum detected patches", title="Gates — admitted (blue) vs rejected (gray)")
    elif stage == "recovery":
        saved = np.load(_artifact(run, meta, "recovered_patch_masks"), mmap_mode="r")
        counts = saved.sum(axis=(1, 2))
        ax.plot(np.arange(len(counts)), counts)
        ax.set(xlabel="RD frame index", ylabel="Recovered patches (union)",
               title="Recovery — framewise recovered patch occupancy")
    elif stage == "segmentation":
        saved = np.load(_artifact(run, meta, "pixel_masks"), mmap_mode="r")
        counts = saved.sum(axis=(1, 2))
        ax.plot(np.arange(len(counts)), counts)
        ax.set(xlabel="RD frame index", ylabel="Segmented pixels (union)",
               title="Segmentation — framewise pixel occupancy")
    elif stage == "catalog":
        df = pd.read_csv(_artifact(run, meta, "height_time"))
        if df.empty or not {"family_id", "timestamp", "front_rsun"}.issubset(df):
            raise ValueError("Catalog height-time measurements cannot be visualized")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for family, part in df.groupby("family_id"):
            ax.plot(part["timestamp"], part["front_rsun"], ".-", ms=3, label=family)
        ax.legend(fontsize=7, ncol=3)
        fig.autofmt_xdate()
        ax.set(xlabel="Time (UTC)", ylabel="Leading front (Rsun)",
               title="CME Catalog — height–time tracks (no fitted-line claim)")
    elif stage == "validation":
        df = pd.read_csv(_artifact(run, meta, "catalog_matches"))
        selected = df[df["assigned_match"].astype(bool)]
        if selected.empty:
            ax.text(.5, .5, "No SEEDS matches under configured criteria",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.bar(selected["detected_event_id"].astype(str),
                   selected["mean_angular_iou"].astype(float))
            ax.tick_params(axis="x", rotation=40)
        ax.set(ylabel="Mean temporal angular IoU", title="SEEDS Validation — assigned event matches")
        ax.set_ylim(bottom=0)
    else:
        raise ValueError(f"No summary renderer for stage {stage}")
    fig.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fig.savefig(output, dpi=145)
    plt.close(fig)
    return [output]



def _render_mask_overlays(config, stage, run, destination, mode):
    """Render evaluation-date masks on original saved RD without recomputing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = run.parent / "01_02_preprocessing_detection"
    pre = first / "01_preprocessing/data/acquisition/preprocessed"
    if not pre.is_dir():
        pre = first / "01_preprocessing/data"
    manifest = pd.read_csv(pre / "01_rd_manifest.csv")
    rd = np.load(pre / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(pre / "north_up_pair_valid_masks.npy", mmap_mode="r")
    meta = verify_stage(run, stage)
    key = "recovered_patch_masks" if stage == "recovery" else "pixel_masks"
    masks = np.load(_artifact(run, meta, key), mmap_mode="r")
    if len(manifest) != len(rd) or valid.shape != rd.shape or len(masks) != len(rd):
        raise ValueError("Original RD and saved mask frames are not aligned")
    dates = pd.to_datetime(manifest["end_time"], utc=True).dt.strftime("%Y%m%d")
    idx = np.flatnonzero(dates.eq(str(config.data["experiment"]["date"])).to_numpy())
    if not len(idx):
        raise ValueError("No evaluation-date frames for mask overlays")
    if mode != "all":
        idx = idx[np.unique(np.linspace(0, len(idx)-1, min(6, len(idx)), dtype=int))]
    output = []
    for i in idx:
        i = int(i)
        mask = np.asarray(masks[i], bool)
        if stage == "recovery":
            if mask.shape != (64, 64):
                raise ValueError("Recovery mask must have shape (64,64)")
            mask = np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)
        if mask.shape != (256, 256):
            raise ValueError("Pixel mask must have shape (256,256)")
        image = np.ma.masked_where(~np.asarray(valid[i], bool), rd[i])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        for ax in axes:
            ax.imshow(image, origin="lower", cmap="gray", vmin=-600, vmax=600)
            ax.set(xlim=(-.5, 255.5), ylim=(-.5, 255.5),
                   xlabel="X (pixel)", ylabel="Y (pixel)", aspect="equal")
        overlay = np.ma.masked_where(~mask, mask.astype(float))
        axes[1].imshow(overlay, origin="lower", cmap="autumn",
                       alpha=.45, interpolation="nearest", vmin=0, vmax=1)
        axes[0].set_title("Original north-up RD")
        axes[1].set_title(f"{stage.capitalize()} mask overlay")
        fig.suptitle(str(manifest.iloc[i]["end_time"]) + " UTC")
        filename = destination / f"pair_{int(manifest.iloc[i]['pair_index']):03d}_{stage}.png"
        if filename.exists():
            raise FileExistsError(filename)
        fig.savefig(filename, dpi=130)
        plt.close(fig)
        output.append(filename)
    return output


def render_full_stage(config, stage: str, run: Path, top_run: Path,
                      visualize: str | None = None) -> list[Path]:
    mode = config.vis_mode(stage, visualize)
    if mode == "off":
        return []
    destination = top_run / "visualizations" / STAGE_BY_NAME[stage].folder
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Visualization already exists: {destination}")
    date = str(config.data["experiment"]["date"])
    if stage == "preprocessing":
        from cme_pipeline.rd_diagnostics import save_rd_diagnostics
        paths = save_rd_diagnostics(
            (run / "01_preprocessing/data/acquisition/preprocessed"
             if (run / "01_preprocessing/data/acquisition/preprocessed").is_dir()
             else run / "01_preprocessing/data"),
            date=date, all_frames=(mode == "all"),
            output_dir=destination
        )
    elif stage == "detection":
        from cme_pipeline.detection_diagnostics import save_detection_diagnostics
        metadata = verify_stage(run, "detection")
        detector = _artifact(run, metadata, "patch_activation").parent
        paths = save_detection_diagnostics(
            detector, date=date, all_frames=(mode == "all"),
            output_dir=destination
        )
        # P16's per-frame values and QC decisions are computed within
        # Detection, not supplied by configuration as observations.
        if "p16_qc_decisions" in metadata["outputs"]:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            metrics_item = metadata["outputs"].get("p16_spatial_metrics")
            if metrics_item is not None:
                metrics_path = run / metrics_item["path"]
            else:
                source = (config.data.get("input") or {}).get("spatial_metrics_csv")
                if not source:
                    raise ValueError("Enabled P16 QC lacks metrics provenance")
                metrics_path = Path(source).expanduser().resolve()
            metrics = pd.read_csv(metrics_path)
            choices = pd.read_csv(_artifact(run, metadata, "p16_qc_decisions"))
            merged = metrics.merge(choices[["pair_index", "qc_status"]],
                                   on="pair_index", validate="one_to_one")
            if merged.empty:
                raise ValueError("Missing P16 QC decision rows")
            cutoff = float((config.data.get("detection") or {}).get("qc", {})[
                "normalized_weighted_tile_variance_min"])
            fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
            ax.plot(merged["pair_index"], merged["p16_normalized_weighted_tile_variance"],
                    ".-", label="Observed normalized P16 variance")
            ax.axhline(cutoff, linestyle="--", label=f"Configured QC cutoff = {cutoff:g}")
            flagged = merged["qc_status"].astype(str).eq("SPATIAL_QC")
            ax.scatter(merged.loc[flagged, "pair_index"],
                       merged.loc[flagged, "p16_normalized_weighted_tile_variance"],
                       marker="x", label="New P16 QC rejection")
            ax.set(xlabel="RD pair index", ylabel="Normalized weighted tile variance",
                   title="Stage 02 P16 metric and QC (computed from saved D4 output)")
            ax.legend()
            qc_plot = destination / "p16_spatial_qc_timeline.png"
            fig.savefig(qc_plot, dpi=145)
            plt.close(fig)
            paths.append(qc_plot)
    elif stage in ("recovery", "segmentation"):
        destination.mkdir(parents=True, exist_ok=False)
        paths = _render_mask_overlays(config, stage, run, destination, mode)
        paths.extend(_plot(stage, run, destination))
    elif stage == "tracking":
        from cme_pipeline.tracking_diagnostics import save_tracking_diagnostics
        data_root = run / "03_tracking/data"
        save_tracking_diagnostics(
            data_root, date=date, output_dir=destination
        )
        paths = list(destination.glob("*.png"))
    else:
        paths = _plot(stage, run, destination)
    if not paths:
        raise RuntimeError(f"Requested visualization for {stage} returned no PNG")
    record = {
        "stage": stage, "mode": mode,
        "read_only_saved_stage": True, "count": len(paths),
        "files": [str(p.relative_to(top_run)) for p in paths],
    }
    (destination / "visualization_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return paths
