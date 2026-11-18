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
            run / "01_preprocessing/data/preprocessed",
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
