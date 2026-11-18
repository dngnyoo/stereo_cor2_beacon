"""Stage 09: isolated post-hoc official SEEDS-only validation.

A completed Stage 08 Catalog Run provides immutable event/frame products.
Official GMU SEEDS frame measurements must be prepared independently by the
existing acquisition/parser scripts. Reference data never changes detection.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from cme_pipeline.core.run_layout import initialize_run, complete_stage, verify_stage, _hash_file
from cme_pipeline.standard_catalog_parent import resolve_segmentation_parent
from cme_pipeline.standard_runner import _revision, _context, _params, _stage_outputs
from scripts.validate_seeds import _read_seeds, _read_detection_frames, validate_seeds


def validation_parameters(config):
    settings = _params(config, "validation")
    if config.method("validation") != "seeds":
        raise ValueError("Stage 09 currently supports SEEDS only")
    allowed = {"method", "catalogs", "seeds_frames_csv", "max_time_delta_sec",
               "min_mean_iou", "min_common_frames", "visualization"}
    if set(settings) - allowed:
        raise ValueError(f"Unsupported SEEDS validation settings: {sorted(set(settings) - allowed)}")
    if settings.get("catalogs") not in (None, ["SEEDS"], ("SEEDS",)):
        raise ValueError("Validation catalogs must contain only SEEDS")
    source = settings.get("seeds_frames_csv")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            "validation.seeds_frames_csv must explicitly identify a parsed, "
            "date-specific official SEEDS frame CSV; no frozen reference fallback"
        )
    tolerance = float(settings.get("max_time_delta_sec", 900.0))
    min_iou = float(settings.get("min_mean_iou", 0.1))
    common = settings.get("min_common_frames", 1)
    if (not math.isfinite(tolerance) or tolerance < 0
            or not math.isfinite(min_iou) or not 0 < min_iou <= 1
            or isinstance(common, bool) or not isinstance(common, int) or common < 1):
        raise ValueError("Invalid SEEDS temporal/IoU/matching thresholds")
    date = str(config.data["experiment"]["date"])
    datetime.strptime(date, "%Y%m%d")
    reference = Path(source).expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(
            f"Official SEEDS frame reference missing: {reference}. "
            f"Use scripts.fetch_seeds_monthly and scripts.fetch_seeds_details "
            f"for {date}, then set validation.seeds_frames_csv explicitly."
        )
    return reference, tolerance, min_iou, common


def resolve_catalog_parent(config, input_run):
    parent = Path(input_run).expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    run_manifest = json.loads((parent / "run_manifest.json").read_text())
    if run_manifest["stages"].get("catalog") != "COMPLETED":
        raise ValueError("Input Run has no completed Stage 08 Catalog")
    meta = verify_stage(parent, "catalog")
    if meta.get("implementation") != "event_kinematics:canonical_cme_catalog":
        raise ValueError("Catalog method is incompatible with SEEDS validation")
    previous = json.loads((parent / "config_snapshot.json").read_text())
    if str(previous["experiment"]["date"]) != str(config.data["experiment"]["date"]):
        raise ValueError("Catalog input date differs from validation date")
    if (previous.get("catalog") or {}) != _params(config, "catalog"):
        raise ValueError("Catalog configuration differs from parent Run")
    ancestry = run_manifest.get("input_provenance") or {}
    if ancestry.get("input_stage") != "segmentation" or not ancestry.get("input_run"):
        raise ValueError("Catalog lacks an explicit Segmentation parent")
    seg_parent = Path(ancestry["input_run"]).expanduser().resolve()
    seg_root, seg_meta, recovery_root, gates_root, seg_output = resolve_segmentation_parent(
        config, seg_parent
    )

    def artifact(key):
        info = meta["outputs"][key]
        p = (parent / info["path"]).resolve()
        if not p.is_relative_to(parent) or not p.is_file():
            raise ValueError(f"Catalog output missing/outside Run: {key}")
        return p

    report = json.loads(artifact("catalog_metrics").read_text())
    if (Path(report["input_segmentation_run"]).resolve() != seg_root
            or report.get("method") != "event_kinematics"
            or report.get("upstream_recomputed") is not False):
        raise ValueError("Catalog provenance inconsistent with Segmentation")
    events = pd.read_csv(artifact("events"))
    frames_file = artifact("height_time")
    frames = _read_detection_frames(frames_file)
    if (len(events) != int(report["events"])
            or len(frames) != int(report["frame_measurements"])):
        raise ValueError("Catalog event/frame counts differ from saved metadata")
    if not set(frames["event_id"]).issubset(set(events["event_id"])):
        raise ValueError("Catalog frame IDs absent from event table")
    return parent, meta, seg_root, frames_file, events, frames


def execute_from_catalog(config, *, input_run, until, run_dir, visualize=None):
    if until != "validation":
        raise ValueError("Catalog input supports --until validation only")
    if config.vis_mode("validation", visualize) != "off":
        raise NotImplementedError("SEEDS validation visualization is not integrated")
    reference, tolerance, min_iou, common = validation_parameters(config)
    date = str(config.data["experiment"]["date"])
    seeds = _read_seeds(reference)
    # Official detail-derived event identifiers embed the event onset date.
    # Do not use 2010 frozen references or unrelated observing epochs.
    ids = seeds["parent_event_id"].astype(str)
    if not ids.str.startswith(f"SEEDS_COR2A_{date}T").all():
        raise ValueError("SEEDS frame reference contains event IDs for another date or instrument")
    parent, meta, seg_root, frames_file, events, frames = resolve_catalog_parent(
        config, input_run
    )
    output = Path(run_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    pairs, statuses, summary = validate_seeds(
        seeds, frames, tolerance_sec=tolerance, min_iou=min_iou,
        min_common_frames=common
    )
    initialize_run(output, config.data, _revision(), {
        "mode": "explicit_input_run", "input_run": str(parent),
        "input_stage": "catalog", "segmentation_ancestor": str(seg_root),
        "catalog_outputs": meta["outputs"],
        "external_reference": {
            "catalog": "SEEDS", "path": str(reference), "sha256": _hash_file(reference),
            "date": date,
        },
        "upstream_recomputed": False,
    })
    ctx = _context(output, "validation")
    pair_file = ctx.tables_dir / "event_pair_scores.csv"
    status_file = ctx.tables_dir / "event_status.csv"
    metrics_file = ctx.run_dir / "summary.json"
    pairs.to_csv(pair_file, index=False)
    statuses.to_csv(status_file, index=False)
    summary["reference_file"] = str(reference)
    summary["reference_sha256"] = _hash_file(reference)
    summary["catalog_frame_file"] = str(frames_file)
    summary["input_catalog_run"] = str(parent)
    summary["upstream_recomputed"] = False
    summary["validation_implementation"] = "existing_scripts.validate_seeds"
    # Legacy summary contains a 2010-era caution. Replace it with a correct
    # date-aware caveat without changing the matching/scoring algorithm.
    summary["limitations"] = [
        x for x in summary["limitations"]
        if "2010-12-12" not in x
    ]
    summary["limitations"].append(
        f"Results describe only the explicitly supplied {date} SEEDS observation window."
    )
    metrics_file.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    outputs = _stage_outputs(output, {
        "catalog_matches": pair_file,
        "evaluation_metrics": metrics_file,
        "event_status": status_file,
    })
    complete_stage(
        output, "validation",
        {"parent_run": str(parent),
         "height_time": meta["outputs"]["height_time"]["path"],
         "seeds_reference_sha256": _hash_file(reference)},
        outputs, "seeds:one_to_one_temporal_angular_iou",
        _params(config, "validation")
    )
    return output
