"""Stage 08: independent image-plane CME catalog from a verified Segmentation Run.

Reuses the canonical CME catalog builder without rendering optional plots.
No segmentation, recovery or gates are recomputed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.run_layout import verify_stage, initialize_run, complete_stage
from cme_pipeline.core.stage import BirthGateOutput, SizeGateOutput, SegmentationOutput
from cme_pipeline.standard_segmentation_parent import resolve_recovery_parent
from cme_pipeline.standard_runner import _revision, _context, _params, _stage_outputs
from cme_pipeline.stages.segmentation.cme_catalog import build_cme_catalog


def resolve_segmentation_parent(config, input_run):
    parent = Path(input_run).expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    manifest = json.loads((parent / "run_manifest.json").read_text())
    if manifest["stages"].get("segmentation") != "COMPLETED":
        raise ValueError("Input Run has no completed Segmentation stage")
    metadata = verify_stage(parent, "segmentation")
    if metadata.get("implementation") != "island_first":
        raise ValueError("Only saved Island-First Segmentation is supported")
    previous = json.loads((parent / "config_snapshot.json").read_text())
    if str(previous["experiment"]["date"]) != str(config.data["experiment"]["date"]):
        raise ValueError("Segmentation parent experiment date mismatch")
    if (previous.get("segmentation") or {}) != _params(config, "segmentation"):
        raise ValueError("Segmentation configuration changed; create a new Segmentation Run")
    provenance = manifest.get("input_provenance") or {}
    if provenance.get("input_stage") != "recovery" or not provenance.get("input_run"):
        raise ValueError("Segmentation Run lacks an explicit Recovery parent")
    recovery_root = Path(provenance["input_run"]).expanduser().resolve()
    rec_root, rec_meta, gates, gates_meta, assoc, family_summary, admitted, tracks, rec_union, rec_families = (
        resolve_recovery_parent(config, recovery_root)
    )
    def file_for(key):
        file = (parent / metadata["outputs"][key]["path"]).resolve()
        if not file.is_relative_to(parent) or not file.is_file():
            raise ValueError("Segmentation output is missing or outside its Run")
        return file
    report = json.loads(file_for("segmentation_metrics").read_text())
    if (Path(report["input_recovery_run"]).resolve() != rec_root
            or Path(report["input_gates_run"]).resolve() != gates
            or report.get("upstream_recomputed") is not False
            or int(report["recovery_patches"]) != int(rec_union.sum())):
        raise ValueError("Segmentation provenance inconsistent with Recovery")
    union = np.load(file_for("pixel_masks"), allow_pickle=False)
    with np.load(file_for("family_pixel_masks"), allow_pickle=False) as saved:
        family = {key.removeprefix("mask_"): saved[key] for key in saved.files
                  if key.startswith("mask_")}
        all_names = set(saved.files)
    if all_names != {"mask_" + fid for fid in admitted}:
        raise ValueError("Unexpected family segmentation keys")
    n = len(assoc.tracking_output.manifest_df)
    if union.shape != (n, 256, 256) or union.dtype != np.bool_:
        raise ValueError("Invalid global pixel mask dimensions or dtype")
    combined = np.zeros_like(union, dtype=bool)
    for fid, mask in family.items():
        if mask.shape != union.shape or mask.dtype != np.bool_:
            raise ValueError(f"Invalid Family pixel mask: {fid}")
        if np.any(mask & ~union):
            raise ValueError(f"Family pixel mask outside union: {fid}")
        combined |= mask
    if not np.array_equal(combined, union):
        raise ValueError("Family pixel masks do not reconstruct global mask")
    if (int(report["total_segmented_pixels"]) != int(union.sum())
            or int(report["n_families"]) != len(family)):
        raise ValueError("Segmentation summary does not match saved masks")
    framewise = pd.read_csv(file_for("framewise_pixel_summary"))
    family_stats = pd.read_csv(file_for("family_pixel_statistics"))
    if len(framewise) != n or len(family_stats) != len(admitted):
        raise ValueError("Segmentation summary table row count mismatch")
    if not np.array_equal(framewise["pair_index"].to_numpy(),
                          assoc.tracking_output.manifest_df["pair_index"].to_numpy()):
        raise ValueError("Framewise pair_index differs from upstream manifest")
    if not np.array_equal(framewise["segmented_pixels"].to_numpy(),
                          union.sum(axis=(1, 2))):
        raise ValueError("Framewise segmented pixel counts disagree with masks")
    if set(family_stats["family_id"]) != set(admitted):
        raise ValueError("Family pixel statistics contain incorrect Family IDs")
    counts = {fid: int(mask.sum()) for fid, mask in family.items()}
    for row in family_stats.itertuples(index=False):
        if int(row.total_segmented_pixels) != counts[row.family_id]:
            raise ValueError("Family pixel summary disagree with saved masks")

    gate_settings = _params(config, "gates")
    cfg = PipelineConfig()
    cfg.date_str = int(config.data["experiment"]["date"])
    cfg.birth_gate.max_birth_radius_px = float(gate_settings.get("birth_max_radius_px", 30.0))
    cfg.size_gate.min_detected_patches = int(
        (gate_settings.get("persistence") or {}).get("min_detected_patches", 10))
    birth_pass = sorted(family_summary.loc[
        family_summary["anchor_count"].astype(int) > 0, "family_id"].astype(str))
    birth = BirthGateOutput(association_output=assoc, birth_gated_family_ids=birth_pass,
                            max_birth_radius_px=cfg.birth_gate.max_birth_radius_px)
    size = SizeGateOutput(
        birth_gate_output=birth, admitted_family_ids=admitted, admitted_track_ids=tracks,
        family_summary_df=family_summary,
        size_gate_decisions_df=family_summary[["family_id", "status", "rejection_reason"]].copy(),
        min_detected_patches=cfg.size_gate.min_detected_patches,
        recovered_patch_masks=rec_union, family_recovered_patch_masks=rec_families,
        total_recovered_patches=int(rec_union.sum()))
    segmentation = SegmentationOutput(
        size_gate_output=size, final_pixel_segmentation_union=union,
        family_pixel_masks=family, framewise_pixel_summary_df=framewise,
        family_pixel_statistics_df=family_stats, total_segmented_pixels=int(union.sum()))
    return parent, metadata, recovery_root, gates, segmentation


def execute_from_segmentation(config, *, input_run, until, run_dir, visualize=None):
    if until != "catalog":
        raise ValueError("Segmentation input supports --until catalog only")
    if config.vis_mode("catalog", visualize) != "off":
        raise NotImplementedError("Catalog visualization integration is not yet supported")
    settings = _params(config, "catalog")
    if config.method("catalog") != "event_kinematics":
        raise ValueError("Only the canonical event_kinematics catalog is supported")
    allowed = {"method", "rsun_px", "solar_radius_km", "visualization"}
    if set(settings) - allowed:
        raise ValueError(f"Unsupported Catalog keys: {sorted(set(settings) - allowed)}")
    if "rsun_px" not in settings:
        raise ValueError("catalog.rsun_px is required: verify solar radius in pixels from FITS/WCS")
    rsun = float(settings["rsun_px"])
    solar_km = float(settings.get("solar_radius_km", 695700.0))
    if not math.isfinite(rsun) or rsun <= 0 or not math.isfinite(solar_km) or solar_km <= 0:
        raise ValueError("Invalid positive solar radius or solar_radius_km")
    parent, parent_meta, recovery_root, gates_root, segmentation = resolve_segmentation_parent(
        config, input_run)
    destination = Path(run_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    segmentation.size_gate_output.birth_gate_output.association_output.tracking_output.geometry.rsun_px = rsun
    # Compute before creating output so invalid input/time data fails without
    # leaving an incomplete child Run.
    events, frames = build_cme_catalog(segmentation, solar_radius_km=solar_km)
    kinematics_columns = [
        "event_id", "family_id", "plane_of_sky_speed_km_s",
        "speed_fit_r2", "speed_fit_frames", "speed_quality",
        "max_front_rsun",
    ]
    kinematics = events[kinematics_columns].copy()
    initialize_run(destination, config.data, _revision(), {
        "mode": "explicit_input_run", "input_run": str(parent),
        "input_stage": "segmentation",
        "recovery_ancestor": str(recovery_root), "gates_ancestor": str(gates_root),
        "segmentation_outputs": parent_meta["outputs"],
        "solar_radius_px": rsun, "solar_radius_km": solar_km,
    })
    ctx = _context(destination, "catalog")
    event_path = ctx.tables_dir / "cme_event_catalog.csv"
    frame_path = ctx.tables_dir / "cme_frame_measurements.csv"
    speed_path = ctx.tables_dir / "cme_kinematics.csv"
    events.to_csv(event_path, index=False)
    frames.to_csv(frame_path, index=False)
    kinematics.to_csv(speed_path, index=False)
    report_path = ctx.run_dir / "catalog_summary.json"
    report_path.write_text(json.dumps({
        "input_segmentation_run": str(parent),
        "method": "event_kinematics",
        "events": int(len(events)), "frame_measurements": int(len(frames)),
        "solar_radius_px": rsun, "solar_radius_km": solar_km,
        "upstream_recomputed": False,
        "kinematics_interpretation": "exploratory_linear_fit_plane_of_sky",
    }, indent=2) + "\n")
    outputs = _stage_outputs(destination, {
        "events": event_path, "height_time": frame_path,
        "kinematics": speed_path, "catalog_metrics": report_path,
    })
    complete_stage(destination, "catalog", {
        "parent_run": str(parent),
        "pixel_masks": parent_meta["outputs"]["pixel_masks"]["path"],
    }, outputs, "event_kinematics:canonical_cme_catalog", settings)
    return destination
