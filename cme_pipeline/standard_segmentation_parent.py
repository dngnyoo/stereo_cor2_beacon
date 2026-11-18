"""Stage 07: isolated Island-First Segmentation from a verified Recovery Run.

Saved Recovery masks remain authoritative; no upstream algorithm is rerun.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.run_layout import initialize_run, complete_stage, verify_stage
from cme_pipeline.core.stage import BirthGateOutput, SizeGateOutput
from cme_pipeline.standard_recovery_parent import resolve_gates_parent
from cme_pipeline.standard_runner import _revision, _context, _params, _stage_outputs
from cme_pipeline.stages.segmentation.v01_island_first_segmentation import IslandFirstSegmentationStage


def validate_saved_masks(union, family_masks, admitted):
    """Check the complete saved Union/Family mask contract, including overlaps."""
    if union.ndim != 3 or union.shape[1:] != (64, 64) or union.dtype != np.bool_:
        raise ValueError("Recovery union must be Boolean (N,64,64)")
    expected = {"rec_" + fid for fid in admitted}
    if set(family_masks) != expected:
        raise ValueError("Recovery family keys differ from admitted Gates families")
    combined = np.zeros_like(union, dtype=bool)
    for key, mask in family_masks.items():
        if mask.shape != union.shape or mask.dtype != np.bool_:
            raise ValueError(f"Invalid Recovery Family mask: {key}")
        combined |= mask
    if not np.array_equal(combined, union):
        raise ValueError("Recovery union is not the union of saved Family masks")
    return int(union.sum())


def resolve_recovery_parent(config, input_run):
    root = Path(input_run).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["stages"].get("recovery") != "COMPLETED":
        raise ValueError("Input Run has no completed Recovery stage")
    stage = verify_stage(root, "recovery")
    if stage["implementation"] != "signed_dynamic_pa:apply_after_gates":
        raise ValueError("Only saved Signed Dynamic PA Recovery is supported")
    old = json.loads((root / "config_snapshot.json").read_text())
    if str(old["experiment"]["date"]) != str(config.data["experiment"]["date"]):
        raise ValueError("Recovery input date differs from requested experiment")
    if (old.get("recovery") or {}) != _params(config, "recovery"):
        raise ValueError("Recovery configuration differs from parent; create another Recovery Run")
    origin = manifest.get("input_provenance") or {}
    if origin.get("input_stage") != "gates" or not origin.get("input_run"):
        raise ValueError("Recovery Run has no explicit Gates parent")
    gates_root = Path(origin["input_run"]).expanduser().resolve()
    gates, gates_meta, association, family_summary, admitted, tracks, track_root, det_root = (
        resolve_gates_parent(config, gates_root)
    )
    def output(key):
        p = (root / stage["outputs"][key]["path"]).resolve()
        if not p.is_relative_to(root):
            raise ValueError("Recovery artifact escapes parent Run")
        return p
    report = json.loads(output("recovery_metrics").read_text())
    if (Path(report["source_gates_run"]).resolve() != gates
            or Path(report["source_tracking_run"]).resolve() != track_root
            or Path(report["source_detection_run"]).resolve() != det_root
            or report.get("gate_decisions_recomputed") is not False
            or report.get("method") != "signed_dynamic_pa"
            or report.get("parameters") != _params(config, "recovery")
            or sorted(report["admitted_families"]) != sorted(admitted)):
        raise ValueError("Recovery provenance, parameters or admission differ from verified Gates")
    union = np.load(output("recovered_patch_masks"), allow_pickle=False)
    with np.load(output("family_recovered_patch_masks"), allow_pickle=False) as saved:
        family = {name: saved[name] for name in saved.files}
    count = validate_saved_masks(union, family, admitted)
    if count != int(report["recovered_patches_all_frames"]):
        raise ValueError("Recovery patch total differs from saved report")
    n = len(association.tracking_output.manifest_df)
    if union.shape[0] != n:
        raise ValueError("Recovery frame count differs from original Detection")
    return root, stage, gates, gates_meta, association, family_summary, admitted, tracks, union, family


def execute_from_recovery(config, *, input_run, until, run_dir, visualize=None):
    if until != "segmentation":
        raise ValueError("Recovery Run supports --until segmentation only")
    if config.vis_mode("segmentation", visualize) != "off":
        raise NotImplementedError("Segmentation visualization is not integrated")
    settings = _params(config, "segmentation")
    if config.method("segmentation") != "island_first":
        raise ValueError("Only island_first segmentation is supported")
    allowed = {"method", "min_island_area_px", "max_hole_fill_px",
               "occulter_r_min_px", "fov_r_max_px", "visualization"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown segmentation options: {sorted(set(settings) - allowed)}")
    cfg = PipelineConfig()
    cfg.date_str = int(config.data["experiment"]["date"])
    cfg.segmentation.method = "island_first"
    cfg.segmentation.min_island_area_px = int(settings.get("min_island_area_px", 16))
    cfg.segmentation.max_hole_fill_px = int(settings.get("max_hole_fill_px", 16))
    cfg.segmentation.occulter_r_min_px = float(settings.get("occulter_r_min_px", 22.0))
    cfg.segmentation.fov_r_max_px = float(settings.get("fov_r_max_px", 122.5))
    if (cfg.segmentation.min_island_area_px < 1 or cfg.segmentation.max_hole_fill_px < 0
            or not math.isfinite(cfg.segmentation.occulter_r_min_px)
            or not math.isfinite(cfg.segmentation.fov_r_max_px)
            or cfg.segmentation.occulter_r_min_px <= 0
            or cfg.segmentation.fov_r_max_px <= cfg.segmentation.occulter_r_min_px):
        raise ValueError("Invalid Island-First segmentation parameters")

    root, rec_meta, gates, gates_meta, assoc, family_summary, admitted, tracks, union, family = (
        resolve_recovery_parent(config, input_run)
    )
    output = Path(run_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    gate_settings = _params(config, "gates")
    cfg.birth_gate.max_birth_radius_px = float(gate_settings.get("birth_max_radius_px", 30))
    cfg.size_gate.min_detected_patches = int(
        (gate_settings.get("persistence") or {}).get("min_detected_patches", 10))
    birth_pass = sorted(family_summary.loc[
        family_summary["anchor_count"].astype(int) > 0, "family_id"].astype(str))
    birth = BirthGateOutput(
        association_output=assoc, birth_gated_family_ids=birth_pass,
        max_birth_radius_px=cfg.birth_gate.max_birth_radius_px
    )
    size = SizeGateOutput(
        birth_gate_output=birth, admitted_family_ids=admitted,
        admitted_track_ids=tracks, family_summary_df=family_summary,
        size_gate_decisions_df=family_summary[["family_id", "status", "rejection_reason"]].copy(),
        min_detected_patches=cfg.size_gate.min_detected_patches,
        recovered_patch_masks=union, family_recovered_patch_masks=family,
        total_recovered_patches=int(union.sum())
    )
    initialize_run(output, config.data, _revision(), {
        "mode": "explicit_input_run",
        "input_run": str(root),
        "input_stage": "recovery",
        "gates_ancestor": str(gates),
        "recovery_outputs": rec_meta["outputs"],
        "gate_outputs": gates_meta["outputs"],
    })
    ctx = _context(output, "segmentation")
    result = IslandFirstSegmentationStage().run(size, cfg, ctx)
    masks = ctx.arrays_dir / "final_pixel_segmentation_union.npy"
    family_file = ctx.arrays_dir / "family_pixel_masks.npz"
    framewise = ctx.tables_dir / "06_full_day_framewise_pixel_summary.csv"
    familystats = ctx.tables_dir / "07_full_day_family_pixel_statistics.csv"
    if result.final_pixel_segmentation_union.shape != (len(union), 256, 256):
        raise ValueError("Segmentation output unexpected pixel dimensions")
    metrics = ctx.run_dir / "segmentation_summary.json"
    metrics.write_text(json.dumps({
        "input_recovery_run": str(root),
        "input_gates_run": str(gates),
        "recovery_patches": int(union.sum()),
        "total_segmented_pixels": int(result.total_segmented_pixels),
        "n_families": len(result.family_pixel_masks),
        "family_overlap_policy": "preserve_independent_masks",
        "upstream_recomputed": False,
    }, indent=2) + "\n")
    outputs = _stage_outputs(output, {
        "pixel_masks": masks,
        "segmentation_metrics": metrics,
        "family_pixel_masks": family_file,
        "framewise_pixel_summary": framewise,
        "family_pixel_statistics": familystats,
    })
    complete_stage(output, "segmentation",
                   {"parent_run": str(root),
                    "recovered_patch_masks": rec_meta["outputs"]["recovered_patch_masks"]["path"]},
                   outputs, "island_first", settings)
    return output
