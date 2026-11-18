"""Stage 06: Recovery from an explicit standard Gates Run.

Only the saved Gate decisions determine admitted families. No Detection,
Tracking, Association, Birth Gate or Size Gate computation is repeated.
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
from cme_pipeline.standard_association_parent import resolve_association_parent
from cme_pipeline.standard_runner import _revision, _context, _params, _stage_outputs
from cme_pipeline.stages.recovery.v02_apply_after_gates import ApplyRecoveryAfterGatesStage


def validate_saved_gate_decisions(summary: pd.DataFrame, saved_ids: dict,
                                  association, *, config) -> tuple[list[str], list[int]]:
    """Check saved pure Gate decision integrity, never recompute the Gates."""
    needed = {"family_id", "status", "anchor_count", "max_detected_patches"}
    if summary.empty or not needed.issubset(summary.columns):
        raise ValueError("Missing required saved Gate summary columns")
    if summary.family_id.isna().any() or summary.family_id.duplicated().any():
        raise ValueError("Missing or duplicate Gate Family IDs")
    if not summary.status.isin(("ADMITTED", "REJECTED")).all():
        raise ValueError("Invalid saved Gate statuses")
    expected = set(association.family_members)
    if set(summary.family_id.astype(str)) != expected:
        raise ValueError("Gate Family set differs from saved Association")
    admitted = sorted(summary.loc[summary.status == "ADMITTED", "family_id"].astype(str))
    ids = sorted(str(x) for x in saved_ids.get("family_ids", []))
    if admitted != ids:
        raise ValueError("Saved admitted Family IDs disagree with Gate summary")
    tracks = sorted({tid for fid in admitted for tid in association.family_members[fid]})
    if sorted(saved_ids.get("track_ids", [])) != tracks:
        raise ValueError("Saved admitted track IDs disagree with Gate summary")
    return admitted, tracks


def resolve_gates_parent(config, input_run):
    parent = Path(input_run).expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    run_manifest = json.loads((parent / "run_manifest.json").read_text())
    if run_manifest["stages"].get("gates") != "COMPLETED":
        raise ValueError("Input Run has no completed Gates stage")
    gate_stage = verify_stage(parent, "gates")
    if gate_stage["implementation"] != "birth+consecutive_size_qualified":
        raise ValueError("Only existing pure birth/consecutive-size Gates supported")
    old = json.loads((parent / "config_snapshot.json").read_text())
    if str(old["experiment"]["date"]) != str(config.data["experiment"]["date"]):
        raise ValueError("Gate parent date mismatch")
    if (old.get("gates") or {}) != _params(config, "gates"):
        raise ValueError("Gate parameters changed; create a new Gates Run first")

    provenance = run_manifest.get("input_provenance") or {}
    if provenance.get("input_stage") != "association" or not provenance.get("input_run"):
        raise ValueError(
            "Initial Recovery adapter requires a Gates-only Run with explicit "
            "Association parent; other Gates lineage types are not yet supported"
        )
    association_root = Path(provenance["input_run"]).expanduser().resolve()
    assoc_parent, assoc_stage, tracking_root, tracking_stage, detector_root, assoc = (
        resolve_association_parent(config, association_root)
    )
    if assoc_parent != association_root:
        raise ValueError("Association provenance mismatch")
    if set(assoc.family_members) != set(assoc.track_to_family.values()):
        raise ValueError("Association Family mapping inconsistent")

    def gate_path(key):
        file = (parent / gate_stage["outputs"][key]["path"]).resolve()
        if not file.is_relative_to(parent):
            raise ValueError("Gate output escapes parent")
        return file

    summary = pd.read_csv(gate_path("gate_decisions"))
    saved_ids = json.loads(gate_path("admitted_family_ids").read_text())
    admitted, tracks = validate_saved_gate_decisions(summary, saved_ids, assoc, config=config)
    qualified = pd.read_csv(gate_path("qualified_pair_ranges"))
    if set(qualified.family_id.astype(str)) != set(summary.family_id.astype(str)):
        raise ValueError("Qualified range Family set differs from Gate summary")
    if "size_qualified_pairs" not in summary or "size_qualified_pairs" not in qualified:
        raise ValueError("Saved qualified pairs missing")
    compare = summary.set_index("family_id")["size_qualified_pairs"].sort_index()
    other = qualified.set_index("family_id")["size_qualified_pairs"].sort_index()
    if not compare.equals(other):
        raise ValueError("Qualified pairs disagree with Gate summary")
    return parent, gate_stage, assoc, summary, admitted, tracks, tracking_root, detector_root


def execute_from_gates(config, *, input_run, until, run_dir, visualize=None):
    if until != "recovery":
        raise ValueError("Gates input supports --until recovery only")
    if config.vis_mode("recovery", visualize) != "off":
        raise NotImplementedError("Recovery visualization is not integrated")
    settings = _params(config, "recovery")
    if config.method("recovery") != "signed_dynamic_pa":
        raise ValueError("Only signed_dynamic_pa Recovery is supported")
    permitted = {"method", "inner_radius_px", "radial_window_px", "pa_margin_deg",
                 "gaussian_sigma", "visualization"}
    unknown = set(settings) - permitted
    if unknown:
        raise ValueError(f"Unsupported Recovery parameters: {sorted(unknown)}")
    if "inner_radius_px" not in settings:
        raise ValueError("recovery.inner_radius_px is mandatory; verify the physical occulter radius")
    for field in ("inner_radius_px", "radial_window_px", "pa_margin_deg"):
        val = float(settings.get(field, {"radial_window_px": 13.35, "pa_margin_deg": 5.0}.get(field)))
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"Invalid recovery.{field}: {val}")
    if float(settings.get("gaussian_sigma", 1.0)) != 1.0:
        raise ValueError("Only Gaussian sigma 1.0 loader is currently supported")

    parent, stage, assoc, saved_summary, admitted, tracks, tracking_root, detector_root = (
        resolve_gates_parent(config, input_run)
    )
    dest = Path(run_dir).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(dest)
    cfg = PipelineConfig()
    cfg.date_str = int(config.data["experiment"]["date"])
    cfg.recovery.method = "signed_dynamic_pa"
    cfg.recovery.radial_window_px = float(settings.get("radial_window_px", 13.35))
    cfg.recovery.pa_margin_deg = float(settings.get("pa_margin_deg", 5.0))
    cfg.recovery.gaussian_sigma = 1.0
    cfg.birth_gate.max_birth_radius_px = float(
        _params(config, "gates").get("birth_max_radius_px", 30)
    )
    cfg.size_gate.min_detected_patches = int(
        (_params(config, "gates").get("persistence") or {}).get("min_detected_patches", 10)
    )
    # This geometry field is explicitly unset by load_live_tracking_input.
    assoc.tracking_output.geometry.occulter_inner_radius_px = float(settings["inner_radius_px"])
    birth_pass = sorted(
        saved_summary.loc[saved_summary.anchor_count.astype(int) > 0, "family_id"].astype(str)
    )
    birth = BirthGateOutput(
        association_output=assoc,
        birth_gated_family_ids=birth_pass,
        max_birth_radius_px=cfg.birth_gate.max_birth_radius_px,
    )
    size = SizeGateOutput(
        birth_gate_output=birth, admitted_family_ids=admitted,
        admitted_track_ids=tracks, family_summary_df=saved_summary,
        size_gate_decisions_df=saved_summary[
            ["family_id", "status", "rejection_reason"]
        ].copy(),
        min_detected_patches=cfg.size_gate.min_detected_patches,
    )
    initialize_run(dest, config.data, _revision(), {
        "mode": "explicit_input_run", "input_run": str(parent),
        "input_stage": "gates",
        "tracking_ancestor": str(tracking_root),
        "detection_ancestor": str(detector_root),
        "gate_outputs": stage["outputs"],
        "gate_decisions_recomputed": False,
    })
    ctx = _context(dest, "recovery")
    result = ApplyRecoveryAfterGatesStage().run(size, cfg, ctx)
    mask = ctx.arrays_dir / "recovered_patch_masks_pipeline_b_full_day.npy"
    family_file = ctx.arrays_dir / "family_recovered_patch_masks.npz"
    summary_path = ctx.run_dir / "recovery_summary.json"
    summary_path.write_text(json.dumps({
        "source_gates_run": str(parent),
        "source_tracking_run": str(tracking_root),
        "source_detection_run": str(detector_root),
        "gate_decisions_recomputed": False,
        "method": "signed_dynamic_pa",
        "parameters": settings,
        "admitted_families": admitted,
        "recovered_patches_all_frames": int(result.total_recovered_patches),
    }, indent=2) + "\n")
    outputs = _stage_outputs(dest, {
        "recovered_patch_masks": mask,
        "recovery_metrics": summary_path,
        "family_recovered_patch_masks": family_file,
    })
    complete_stage(dest, "recovery", {
        "parent_run": str(parent),
        "admitted_family_ids": stage["outputs"]["admitted_family_ids"]["path"],
    }, outputs, "signed_dynamic_pa:apply_after_gates", settings)
    return dest
