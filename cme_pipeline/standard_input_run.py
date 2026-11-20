"""Run new Tracking/Association/Gates against one explicitly selected Detection Run.

This adapter NEVER searches old runs, resumes or modifies its input. It uses
verified parent Stage 02 outputs, records their SHA-256 provenance, and writes
only Stage 03 onward in a fresh output Run. It does not re-run D4.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.core.standard_stages import stages_until
from cme_pipeline.core.run_layout import initialize_run, verify_stage, complete_stage
from cme_pipeline.standard_runner import _revision, _context, _params, _stage_outputs
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.stages.tracking.registry import get_tracking_spec, get_tracking_stage
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output
from cme_pipeline.stages.gates.pre_tracking_component_size import filter_small_components
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage


def resolve_detection_parent(config: StandardConfig, parent: str | Path) -> tuple[Path, Path, dict]:
    """Check declared parent artifacts and scientific compatibility before creating output."""
    root = Path(parent).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["stages"].get("detection") != "COMPLETED":
        raise ValueError("Input Run has no completed Detection stage")
    metadata = verify_stage(root, "detection")
    if metadata["implementation"] != "d4" or config.method("detection") != "d4":
        raise ValueError("Input Run detector and selected detector must both be D4")
    # The parent's immutable detection settings are authoritative. No silent
    # scientific changes while reusing a result already computed elsewhere.
    if metadata["parameters"] != _params(config, "detection"):
        raise ValueError("Detection config differs from parent Run; create a new Detection Run")
    parent_cfg = json.loads((root / "config_snapshot.json").read_text())
    if str(parent_cfg["experiment"]["date"]) != str(config.data["experiment"]["date"]):
        raise ValueError("Parent Run and requested experiment date disagree")
    active = (root / metadata["outputs"]["patch_activation"]["path"]).resolve()
    if not active.is_relative_to(root) or not active.is_file():
        raise ValueError("Invalid input Detection file path")
    detector = active.parent
    detector_meta = json.loads((detector / "detection_metadata.json").read_text())
    if str(detector_meta["requested_date"]) != str(config.data["experiment"]["date"]):
        raise ValueError("Detector metadata date mismatch")
    if detector_meta.get("detector") != "d4":
        raise ValueError("Parent detector is not D4")
    return root, detector, metadata


def execute_from_detection(
    config: StandardConfig, *, input_run: str | Path, until: str,
    run_dir: str | Path, visualize: str | None = None,
) -> Path:
    selected = stages_until(until)
    if not 3 <= len(selected) <= 5:
        raise ValueError("Detection input supports --until tracking, association, or gates")
    for stage in selected[2:]:
        if config.vis_mode(stage.name.value, visualize) != "off":
            raise NotImplementedError("Non-off visualization is not yet integrated")
    parent, detector_dir, parent_meta = resolve_detection_parent(config, input_run)
    run = Path(run_dir).expanduser().resolve()
    if run.exists():
        raise FileExistsError(run)
    if run == parent:
        raise ValueError("Output must differ from parent input Run")

    tr = _params(config, "tracking")
    method = config.method("tracking")
    if method not in ("a3", "a4", "a5b", "s1", "s2"):
        raise ValueError("Supported input-run Tracking methods: a3, a4, a5b, s1, s2")
    params = tr.get("parameters") or {}
    if not isinstance(params, dict):
        raise ValueError("tracking.parameters must be a mapping")
    spec = get_tracking_spec(method)
    bad = set(params) - spec.supported_parameters
    if bad:
        raise ValueError(f"Unsupported {method} tracking parameters: {sorted(bad)}")
    pre = tr.get("pre_size") or {"enabled": True, "connectivity": 8, "min_component_patches": 8}
    if not isinstance(pre, dict) or pre.get("connectivity", 8) != 8:
        raise ValueError("Pre-size must be a mapping with connectivity 8")
    if int(pre.get("min_component_patches", 8)) < 1:
        raise ValueError("Invalid pre-size minimum")
    if config.method("association") != "identity":
        raise ValueError("Only Identity Association supported")
    if config.method("gates") != "consecutive_size_qualified":
        raise ValueError("Only consecutive size-qualified gate supported")
    gates = _params(config, "gates")
    if (gates.get("outward_split") or {}).get("enabled", False):
        raise ValueError("Outward Split excluded from standard baseline")
    pers = gates.get("persistence") or {}
    if pers.get("method", "consecutive_size_qualified") != "consecutive_size_qualified":
        raise ValueError("Unsupported persistence method")

    # Preserve parent's file hashes in child provenance, not copied upstream data.
    provenance = {
        "mode": "explicit_input_run",
        "input_run": str(parent),
        "input_stage": "detection",
        "input_outputs": parent_meta["outputs"],
        "input_source_revision": json.loads((parent / "run_manifest.json").read_text()).get("source_revision"),
    }
    initialize_run(run, config.data, _revision(), provenance)
    detected = load_live_tracking_input(detector_dir)
    if pre.get("enabled", True):
        detected = filter_small_components(detected,
            min_component_patches=int(pre.get("min_component_patches", 8)))
    cfg = PipelineConfig()
    cfg.date_str = int(config.data["experiment"]["date"])
    cfg.detection.method = "d4"
    cfg.detection.denoiser = "gaussian"
    cfg.tracking.method = method
    cfg.tracking.parameters = dict(params)
    cfg.association.method = "identity"
    cfg.birth_gate.max_birth_radius_px = float(gates.get("birth_max_radius_px", 30))
    cfg.size_gate.min_detected_patches = int(pers.get("min_detected_patches", 10))
    cfg.size_gate.min_observed_pairs = int(pers.get("min_observed_pairs", 3))
    cfg.size_gate.size_qualified_persistence = True
    cfg.size_gate.consecutive_size_qualified = True

    ctx = _context(run, "tracking")
    result = get_tracking_stage(method).run(detected, cfg, ctx)
    if result.track_ownership_masks is None:
        raise RuntimeError("Tracker did not produce owned masks")
    np.save(ctx.arrays_dir / "track_ownership_masks.npy", result.track_ownership_masks)
    np.save(ctx.arrays_dir / "tracking_input_activation_p04.npy", detected.patch_activation_grid4)
    np.save(ctx.arrays_dir / "tracking_input_component_labels.npy", detected.raw_component_labels)
    result.track_states_df.to_csv(ctx.tables_dir / "live_tracking_states.csv", index=False)
    result.manifest_df.to_csv(ctx.tables_dir / "live_observation_manifest.csv", index=False)
    (ctx.run_dir / "tracking_metadata.json").write_text(json.dumps({
        "requested_date": str(cfg.date_str), "method": method,
        "detection_dir": str(detector_dir), "total_tracks": int(result.total_tracks),
        "parameters": result.tracking_parameters, "pre_size": pre,
        "parent_run": str(parent),
    }, indent=2))
    track_files = _stage_outputs(run, {
        "track_states": ctx.tables_dir / "live_tracking_states.csv",
        "track_ownership": ctx.arrays_dir / "track_ownership_masks.npy",
        "tracking_provenance": ctx.run_dir / "tracking_metadata.json",
    })
    complete_stage(run, "tracking", {
        "patch_activation": parent_meta["outputs"]["patch_activation"]["path"],
        "parent_run": str(parent),
    }, track_files, method, tr)
    if len(selected) == 3:
        return run

    result = adapt_tracking_output(result, max_birth_radius_px=cfg.birth_gate.max_birth_radius_px)
    assoc_ctx = _context(run, "association")
    associated = IdentityAssociationStage().run(result, cfg, assoc_ctx)
    assoc_files = _stage_outputs(run, {
        "family_membership": assoc_ctx.tables_dir / "02_full_day_e1_track_states.csv",
        "association_decisions": assoc_ctx.tables_dir / "03_full_day_parent_child_edges.csv",
    })
    complete_stage(run, "association",
                   {"track_states": track_files["track_states"]},
                   assoc_files, "identity", _params(config, "association"))
    if len(selected) == 4:
        return run

    gate_ctx = _context(run, "gates")
    birth = FamilyBirthRadiusGateStage().run(associated, cfg, gate_ctx)
    finished = FamilySizeGateStage().run(birth, cfg, gate_ctx)
    admitted = gate_ctx.run_dir / "admitted_families.json"
    admitted.write_text(json.dumps({
        "family_ids": sorted(finished.admitted_family_ids),
        "track_ids": sorted(finished.admitted_track_ids),
    }, indent=2))
    qualified = gate_ctx.run_dir / "qualified_pairs.csv"
    finished.family_summary_df[["family_id", "size_qualified_pairs",
                               "longest_consecutive_size_qualified"]].to_csv(qualified, index=False)
    gate_files = _stage_outputs(run, {
        "gate_decisions": gate_ctx.tables_dir / "04_full_day_family_summary.csv",
        "admitted_family_ids": admitted,
        "qualified_pair_ranges": qualified,
    })
    complete_stage(run, "gates", {
        "family_membership": assoc_files["family_membership"]},
        gate_files, "birth+consecutive_size_qualified", gates)
    return run
