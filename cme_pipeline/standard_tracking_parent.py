"""Reuse a completed Tracking Run as the explicit parent for Association/Gates."""
from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.run_layout import verify_stage, initialize_run, complete_stage
from cme_pipeline.core.stage import TrackingOutput
from cme_pipeline.core.standard_stages import stages_until
from cme_pipeline.standard_input_run import resolve_detection_parent
from cme_pipeline.standard_runner import _revision, _context, _params, _stage_outputs
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage


def resolve_tracking_parent(config, input_run):
    parent = Path(input_run).expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    manifest = json.loads((parent / "run_manifest.json").read_text())
    if manifest["stages"].get("tracking") != "COMPLETED":
        raise ValueError("Input Run has no completed Tracking stage")
    meta = verify_stage(parent, "tracking")
    old = json.loads((parent / "config_snapshot.json").read_text())
    if str(old["experiment"]["date"]) != str(config.data["experiment"]["date"]):
        raise ValueError("Tracking input date mismatch")
    if (old.get("tracking") or {}) != _params(config, "tracking"):
        raise ValueError("Parent Tracking configuration differs from this experiment")
    if meta["implementation"] != config.method("tracking"):
        raise ValueError("Parent Tracking method mismatch")
    origin = manifest.get("input_provenance") or {}
    if manifest["stages"].get("detection") == "COMPLETED":
        detection_run = parent
    elif origin.get("input_stage") == "detection" and origin.get("input_run"):
        detection_run = Path(origin["input_run"]).resolve()
    else:
        raise ValueError("Missing verifiable Detection ancestor")
    det_root, det_dir, _ = resolve_detection_parent(config, detection_run)
    def output(key):
        p = (parent / meta["outputs"][key]["path"]).resolve()
        if not p.is_relative_to(parent):
            raise ValueError("Artifact path escapes parent Run")
        return p
    provenance = json.loads(output("tracking_provenance").read_text())
    if (str(provenance.get("requested_date")) != str(config.data["experiment"]["date"])
            or provenance.get("method") != config.method("tracking")
            or Path(provenance["detection_dir"]).resolve() != det_dir):
        raise ValueError("Tracking metadata incompatible with its Detection ancestor")
    observed = pd.read_csv(output("track_states"))
    owned = np.load(output("track_ownership"))
    detected = load_live_tracking_input(det_dir)
    saved_dir = output("track_ownership").parent
    active_path = saved_dir / "tracking_input_activation_p04.npy"
    labels_path = saved_dir / "tracking_input_component_labels.npy"
    if not active_path.is_file() or not labels_path.is_file():
        raise ValueError("Parent missing original post-pre-size tracking inputs")
    active = np.load(active_path).astype(bool)
    labels = np.load(labels_path)
    if (active.shape != detected.patch_activation_grid4.shape
            or labels.shape != detected.raw_component_labels.shape
            or owned.shape != labels.shape):
        raise ValueError("Tracking input shapes disagree with Detection")
    usable = detected.manifest_df["usable_detection"].astype(bool).to_numpy()
    if (np.any(active & ~detected.patch_activation_grid4)
            or np.any((labels[usable] > 0) != active[usable])
            or np.any(active[~usable])
            or np.any(labels[~usable] != -1)
            or np.any((owned > 0) & ~active)):
        raise ValueError("Saved tracking input inconsistent with Detection")
    if observed.empty or not observed["pair_index"].isin(detected.manifest_df["pair_index"]).all():
        raise ValueError("Invalid Tracking state rows")
    detected = replace(detected, patch_activation_grid4=active,
                       raw_component_labels=labels.astype(np.int32))
    tracking = TrackingOutput(
        manifest_df=detected.manifest_df,
        track_states_df=observed,
        patch_component_labels=detected.raw_component_labels,
        patch_activation_grid4=detected.patch_activation_grid4,
        frame_cache=detected.frame_cache,
        geometry=detected.geometry,
        total_tracks=int(provenance["total_tracks"]),
        track_ownership_masks=np.asarray(owned, dtype=np.int32),
        tracking_method=config.method("tracking"),
        tracking_parameters=provenance.get("parameters"),
        dataset_metadata=detected.dataset_metadata,
    )
    return parent, tracking, meta, det_root, (active_path, labels_path)


def execute_from_tracking(config, *, input_run, until, run_dir, visualize=None):
    order = stages_until(until)
    if len(order) not in (4, 5):
        raise ValueError("Tracking input supports association or gates only")
    for item in order[3:]:
        if config.vis_mode(item.name.value, visualize) != "off":
            raise NotImplementedError("Visualization is not integrated")
    if config.method("association") != "identity" or config.method("gates") != "consecutive_size_qualified":
        raise ValueError("Only Identity/Consecutive-Qualified Gates supported")
    gates = _params(config, "gates")
    if (gates.get("outward_split") or {}).get("enabled", False):
        raise ValueError("Outward Split is excluded")
    pers = gates.get("persistence") or {}
    if pers.get("method", "consecutive_size_qualified") != "consecutive_size_qualified":
        raise ValueError("Unsupported gate persistence method")
    parent, tracking, parent_meta, detection_run, extra = resolve_tracking_parent(config, input_run)
    target = Path(run_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    from cme_pipeline.core.run_layout import _hash_file
    extra_hashes = {file.name: {"path": str(file.relative_to(parent)), "sha256": _hash_file(file)}
                    for file in extra}
    cfg = PipelineConfig()
    cfg.date_str = int(config.data["experiment"]["date"])
    cfg.tracking.method = config.method("tracking")
    cfg.association.method = "identity"
    cfg.birth_gate.max_birth_radius_px = float(gates.get("birth_max_radius_px", 30))
    cfg.size_gate.min_detected_patches = int(pers.get("min_detected_patches", 10))
    cfg.size_gate.min_observed_pairs = int(pers.get("min_observed_pairs", 3))
    cfg.size_gate.size_qualified_persistence = True
    cfg.size_gate.consecutive_size_qualified = True
    initialize_run(target, config.data, _revision(), {
        "mode": "explicit_input_run", "input_run": str(parent),
        "input_stage": "tracking", "detection_ancestor": str(detection_run),
        "input_outputs": parent_meta["outputs"], "extra_inputs": extra_hashes,
    })
    tracking = adapt_tracking_output(tracking, max_birth_radius_px=cfg.birth_gate.max_birth_radius_px)
    ctx = _context(target, "association")
    assoc = IdentityAssociationStage().run(tracking, cfg, ctx)
    outputs = _stage_outputs(target, {
        "family_membership": ctx.tables_dir / "02_full_day_e1_track_states.csv",
        "association_decisions": ctx.tables_dir / "03_full_day_parent_child_edges.csv",
    })
    complete_stage(target, "association", {"parent_run": str(parent)},
                   outputs, "identity", _params(config, "association"))
    if len(order) == 4:
        return target
    ctx = _context(target, "gates")
    birth = FamilyBirthRadiusGateStage().run(assoc, cfg, ctx)
    size = FamilySizeGateStage().run(birth, cfg, ctx)
    admitted = ctx.run_dir / "admitted_families.json"
    admitted.write_text(json.dumps({"family_ids": sorted(size.admitted_family_ids),
                                   "track_ids": sorted(size.admitted_track_ids)}, indent=2))
    qualified = ctx.run_dir / "qualified_pairs.csv"
    size.family_summary_df[["family_id", "size_qualified_pairs",
                            "longest_consecutive_size_qualified"]].to_csv(qualified, index=False)
    gate_outputs = _stage_outputs(target, {
        "gate_decisions": ctx.tables_dir / "04_full_day_family_summary.csv",
        "admitted_family_ids": admitted, "qualified_pair_ranges": qualified,
    })
    complete_stage(target, "gates", {"family_membership": outputs["family_membership"]},
                   gate_outputs, "birth+consecutive_size_qualified", gates)
    return target
