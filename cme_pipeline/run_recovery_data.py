"""Run Recovery as a standalone command using immutable saved Gate decisions.

No Birth/Size/Persistence decision is recalculated. Gate outputs are read from
the provided Gate run, and ownership is loaded from its original A5b run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import AssociationOutput, BirthGateOutput, SizeGateOutput, TrackingOutput
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.stages.recovery.v02_apply_after_gates import ApplyRecoveryAfterGatesStage
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output


def run_saved_recovery(
    gates_run: str | Path, *, run_name: str, recovery_inner_radius_px: float,
) -> Path:
    gates = Path(gates_run).expanduser().resolve()
    summary = json.loads((gates / "gates_summary.json").read_text())
    if summary.get("recovery_executed") is not False:
        raise ValueError("Expected a completed pure Gate run (recovery_executed=false)")
    if not np.isfinite(recovery_inner_radius_px) or recovery_inner_radius_px <= 0:
        raise ValueError("Explicit positive recovery_inner_radius_px required")
    tracking_run = Path(summary["source_tracking_run"]).resolve()
    tracking_meta = json.loads((tracking_run / "tracking_metadata.json").read_text())
    detector = Path(summary["source_detection_dir"]).resolve()
    if Path(tracking_meta["detection_dir"]).resolve() != detector:
        raise ValueError("Tracking and Gate runs refer to different detectors")
    detected = load_live_tracking_input(detector)
    if str(tracking_meta["requested_date"]) != str(summary["date"]):
        raise ValueError("Tracking date and Gate date mismatch")

    # Reapply the exact saved A5b input, including any pre-tracking size filter.
    act_file = tracking_run / "arrays" / "tracking_input_activation_p04.npy"
    lab_file = tracking_run / "arrays" / "tracking_input_component_labels.npy"
    if act_file.exists() != lab_file.exists():
        raise ValueError("Incomplete saved tracking input pair")
    if act_file.exists():
        from dataclasses import replace
        active = np.load(act_file)
        labels = np.load(lab_file)
        if active.shape != detected.patch_activation_grid4.shape or labels.shape != detected.raw_component_labels.shape:
            raise ValueError("Tracking input shape mismatch")
        if np.any(active & ~detected.patch_activation_grid4):
            raise ValueError("Tracking input contains patches absent from original C3")
        usable = detected.manifest_df["usable_detection"].astype(bool).to_numpy()
        if np.any((labels[usable] > 0) != active[usable]):
            raise ValueError("Tracking input labels/activation mismatch")
        if np.any(active[~usable]) or np.any(labels[~usable] != -1):
            raise ValueError("Unavailable tracking frame modified")
        detected = replace(detected, patch_activation_grid4=active.astype(bool),
                           raw_component_labels=labels.astype(np.int32))

    own = np.load(tracking_run / "arrays" / "track_ownership_masks.npy")
    states = pd.read_csv(tracking_run / "tables" / "live_tracking_states.csv")
    if own.shape != detected.patch_activation_grid4.shape:
        raise ValueError("Tracking ownership shape mismatch")
    if np.any((own > 0) & ~detected.patch_activation_grid4):
        raise ValueError("Ownership outside tracking input")
    if not states["pair_index"].isin(detected.manifest_df["pair_index"]).all():
        raise ValueError("Unknown pair index in saved tracking states")

    family = pd.read_csv(gates / "stages" / "size_gate" / "family_summary.csv")
    if family.empty or family["family_id"].duplicated().any():
        raise ValueError("Missing or duplicate Family decisions")
    if not set(family["status"]).issubset({"ADMITTED", "REJECTED"}):
        raise ValueError("Unexpected Gate status")
    family_members = {}
    track_to_family = {}
    for fid in family["family_id"]:
        fid = str(fid)
        if not fid.startswith("FAM_T"):
            raise ValueError("Only A5b Identity Association Family IDs supported")
        tid = int(fid.removeprefix("FAM_T"))
        if tid in track_to_family:
            raise ValueError("Ambiguous Family ID mapping")
        family_members[fid] = [tid]
        track_to_family[tid] = fid
    observed_ids = set(states.loc[states["state"] == "OBSERVED", "track_id"].astype(int))
    if set(track_to_family) != observed_ids:
        raise ValueError("Gate decisions do not match original tracked IDs")

    admitted_families = family.loc[family["status"] == "ADMITTED", "family_id"].astype(str).tolist()
    admitted_tracks = sorted(track_to_family.keys() & {int(fid.removeprefix("FAM_T")) for fid in admitted_families})
    if len(admitted_families) != summary["size_gate_admitted"] or len(family) != summary["candidate_families"]:
        raise ValueError("Gate decision counts differ from saved summary")

    cfg = PipelineConfig()
    cfg.date_str = int(summary["date"])
    cfg.run_name = run_name
    cfg.tracking.method = "a5b"
    cfg.association.method = "identity"
    cfg.birth_gate.max_birth_radius_px = float(summary["birth_max_radius_px"])
    cfg.size_gate.min_detected_patches = int(summary["min_detected_patches"])
    cfg.size_gate.min_observed_pairs = int(summary.get("min_observed_pairs", 1))
    detected.geometry.occulter_inner_radius_px = float(recovery_inner_radius_px)
    dest = (Path(cfg.repo_root).resolve() / cfg.output_base_dir / run_name).resolve()
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite Recovery run: {dest}")
    track = TrackingOutput(
        manifest_df=detected.manifest_df, track_states_df=states,
        patch_component_labels=detected.raw_component_labels,
        patch_activation_grid4=detected.patch_activation_grid4,
        frame_cache=detected.frame_cache, geometry=detected.geometry,
        total_tracks=int(tracking_meta["total_tracks"]),
        track_ownership_masks=np.asarray(own, dtype=np.int32),
        tracking_method="a5b", dataset_metadata=detected.dataset_metadata,
    )
    track = adapt_tracking_output(track, max_birth_radius_px=cfg.birth_gate.max_birth_radius_px)
    # These are saved decisions, not new Gate computations.
    assoc = AssociationOutput(
        tracking_output=track, recovery_output=None,
        track_to_family=track_to_family, family_members=family_members,
        family_anchors={}, edges_df=pd.DataFrame(),
        track_membership_df=pd.DataFrame(), candidate_family_count=len(family),
    )
    birth_pass = family.loc[family["anchor_count"] > 0, "family_id"].astype(str).tolist()
    birth = BirthGateOutput(
        association_output=assoc, birth_gated_family_ids=birth_pass,
        max_birth_radius_px=cfg.birth_gate.max_birth_radius_px,
    )
    gated = SizeGateOutput(
        birth_gate_output=birth,
        admitted_family_ids=admitted_families,
        admitted_track_ids=admitted_tracks,
        family_summary_df=family,
        size_gate_decisions_df=family[["family_id", "status", "rejection_reason"]].copy(),
        min_detected_patches=cfg.size_gate.min_detected_patches,
    )
    ctx = PipelineContext(cfg)
    result = ApplyRecoveryAfterGatesStage().run(gated, cfg, ctx)
    report = {
        "source_gates_run": str(gates),
        "source_tracking_run": str(tracking_run),
        "source_detection_dir": str(detector),
        "gate_decisions_recomputed": False,
        "recovery_executed": True,
        "recovery_inner_radius_px": float(recovery_inner_radius_px),
        "admitted_families": len(admitted_families),
        "recovered_patches_all_frames": result.total_recovered_patches,
        "status": "RECOVERY_READY",
    }
    (ctx.run_dir / "recovery_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return ctx.run_dir
