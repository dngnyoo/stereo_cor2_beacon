"""Apply Identity -> Birth Gate -> Size Gate to saved A5b runs, with optional Recovery.

No acquisition, RD preprocessing, detection or tracking is recomputed.
The recovery inner radius (pixels) is mandatory: 2024 solar-radius calibration
is not assumed from a 2010 reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import TrackingOutput
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output
from cme_pipeline.stages.recovery.v01_signed_pa_recovery import SignedPARecoveryStage
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage


def run_saved_a5b_gates(
    tracking_run: str | Path, *, run_name: str,
    recovery_inner_radius_px: float | None = None,
    birth_max_radius_px: float = 30.0,
    min_detected_patches: int = 5,
    min_observed_pairs: int = 1,
    with_recovery: bool = False,
) -> Path:
    """Reconstruct saved A5b TrackingOutput; run pure Gates and optionally Recovery."""
    run = Path(tracking_run).expanduser().resolve()
    meta = json.loads((run / "tracking_metadata.json").read_text())
    if meta.get("method") != "a5b":
        raise ValueError("Only saved A5b runs are supported")
    if with_recovery and (not np.isfinite(recovery_inner_radius_px)
                          or recovery_inner_radius_px <= 0):
        raise ValueError("Recovery requires a positive --recovery-inner-radius-px")
    if not (np.isfinite(birth_max_radius_px) and birth_max_radius_px > 0):
        raise ValueError("Invalid birth maximum radius")
    if min_detected_patches < 1:
        raise ValueError("min_detected_patches must be >=1")
    if min_observed_pairs < 1:
        raise ValueError("min_observed_pairs must be >=1")
    detdir = Path(meta["detection_dir"]).resolve()
    detected = load_live_tracking_input(detdir)
    # Replay the exact A5b input, including optional Pre-size filtering.
    act_path = run / "arrays" / "tracking_input_activation_p04.npy"
    labels_path = run / "arrays" / "tracking_input_component_labels.npy"
    if act_path.exists() != labels_path.exists():
        raise ValueError("Incomplete saved tracking input: activation/labels must coexist")
    if act_path.exists():
        from dataclasses import replace
        filtered_active = np.load(act_path)
        filtered_labels = np.load(labels_path)
        if filtered_active.shape != detected.patch_activation_grid4.shape or filtered_labels.shape != detected.raw_component_labels.shape:
            raise ValueError("Saved A5b input shape differs from C3 detection")
        usable = detected.manifest_df["usable_detection"].astype(bool).to_numpy()
        if np.any(filtered_active & ~detected.patch_activation_grid4):
            raise ValueError("Saved A5b input introduced patches absent from C3")
        if np.any((filtered_labels[usable] > 0) != filtered_active[usable]):
            raise ValueError("Saved A5b input component labels disagree with active grid")
        if np.any(filtered_active[~usable]) or np.any(filtered_labels[~usable] != -1):
            raise ValueError("Unavailable observations changed in saved A5b input")
        detected = replace(
            detected,
            patch_activation_grid4=filtered_active.astype(bool),
            raw_component_labels=filtered_labels.astype(np.int32),
        )
    observed = pd.read_csv(run / "tables" / "live_tracking_states.csv")
    own = np.load(run / "arrays" / "track_ownership_masks.npy")
    if own.shape != detected.raw_component_labels.shape:
        raise ValueError("Saved A5b ownership does not align to detector inputs")
    if np.any((own > 0) & ~detected.patch_activation_grid4):
        raise ValueError("Track ownership outside active detector mask")
    if observed.empty:
        raise ValueError("Saved tracking states empty: inspect run before continuing")
    if not bool(observed["pair_index"].isin(detected.manifest_df["pair_index"]).all()):
        raise ValueError("Tracking states reference unknown observations")

    cfg = PipelineConfig()
    cfg.date_str = int(meta["requested_date"])
    cfg.tracking.method = "a5b"
    cfg.association.method = "identity"
    cfg.birth_gate.max_birth_radius_px = birth_max_radius_px
    cfg.size_gate.min_detected_patches = int(min_detected_patches)
    cfg.size_gate.min_observed_pairs = int(min_observed_pairs)
    cfg.run_name = run_name
    # This is a user-specified *operational pixel cutoff*, not an inferred 2 R_sun.
    if with_recovery:
        detected.geometry.occulter_inner_radius_px = float(recovery_inner_radius_px)
    tracking = TrackingOutput(
        manifest_df=detected.manifest_df,
        track_states_df=observed,
        patch_component_labels=detected.raw_component_labels,
        patch_activation_grid4=detected.patch_activation_grid4,
        frame_cache=detected.frame_cache,
        geometry=detected.geometry,
        total_tracks=int(meta["total_tracks"]),
        track_ownership_masks=np.asarray(own, dtype=np.int32),
        tracking_method="a5b",
        dataset_metadata=detected.dataset_metadata,
    )
    tracking = adapt_tracking_output(tracking, max_birth_radius_px=birth_max_radius_px)
    # Avoid writing into an existing run or reusing the original A5b run directory.
    output = (Path(cfg.repo_root).resolve() / cfg.output_base_dir / run_name).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output}")
    ctx = PipelineContext(cfg)
    # Gate-only mode does not instantiate a Recovery operator.
    assoc = IdentityAssociationStage().run(tracking, cfg, ctx)
    birth = FamilyBirthRadiusGateStage().run(assoc, cfg, ctx)
    gated = FamilySizeGateStage().run(birth, cfg, ctx)
    if with_recovery:
        from cme_pipeline.stages.recovery.v02_apply_after_gates import ApplyRecoveryAfterGatesStage
        size = ApplyRecoveryAfterGatesStage().run(gated, cfg, ctx)
    else:
        size = gated
    np.save(ctx.arrays_dir / "saved_track_ownership.npy", own)
    np.save(ctx.arrays_dir / "saved_detection_p04.npy", detected.patch_activation_grid4)
    summary = {
        "date": str(meta["requested_date"]),
        "source_tracking_run": str(run),
        "source_detection_dir": str(detdir),
        "processing": ["identity", "birth_gate", "size_gate"] + (["recovery_apply"] if with_recovery else []),
        "recovery_inner_radius_px": float(recovery_inner_radius_px) if with_recovery else None,
        "recovery_inner_radius_interpretation": "explicit operational cutoff; not calibrated to 2 solar radii",
        "birth_max_radius_px": float(birth_max_radius_px),
        "min_detected_patches": int(min_detected_patches),
        "min_observed_pairs": int(min_observed_pairs),
        "tracks": int(meta["total_tracks"]),
        "candidate_families": int(assoc.candidate_family_count),
        "birth_gate_pass": len(birth.birth_gated_family_ids),
        "size_gate_admitted": len(size.admitted_family_ids),
        "total_recovered_patches_all_frames": int(size.total_recovered_patches) if with_recovery else None,
        "recovery_executed": with_recovery,
        "status": "GATES_AND_RECOVERY_READY" if with_recovery else "GATES_ONLY_NO_RECOVERY",
    }
    (ctx.run_dir / "gates_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return ctx.run_dir
