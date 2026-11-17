"""Execute Signed PA Recovery only after the Family Gates have admitted tracks.

Input/output is SizeGateOutput to preserve existing segmentation contracts.
Pure SizeGateStage never invokes this module.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from cme_pipeline.core.stage import PipelineStage, SizeGateOutput
from cme_pipeline.stages.recovery.v01_signed_pa_recovery import DynamicPARecoveryRunner
from cme_pipeline.utils.io import compute_dict_hash


class ApplyRecoveryAfterGatesStage(PipelineStage):
    name = "recovery_apply"
    version = "1.0.0"

    def compute_fingerprint(self, inputs, config, context):
        if not isinstance(inputs, SizeGateOutput):
            raise TypeError("Recovery application requires a completed SizeGateOutput")
        return compute_dict_hash({
            "stage": self.name,
            "version": self.version,
            "method": config.recovery.method,
            "radial_window_px": config.recovery.radial_window_px,
            "pa_margin_deg": config.recovery.pa_margin_deg,
            "gaussian_sigma": config.recovery.gaussian_sigma,
            "admitted_family_ids": sorted(inputs.admitted_family_ids),
            "admitted_track_ids": sorted(inputs.admitted_track_ids),
            "upstream_size_gate_threshold": inputs.min_detected_patches,
        })

    def run(self, inputs: SizeGateOutput, config, context) -> SizeGateOutput:
        if not isinstance(inputs, SizeGateOutput):
            raise TypeError("Recovery application requires a completed SizeGateOutput")
        context.info("Running Recovery independently AFTER Family Size/Persistence Gates...")
        assoc = inputs.birth_gate_output.association_output
        tracking = assoc.tracking_output
        existing = assoc.recovery_output
        runner = (existing.recovery_func if existing is not None and
                  existing.recovery_func is not None else None)
        if runner is None:
            runner = DynamicPARecoveryRunner(tracking, config)

        n_frames = len(tracking.manifest_df)
        if inputs.admitted_track_ids:
            union = runner(inputs.admitted_track_ids)
        else:
            union = np.zeros((n_frames, 64, 64), dtype=bool)
        if union.shape != (n_frames, 64, 64):
            raise ValueError("Recovery mask must have shape (N,64,64)")

        families = {}
        for fid in inputs.admitted_family_ids:
            members = [tid for tid in assoc.family_members[fid]
                       if tid in inputs.admitted_track_ids]
            rec = runner(members) if members else np.zeros_like(union)
            if rec.shape != union.shape:
                raise ValueError(f"Invalid Family recovery shape for {fid}")
            families["rec_" + fid] = rec

        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.save(stage_dir / "recovered_patch_masks.npy", union)
        np.savez_compressed(stage_dir / "family_recovered_patch_masks.npz", **families)

        # Historical paths retained for downstream consumers.
        np.save(context.arrays_dir / "recovered_patch_masks_pipeline_b_full_day.npy", union)
        np.savez_compressed(context.arrays_dir / "family_recovered_patch_masks.npz", **families)
        total = int(union.sum())
        context.info(f"Recovery applied: {total} cumulative patches across admitted tracks.")
        return replace(
            inputs, recovered_patch_masks=union,
            family_recovered_patch_masks=families,
            total_recovered_patches=total,
        )
