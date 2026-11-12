"""E1 1:1 Patch Tracking Stage (Baseline Gate OFF)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import DetectionOutput, PipelineStage, TrackingOutput
from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output
from cme_pipeline.stages.tracking.registry import resolve_and_validate_tracking_parameters
from cme_pipeline.utils.io import compute_dict_hash


class E1TrackingStage(PipelineStage):
    """Executes or loads baseline E1 1:1 Patch Tracking from shared frozen detection inputs."""
    name: str = "tracking"
    version: str = "1.0.0"

    def compute_fingerprint(self, inputs: Any, config: PipelineConfig, context: PipelineContext) -> str:
        params = resolve_and_validate_tracking_parameters("e1", config.tracking)
        desc: Dict[str, Any] = {
            "stage": self.name,
            "version": self.version,
            "date_str": config.date_str,
            "method": config.tracking.method,
            "alpha": params["alpha"],
            "patch_size": params["patch_size"],
        }
        if config.tracking.parameters:
            desc["parameters"] = params
        return compute_dict_hash(desc)

    def run(self, inputs: Any, config: PipelineConfig, context: PipelineContext) -> TrackingOutput:
        effective_params = resolve_and_validate_tracking_parameters("e1", config.tracking)
        if isinstance(inputs, TrackingOutput):
            # E1 has already been recomputed from independent C0 detections.
            # Do not read the historical Exp 11 state table in this path.
            if inputs.tracking_method != "e1":
                raise ValueError("Injected E1 tracking output must have tracking_method='e1'")
            if not inputs.dataset_metadata or not inputs.dataset_metadata.get("replay_mode"):
                raise ValueError("Injected tracking must include live/replay provenance")
            context.info("Using independently recomputed E1 TrackingOutput (no Frozen CSV read).")
            return inputs
        context.info("Running E1TrackingStage (2010-12-12 Baseline)...")

        # 1. Obtain shared frozen preprocessing & hierarchical detection inputs
        if isinstance(inputs, DetectionOutput):
            det_out = inputs
        else:
            det_out = load_frozen_detection_inputs(config, context)

        df_manifest = det_out.manifest_df
        n_pairs = len(df_manifest)
        geom = det_out.geometry
        grid4_all = det_out.patch_activation_grid4
        lbls_all = det_out.raw_component_labels

        # 2. Load E1 Tracking Baseline Track Table (Exp 11)
        exp11_dir = context.repo_root / "validation_20101212/experiments/11_d7_patch_first_tracking_feasibility"
        df_states_e1 = pd.read_csv(exp11_dir / "results/tables/03_patch_e1_track_states.csv")

        obs_all = df_states_e1[df_states_e1["state"] == "OBSERVED"].copy()
        birth_df = obs_all.sort_values(["track_id", "pair_index"]).groupby("track_id").first().reset_index()
        track_birth_rmin = {int(r["track_id"]): float(r["birth_r_min_px"]) for _, r in birth_df.iterrows()}
        anchor_tracks = sorted(
            [tid for tid, rmin in track_birth_rmin.items() if rmin <= config.birth_gate.max_birth_radius_px]
        )

        context.info(f"Loaded {n_pairs} frames, {len(birth_df)} E1 tracks (Gate OFF), {len(anchor_tracks)} solar root anchors.")

        # Save stage tables
        stage_dir = context.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        df_manifest.to_csv(stage_dir / "frame_manifest.csv", index=False)
        df_manifest.to_csv(context.tables_dir / "01_full_day_frame_manifest.csv", index=False)

        raw_out = TrackingOutput(
            manifest_df=df_manifest,
            track_states_df=df_states_e1,
            patch_component_labels=lbls_all,
            patch_activation_grid4=grid4_all,
            frame_cache=det_out.frame_cache,
            geometry=geom,
            total_tracks=len(birth_df),
            anchor_tracks=anchor_tracks,
            tracking_method="e1",
            tracking_parameters=effective_params,
            dataset_metadata=det_out.dataset_metadata,
        )
        return adapt_tracking_output(raw_out, max_birth_radius_px=config.birth_gate.max_birth_radius_px)
