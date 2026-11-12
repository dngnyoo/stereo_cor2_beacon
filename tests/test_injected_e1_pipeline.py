"""Regression safeguards for injecting independently generated TrackingOutput."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline
from cme_pipeline.core.stage import PipelineStage, TrackingOutput
from cme_pipeline.stages.tracking.v01_baseline_tracking import E1TrackingStage


class EchoTrackingStage(PipelineStage):
    name = "tracking"
    version = "test"
    def compute_fingerprint(self, inputs, config, context):
        return "independent-e1-test"
    def run(self, inputs, config, context):
        assert inputs == {"source": "independent_e1"}
        return inputs


def test_pipeline_accepts_initial_input(tmp_path):
    cfg = PipelineConfig(repo_root=str(tmp_path), run_name="test_injected_pipeline",
                         cache_enabled=False)
    pipeline = Pipeline(cfg, stages=[EchoTrackingStage()])
    result = pipeline.run(initial_input={"source": "independent_e1"})
    assert result["status"] == "SUCCESS"
    assert result["stage_outputs"]["tracking"] == {"source": "independent_e1"}
    assert (tmp_path / "runs/test_injected_pipeline/manifest.json").exists()


def test_e1_does_not_load_frozen_table_for_injected_tracking():
    cfg = PipelineConfig()
    data = TrackingOutput(
        manifest_df=pd.DataFrame(), track_states_df=pd.DataFrame(),
        patch_component_labels=np.zeros((0, 64, 64), dtype=np.int32),
        patch_activation_grid4=np.zeros((0, 64, 64), dtype=bool),
        frame_cache=[], geometry=None, tracking_method="e1",
        dataset_metadata={"replay_mode": "independent_e1"},
    )
    context = SimpleNamespace(info=lambda message: None)
    assert E1TrackingStage().run(data, cfg, context) is data
