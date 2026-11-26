"""Unit tests for SPT Downstream Adapter and End-to-End Downstream Stages."""

import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.stages.tracking.spt_downstream_adapter import (
    adapt_spt_tracking_output,
    run_spt_downstream_pipeline,
    VERIFIED_RSUN_PX,
)
from cme_pipeline.stages.tracking.v13_spt2_patch_ownership import SPT2PatchOwnershipTrackingStage
from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs
from tests.test_spt_synthetic import (
    create_synthetic_geometry,
    build_synthetic_detection_output,
    generate_arc_patches,
)


class TestSPTDownstreamAdapter(unittest.TestCase):
    """Test suite verifying SPT downstream adapter integrity and pipeline compatibility."""

    def test_adapt_spt_tracking_output_geometry(self):
        """Verifies solar radius and occulter parameters are properly populated."""
        geom = create_synthetic_geometry()
        geom.rsun_px = np.nan
        geom.occulter_inner_radius_px = np.nan

        track_df = pd.DataFrame([
            {"track_id": 1, "pair_index": 0, "state": "OBSERVED", "central_pa": 90.0, "front_radius": 25.0}
        ])
        manifest_df = pd.DataFrame([{"pair_index": 0, "end_time": "2024-05-03T01:00:00Z"}])
        own_masks = np.zeros((1, 64, 64), dtype=np.int32)
        own_masks[0, 10:15, 10:15] = 1

        from cme_pipeline.core.stage import TrackingOutput
        raw_out = TrackingOutput(
            manifest_df=manifest_df,
            track_states_df=track_df,
            patch_component_labels=np.zeros((1, 64, 64), dtype=np.int32),
            patch_activation_grid4=np.zeros((1, 64, 64), dtype=bool),
            frame_cache=[{"pair_idx": 0}],
            geometry=geom,
            total_tracks=1,
            track_ownership_masks=own_masks,
            tracking_method="spt2",
            tracking_parameters={},
            dataset_metadata={},
        )

        adapted = adapt_spt_tracking_output(raw_out, date_str="20240503")
        self.assertAlmostEqual(adapted.geometry.rsun_px, 8.525)
        self.assertAlmostEqual(adapted.geometry.occulter_inner_radius_px, 17.050)
        self.assertIn("pa_min", adapted.track_states_df.columns)
        self.assertIn("pa_max", adapted.track_states_df.columns)
        self.assertIn("birth_r_min_px", adapted.track_states_df.columns)

    def test_end_to_end_spt_downstream_execution_20101212(self):
        """Tests complete execution through Stage 08 Catalog on 2010-12-12 benchmark."""
        cfg = PipelineConfig()
        cfg.date_str = 20101212
        ctx = PipelineContext(cfg)

        det = load_frozen_detection_inputs(cfg, ctx)

        cfg_run = PipelineConfig()
        cfg_run.date_str = 20101212
        cfg_run.tracking.method = "spt2"
        cfg_run.tracking.parameters = {
            "patch_size": 4,
            "max_cadence_gap_sec": 4500.0,
            "outward_gate_enabled": True,
            "outward_gate_tolerance_px": 4.0,
            "seed_min_radius_px": 22.0,
            "seed_max_radius_px": 30.0,
            "seed_score_cutoff": 0.0,
            "max_missed_frames": 3,
            "min_confirmation_pairs": 2,
            "min_component_patches": 1,
            "bridge_margin_threshold": 0.05,
            "separation_mode": "marker_geodesic",
        }
        cfg_run.birth_gate.max_birth_radius_px = 30.0
        cfg_run.size_gate.min_detected_patches = 10
        cfg_run.size_gate.min_observed_pairs = 3
        cfg_run.size_gate.size_qualified_persistence = True
        cfg_run.size_gate.consecutive_size_qualified = True

        stage = SPT2PatchOwnershipTrackingStage()
        track_out = stage.run(det, cfg_run, ctx)

        # Run downstream pipeline via adapter
        result = run_spt_downstream_pipeline(track_out, cfg_run, ctx)

        self.assertGreater(len(result.admitted_track_ids), 0)
        self.assertEqual(len(result.admitted_track_ids), len(result.admitted_family_ids))
        self.assertGreater(result.recovery_output.total_recovered_patches, 0)
        self.assertGreater(result.segmentation_output.total_segmented_pixels, 0)
        self.assertFalse(result.event_catalog_df.empty)

        # Check catalog kinematic columns
        expected_cols = {"event_id", "family_id", "cpa_deg", "angular_width_deg", "plane_of_sky_speed_km_s"}
        self.assertTrue(expected_cols.issubset(result.event_catalog_df.columns))


if __name__ == "__main__":
    unittest.main()
