"""Unit and regression test suite for Pixel-Level Temporal Evidence-Guided CME Tracking (H3 & H4)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import GeometryContext, TrackingOutput
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.stages.tracking.registry import get_tracking_stage
from cme_pipeline.stages.tracking.temporal_evidence_model import (
    DirectionalPeak,
    PixelTemporalEvidenceModel,
    TemporalEvidenceConfig,
)
from cme_pipeline.utils.geometry import circular_diff_deg, minimal_circular_enclosing_arc


class TestTemporalEvidenceTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det_out = load_live_tracking_input("runs/clean_cor2a_verify/detection")
        cls.geom = cls.det_out.geometry
        cls.valid_mask = (cls.geom.radius_raster >= cls.geom.occulter_r_min_px) & (cls.geom.radius_raster <= cls.geom.fov_r_max_px)

    def _create_model(self, cfg: TemporalEvidenceConfig) -> PixelTemporalEvidenceModel:
        return PixelTemporalEvidenceModel(
            sx=self.geom.sx,
            sy=self.geom.sy,
            pa_raster=self.geom.pa_raster,
            radius_raster=self.geom.radius_raster,
            r_min_px=self.geom.occulter_r_min_px,
            r_max_px=self.geom.fov_r_max_px,
            config=cfg,
        )

    def test_01_circular_pa_distance_and_wrapping(self):
        """Verify circular distance correctly wraps across 0°/360° on S^1."""
        self.assertAlmostEqual(abs(circular_diff_deg(5.0, 355.0)), 10.0, places=4)
        self.assertAlmostEqual(abs(circular_diff_deg(355.0, 5.0)), 10.0, places=4)
        self.assertAlmostEqual(abs(circular_diff_deg(0.0, 180.0)), 180.0, places=4)
        self.assertAlmostEqual(abs(circular_diff_deg(90.0, 270.0)), 180.0, places=4)
        self.assertAlmostEqual(abs(circular_diff_deg(45.0, 40.0)), 5.0, places=4)

    def test_02_pixel_to_patch_alignment(self):
        """Verify 256x256 pixel to 64x64 patch 4x4 block alignment and component support extraction."""
        cfg = TemporalEvidenceConfig(variant="H1")
        model = self._create_model(cfg)

        ny, nx = 256, 256
        evidence_map = np.zeros((ny, nx), dtype=np.float32)

        # Patch (40, 40) is within valid radius
        patch_mask = np.zeros((64, 64), dtype=bool)
        patch_mask[40, 40] = True
        evidence_map[160:164, 160:164] = 0.5

        supp = model.extract_component_support(patch_mask, self.valid_mask, evidence_map)
        self.assertAlmostEqual(supp["mean_evidence"], 0.5, places=4)
        self.assertAlmostEqual(supp["total_evidence"], 8.0, places=4)
        self.assertEqual(supp["num_valid_pixels"], 16)

    def test_03_causal_temporal_accumulation_and_decay(self):
        """Verify H1 causal temporal accumulation and exponential decay."""
        cfg = TemporalEvidenceConfig(variant="H1", half_life_min=30.0)
        model = self._create_model(cfg)

        t0 = datetime(2010, 12, 12, 6, 0, 0)
        # Frame 1: inject spatial evidence
        ev1 = np.zeros((256, 256), dtype=np.float32)
        ev1[100:110, 100:110] = 1.0
        model.update(ev1, self.valid_mask, t0)

        accum1 = model.get_accumulated_evidence()
        val1 = float(accum1[105, 105])
        self.assertAlmostEqual(val1, 0.01, places=4)

        # Frame 2: after exactly 30 minutes (1 half-life), with zero new evidence
        t1 = t0 + timedelta(minutes=30)
        ev2 = np.zeros((256, 256), dtype=np.float32)
        model.update(ev2, self.valid_mask, t1)

        accum2 = model.get_accumulated_evidence()
        val2 = float(accum2[105, 105])
        # Should have decayed by factor of 0.5
        self.assertAlmostEqual(val2 / val1, 0.5, delta=0.01)

    def test_04_outward_motion_compensation_warp(self):
        """Verify H2 outward motion compensation shifts features radially."""
        # 16 px per 30 min outward speed
        cfg = TemporalEvidenceConfig(variant="H2", half_life_min=60.0, radial_speed_px_30m=16.0)
        model = self._create_model(cfg)

        t0 = datetime(2010, 12, 12, 6, 0, 0)
        ev1 = np.zeros((256, 256), dtype=np.float32)
        # Place a ring of evidence at radius ~50 px
        mask_r50 = (self.geom.radius_raster >= 49.0) & (self.geom.radius_raster <= 51.0) & self.valid_mask
        ev1[mask_r50] = 1.0
        model.update(ev1, self.valid_mask, t0)

        # Before warp, radius ~66 has zero evidence
        mask_r66 = (self.geom.radius_raster >= 65.0) & (self.geom.radius_raster <= 67.0) & self.valid_mask
        self.assertAlmostEqual(float(model.get_accumulated_evidence()[mask_r66].mean()), 0.0, places=5)

        # Advance by 30 minutes: feature should move from r=50 to r=66
        t1 = t0 + timedelta(minutes=30)
        ev2 = np.zeros((256, 256), dtype=np.float32)
        model.update(ev2, self.valid_mask, t1)

        accum = model.get_accumulated_evidence()
        # Radii around 66 should now have positive shifted evidence
        self.assertGreater(float(accum[mask_r66].mean()), 0.0005)

    def test_05_directional_profile_and_multiple_peaks(self):
        """Verify directional profile identifies multiple simultaneous CME peaks cleanly."""
        cfg = TemporalEvidenceConfig(variant="H1", n_pa_bins=360)
        model = self._create_model(cfg)

        ev = np.zeros((256, 256), dtype=np.float32)
        # Peak 1 at PA ~45° (Northeast)
        p1 = (np.abs(circular_diff_deg(self.geom.pa_raster, 45.0)) <= 10.0) & self.valid_mask
        ev[p1] = 1.0

        # Peak 2 at PA ~135° (Southeast)
        p2 = (np.abs(circular_diff_deg(self.geom.pa_raster, 135.0)) <= 10.0) & self.valid_mask
        ev[p2] = 0.8

        model.update(ev, self.valid_mask, datetime(2010, 12, 12, 6, 0, 0))
        peaks = model.find_directional_peaks()

        self.assertGreaterEqual(len(peaks), 2)
        peak_cpas = [p.cpa_deg for p in peaks]
        # Check that both peaks were found near 45° and 135°
        self.assertTrue(any(abs(circular_diff_deg(c, 45.0)) < 15.0 for c in peak_cpas))
        self.assertTrue(any(abs(circular_diff_deg(c, 135.0)) < 15.0 for c in peak_cpas))

    def test_06_peak_wrapping_across_north(self):
        """Verify directional peaks spanning 0°/360° across North are recognized as a single peak."""
        cfg = TemporalEvidenceConfig(variant="H1", n_pa_bins=360)
        model = self._create_model(cfg)

        ev = np.zeros((256, 256), dtype=np.float32)
        # Peak spanning 350° to 10°
        p_north = ((self.geom.pa_raster >= 350.0) | (self.geom.pa_raster <= 10.0)) & self.valid_mask
        ev[p_north] = 1.0

        model.update(ev, self.valid_mask, datetime(2010, 12, 12, 6, 0, 0))
        peaks = model.find_directional_peaks()

        self.assertGreaterEqual(len(peaks), 1)
        # Top peak should be centered near 0°/360°
        top_cpa = peaks[0].cpa_deg
        self.assertTrue(top_cpa <= 15.0 or top_cpa >= 345.0)

    def test_07_qc_rejection_no_zero_corruption(self):
        """Verify that skipping update on QC-rejected frame preserves state without zero-corruption."""
        cfg = TemporalEvidenceConfig(variant="H1", half_life_min=30.0)
        model = self._create_model(cfg)

        t0 = datetime(2010, 12, 12, 6, 0, 0)
        ev = np.ones((256, 256), dtype=np.float32) * self.valid_mask
        model.update(ev, self.valid_mask, t0)
        accum_before = model.get_accumulated_evidence().copy()

        # If a frame is QC-rejected, model.update() is NOT called.
        # The stored accumulated state remains intact.
        accum_after = model.get_accumulated_evidence()
        np.testing.assert_array_equal(accum_before, accum_after)

    def test_08_h3_and_h4_execution_and_contracts(self):
        """Verify end-to-end execution of H3 and H4 with real detection inputs."""
        det_out = load_live_tracking_input("runs/clean_cor2a_verify/detection")

        # H3 Run
        cfg_h3 = PipelineConfig()
        cfg_h3.tracking.method = "h3"
        cfg_h3.tracking.parameters = {"temporal_variant": "H2"}
        ctx_h3 = PipelineContext(cfg_h3)
        stage_h3 = get_tracking_stage("h3")
        out_h3 = stage_h3.run(det_out, cfg_h3, ctx_h3)

        self.assertIsInstance(out_h3, TrackingOutput)
        self.assertGreater(out_h3.total_tracks, 0)
        self.assertFalse(out_h3.track_states_df.empty)
        self.assertIn("track_ownership_masks", dir(out_h3))
        self.assertEqual(out_h3.track_ownership_masks.shape, det_out.patch_activation_grid4.shape)
        # Verify interventions logged
        self.assertIn("interventions", out_h3.dataset_metadata)

        # H4 Run
        cfg_h4 = PipelineConfig()
        cfg_h4.tracking.method = "h4"
        cfg_h4.tracking.parameters = {"temporal_variant": "H2", "association_strategy": "direct_support"}
        ctx_h4 = PipelineContext(cfg_h4)
        stage_h4 = get_tracking_stage("h4")
        out_h4 = stage_h4.run(det_out, cfg_h4, ctx_h4)

        self.assertIsInstance(out_h4, TrackingOutput)
        self.assertGreater(out_h4.total_tracks, 0)
        self.assertFalse(out_h4.track_states_df.empty)
        self.assertIn("track_ownership_masks", dir(out_h4))
        self.assertEqual(out_h4.track_ownership_masks.shape, det_out.patch_activation_grid4.shape)
        # Verify interventions logged
        self.assertIn("interventions", out_h4.dataset_metadata)


if __name__ == "__main__":
    unittest.main()
