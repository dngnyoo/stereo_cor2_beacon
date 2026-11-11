"""Independent CME catalog tests with synthetic masks (no external catalogs)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cme_pipeline.stages.segmentation.cme_catalog import (
    build_cme_catalog, save_cme_catalog,
)


def example_segmentation(n=4):
    pa = np.array([[359., 1., 70.], [359., 1., 70.], [359., 1., 70.]])
    radius = np.array([[10., 10., 10.], [20., 20., 20.], [30., 30., 30.]])
    masks = np.zeros((n, 3, 3), dtype=bool)
    # 3 observations, expanding radial front; fourth frame empty
    masks[0, 0, :2] = True
    masks[1, :2, :2] = True
    masks[2, :, :2] = True
    manifest = pd.DataFrame({
        "pair_index": [1, 2, 3, 4],
        "end_time": [
            "2010-12-12T00:00:00", "2010-12-12T00:30:00",
            "2010-12-12T01:00:00", "2010-12-12T01:30:00",
        ],
    })
    geom = SimpleNamespace(pa_raster=pa, radius_raster=radius, rsun_px=10.0)
    track = SimpleNamespace(manifest_df=manifest, geometry=geom)
    assoc = SimpleNamespace(tracking_output=track)
    birth = SimpleNamespace(association_output=assoc)
    size = SimpleNamespace(
        birth_gate_output=birth,
        admitted_family_ids=["FAM_001"],
        family_summary_df=pd.DataFrame([
            {"family_id": "FAM_001", "member_track_ids": "[7, 8]"}
        ]),
    )
    return SimpleNamespace(size_gate_output=size, family_pixel_masks={"FAM_001": masks})


class TestIndependentCMECatalog(unittest.TestCase):
    def test_event_counts_and_angular_wrap(self):
        events, frames = build_cme_catalog(example_segmentation())
        self.assertEqual(len(events), 1)
        self.assertEqual(len(frames), 3)
        event = events.iloc[0]
        self.assertEqual(event["n_observed_frames"], 3)
        self.assertEqual(event["total_mask_pixels"], 12)
        self.assertEqual(event["peak_mask_pixels"], 6)
        self.assertEqual(event["peak_pair_index"], 3)
        self.assertAlmostEqual(event["angular_width_deg"], 2.0, places=4)
        self.assertAlmostEqual(event["cpa_deg"], 0.0, places=4)
        self.assertAlmostEqual(event["duration_min"], 60.0)
        self.assertEqual(event["member_track_ids"], "[7, 8]")
        self.assertEqual(event["speed_quality"], "EXPLORATORY")
        self.assertGreater(event["plane_of_sky_speed_km_s"], 0)

    def test_empty_mask_has_no_false_event(self):
        seg = example_segmentation()
        seg.family_pixel_masks["FAM_001"][:] = False
        events, frames = build_cme_catalog(seg)
        self.assertTrue(events.empty)
        self.assertTrue(frames.empty)
        self.assertIn("event_id", events.columns)

    def test_saved_csvs(self):
        with tempfile.TemporaryDirectory() as temp:
            event_path, frame_path = save_cme_catalog(example_segmentation(), Path(temp))
            self.assertTrue(event_path.is_file())
            self.assertTrue(frame_path.is_file())
            self.assertEqual(len(pd.read_csv(event_path)), 1)

    def test_products_stage_writes_diagnostic_outputs(self):
        from types import SimpleNamespace
        from cme_pipeline.stages.segmentation.cme_products import CMEProductsStage
        from cme_pipeline.core.config import PipelineConfig

        seg = example_segmentation()
        # The diagnostic renderer requires frame-cache RD and validity arrays.
        track = seg.size_gate_output.birth_gate_output.association_output.tracking_output
        track.frame_cache = [
            {"rd": np.zeros((3, 3)), "val": np.ones((3, 3), dtype=bool)}
            for _ in range(4)
        ]
        seg.final_pixel_segmentation_union = seg.family_pixel_masks["FAM_001"].copy()
        seg.total_segmented_pixels = int(seg.final_pixel_segmentation_union.sum())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = SimpleNamespace(
                tables_dir=root / "tables",
                visualizations_dir=root / "visualizations",
                info=lambda message: None,
            )
            cfg = PipelineConfig()
            stage = CMEProductsStage()
            product = stage.run(seg, cfg, context)
            self.assertTrue(stage.outputs_exist(product))
            self.assertTrue((context.tables_dir / "cme_event_catalog.csv").is_file())
            self.assertTrue((context.visualizations_dir / "cme_animation.gif").is_file())
            self.assertEqual(len(list((context.visualizations_dir / "segmentation_overlays").glob("*.png"))), 4)
            self.assertTrue(list((context.visualizations_dir / "height_time").glob("*.png")))
            (context.tables_dir / "cme_event_catalog.csv").unlink()
            self.assertFalse(stage.outputs_exist(product))

    def test_eight_stage_order(self):
        from cme_pipeline.core.config import PipelineConfig
        from cme_pipeline.core.pipeline import resolve_stages
        names = [stage.name for stage in resolve_stages(PipelineConfig())]
        self.assertEqual(names[-3:], ["segmentation", "products", "validation"])


    def test_single_frame_speed_is_not_invented(self):
        seg = example_segmentation()
        seg.family_pixel_masks["FAM_001"][1:] = False
        events, _ = build_cme_catalog(seg)
        self.assertTrue(np.isnan(events.iloc[0]["plane_of_sky_speed_km_s"]))
        self.assertEqual(events.iloc[0]["speed_quality"], "INSUFFICIENT_FRAMES")


if __name__ == "__main__":
    unittest.main()
