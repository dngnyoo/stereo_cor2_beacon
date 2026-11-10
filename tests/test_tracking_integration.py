"""End-to-End Pipeline Integration & Numerical Regression Suite for E1, A3, A4, A5b."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline
from cme_pipeline.stages.segmentation.track_masks import build_track_level_masks
from cme_pipeline.stages.tracking.registry import (
    list_tracking_methods,
    resolve_and_validate_tracking_parameters,
)


class TestTrackingIntegrationAndRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.tmp_dir_obj = tempfile.TemporaryDirectory()
        cls.tmp_dir = cls.tmp_dir_obj.name

        def _run_cfg(yaml_rel: str, run_tag: str):
            cfg = PipelineConfig.from_yaml(cls.repo_root / yaml_rel)
            cfg.output_base_dir = cls.tmp_dir
            cfg.run_name = run_tag
            cfg.force_rerun = True
            pipe = Pipeline(cfg)
            return pipe.run(), pipe

        cls.res_e1, cls.pipe_e1 = _run_cfg("configs/tracking_e1_rule_b.yaml", "reg_e1_rule_b")
        cls.res_a3, cls.pipe_a3 = _run_cfg("configs/tracking_a3_fixed_identity.yaml", "reg_a3_identity")
        cls.res_a4, cls.pipe_a4 = _run_cfg("configs/tracking_a4_identity.yaml", "reg_a4_identity")
        cls.res_a5b_id, cls.pipe_a5b_id = _run_cfg("configs/tracking_a5b_identity.yaml", "reg_a5b_identity")
        cls.res_a5b_rb, cls.pipe_a5b_rb = _run_cfg("configs/tracking_a5b_rule_b.yaml", "reg_a5b_rule_b")

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir_obj.cleanup()

    def test_01_registry_discovery(self):
        methods = list_tracking_methods()
        for m in ("e1", "a3", "a4", "a5b"):
            self.assertIn(m, methods)

    def test_02_strict_parameter_validation(self):
        # Unsupported key in tracking.parameters must raise ValueError on resolution/Pipeline init
        cfg_unknown = PipelineConfig.from_dict({
            "tracking": {
                "method": "a5b",
                "parameters": {"nonexistent_param": 123},
            }
        })
        with self.assertRaises(ValueError):
            resolve_and_validate_tracking_parameters("a5b", cfg_unknown.tracking)

        # a5b-specific param on a4 must raise ValueError
        cfg_bad_a4 = PipelineConfig.from_dict({
            "tracking": {
                "method": "a4",
                "parameters": {"min_pa_width_deg": 8.0},
            }
        })
        with self.assertRaises(ValueError):
            Pipeline(cfg_bad_a4)

        # Conflicting top-level vs nested parameters must raise ValueError
        with self.assertRaises(ValueError):
            PipelineConfig.from_dict({
                "tracking": {
                    "method": "a5b",
                    "min_pa_width_deg": 10.0,
                    "parameters": {"min_pa_width_deg": 8.0},
                }
            })

    def test_03_tracking_parameters_actually_applied(self):
        """Verify changing tracking.parameters alters runtime tracking behavior and manifest provenance."""
        cfg_mod = PipelineConfig.from_yaml(self.repo_root / "configs/tracking_a5b_identity.yaml")
        cfg_mod.output_base_dir = self.tmp_dir
        cfg_mod.run_name = "reg_a5b_param_override"
        cfg_mod.tracking.parameters["min_pa_width_deg"] = 40.0
        cfg_mod.tracking.min_pa_width_deg = 40.0
        cfg_mod.force_rerun = True

        res_mod = Pipeline(cfg_mod).run()
        t_mod = res_mod["stage_outputs"]["tracking"]
        t_ref = self.res_a5b_id["stage_outputs"]["tracking"]

        # Changing min_pa_width_deg from 8.0 to 40.0 alters contested Voronoi boundaries
        self.assertFalse(np.array_equal(t_mod.track_ownership_masks, t_ref.track_ownership_masks))

        manifest = json.loads((self.tmp_dir_path() / "reg_a5b_param_override/manifest.json").read_text())
        self.assertEqual(manifest["provenance"]["effective_tracking_parameters"]["min_pa_width_deg"], 40.0)

    def tmp_dir_path(self) -> Path:
        return Path(self.tmp_dir)

    def test_04_identical_detection_inputs_across_all_methods(self):
        t_e1 = self.res_e1["stage_outputs"]["tracking"]
        t_a3 = self.res_a3["stage_outputs"]["tracking"]
        t_a4 = self.res_a4["stage_outputs"]["tracking"]
        t_a5b = self.res_a5b_id["stage_outputs"]["tracking"]

        for other in (t_a3, t_a4, t_a5b):
            self.assertTrue(np.array_equal(t_e1.patch_activation_grid4, other.patch_activation_grid4))
            self.assertTrue(np.array_equal(t_e1.patch_component_labels, other.patch_component_labels))

    def test_05_semantic_separation_and_predicted_zero_leakage(self):
        for tag, res in [
            ("e1", self.res_e1),
            ("a3", self.res_a3),
            ("a4", self.res_a4),
            ("a5b", self.res_a5b_id),
        ]:
            t_out = res["stage_outputs"]["tracking"]
            own = t_out.track_ownership_masks
            comp = t_out.patch_component_labels
            self.assertIsNotNone(own, f"{tag}: missing track_ownership_masks")
            # Component labels and Track IDs must be distinct representations
            self.assertFalse(
                np.array_equal(own, comp),
                f"{tag}: track_ownership_masks and patch_component_labels are identical/conflated!",
            )
            # Zero ownership outside active grid4 and unique ownership per patch
            self.assertTrue(np.all(own[~t_out.patch_activation_grid4] == 0))
            # Zero patches for PREDICTED_ONLY states
            df = t_out.track_states_df
            man = t_out.manifest_df
            for _, r in df[df["state"] == "PREDICTED_ONLY"].iterrows():
                f_idx = man.index[man["pair_index"] == int(r["pair_index"])][0]
                self.assertEqual(int((own[f_idx] == int(r["track_id"])).sum()), 0)

    def test_06_e1_rule_b_regression(self):
        val_out = self.res_e1["final_output"]
        seg_out = val_out.segmentation_output
        size_out = seg_out.size_gate_output
        track_out = self.res_e1["stage_outputs"]["tracking"]

        self.assertEqual(track_out.total_tracks, 219)
        self.assertEqual(size_out.total_recovered_patches, 5174)
        self.assertEqual(seg_out.total_segmented_pixels, 50665)

        exp07_e1 = self.repo_root / "runs/07_a5b_end_to_end_segmentation/baseline_e1/arrays"
        hist_rec = np.load(exp07_e1 / "recovered_patch_masks.npy")
        hist_seg = np.load(exp07_e1 / "final_pixel_segmentation_union.npy")
        self.assertTrue(np.array_equal(size_out.recovered_patch_masks, hist_rec))
        self.assertTrue(np.array_equal(seg_out.final_pixel_segmentation_union, hist_seg))

    def test_07_a3_fixed_regression(self):
        track_out = self.res_a3["stage_outputs"]["tracking"]
        self.assertEqual(track_out.total_tracks, 50)
        ref_a3 = np.load(self.repo_root / "runs/pilot_adaptive_multicomponent_a4/arrays/a3_fixed_ownership_masks.npy")
        self.assertTrue(np.array_equal(track_out.track_ownership_masks, ref_a3))

    def test_08_a4_regression_and_multicomponent(self):
        track_out = self.res_a4["stage_outputs"]["tracking"]
        self.assertEqual(track_out.total_tracks, 18)
        ref_a4 = np.load(self.repo_root / "runs/pilot_adaptive_multicomponent_a4/arrays/a4_ownership_masks.npy")
        self.assertTrue(np.array_equal(track_out.track_ownership_masks, ref_a4))
        t3_p34 = track_out.track_states_df[
            (track_out.track_states_df["track_id"] == 3)
            & (track_out.track_states_df["pair_index"] == 34)
        ].iloc[0]
        self.assertGreaterEqual(int(t3_p34["num_components"]), 2)

    def test_09_a5b_identity_regression_and_track_masks(self):
        val_out = self.res_a5b_id["final_output"]
        seg_out = val_out.segmentation_output
        size_out = seg_out.size_gate_output
        track_out = self.res_a5b_id["stage_outputs"]["tracking"]

        self.assertEqual(track_out.total_tracks, 20)
        self.assertEqual(size_out.total_recovered_patches, 6549)
        self.assertEqual(seg_out.total_segmented_pixels, 59899)

        exp07_b2 = self.repo_root / "runs/07_a5b_end_to_end_segmentation/experimental_a5b_identity/arrays"
        hist_rec = np.load(exp07_b2 / "recovered_patch_masks.npy")
        hist_seg = np.load(exp07_b2 / "final_pixel_segmentation_union.npy")
        self.assertTrue(np.array_equal(size_out.recovered_patch_masks, hist_rec))
        self.assertTrue(np.array_equal(seg_out.final_pixel_segmentation_union, hist_seg))

        # Per-track extraction & 7,285 shared pixel overlap preservation check
        tl = build_track_level_masks(seg_out)
        self.assertTrue(tl["exact_pixel_union_match"])
        self.assertTrue(tl["exact_patch_union_match"])
        self.assertEqual(tl["union_pixel_sum"], 59899)
        self.assertEqual(tl["individual_pixel_sum"], 67184)
        self.assertEqual(tl["shared_pixel_locations"], 7285)
        self.assertEqual(tl["shared_overlap_difference"], 7285)

        # Catalog scores match Exp 07
        cat_map = {r["Reference"]: float(r["Angular IoU"]) for _, r in val_out.comparison_metrics_df.iterrows()}
        self.assertAlmostEqual(cat_map["SEEDS"], 0.5501, places=4)
        self.assertAlmostEqual(cat_map["CACTus (CME 0024)"], 0.8105, places=4)
        self.assertAlmostEqual(cat_map["MVCC / CORSET"], 0.4826, places=4)

    def test_10_a5b_rule_b_vs_identity_bitwise_equal(self):
        seg_id = self.res_a5b_id["final_output"].segmentation_output
        seg_rb = self.res_a5b_rb["final_output"].segmentation_output
        rec_id = seg_id.size_gate_output.recovered_patch_masks
        rec_rb = seg_rb.size_gate_output.recovered_patch_masks

        self.assertTrue(np.array_equal(rec_id, rec_rb))
        self.assertTrue(np.array_equal(seg_id.final_pixel_segmentation_union, seg_rb.final_pixel_segmentation_union))

    def test_11_provenance_recorded_in_manifest(self):
        manifest = json.loads((self.tmp_dir_path() / "reg_a5b_identity/manifest.json").read_text())
        prov = manifest["provenance"]
        self.assertEqual(prov["tracking_algorithm"], "a5b")
        self.assertEqual(prov["association_algorithm"], "identity")
        self.assertEqual(prov["effective_tracking_parameters"]["min_pa_width_deg"], 8.0)
        self.assertEqual(prov["effective_tracking_parameters"]["outward_gate_enabled"], True)
        self.assertEqual(prov["effective_tracking_parameters"]["outward_gate_tolerance_px"], 6.0)
        self.assertEqual(prov["effective_tracking_parameters"]["ema_alpha"], 0.1)
        self.assertIn("git_commit", prov["code_revision"])
        self.assertEqual(prov["input_dataset"]["n_pairs"], 32)


if __name__ == "__main__":
    unittest.main()
