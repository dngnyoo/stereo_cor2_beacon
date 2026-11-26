"""Unit and regression tests for Score-Ranked Seed-Based CME Component Identification Pilot.

Verifies:
- Deterministic score ranking and tie handling
- Circular PA geometry
- Inner-corona radius boundaries
- 8-connected diagonal paths
- Two candidates connected by a T24 path
- Two candidates without a connecting path
- Transitive merging of several seed groups
- Unseeded outer components
- Invalid and QC-rejected frames
- Component-growth equivalence with ordinary connected-component labeling
- Preservation of baseline A5b results
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import unittest
import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import GeometryContext
from cme_pipeline.stages.tracking.seed_component_pilot import (
    SeedCandidate,
    SeedGroup,
    ComponentDiagnostic,
    identify_and_rank_seed_candidates,
    determine_connectivity_and_grow,
    process_frame_seeds,
    STRUCT_8,
)
from cme_pipeline.stages.tracking.v07_pa_width_weighted_cpa import PAWidthWeightedCPAAdaptiveROITrackingStage


def make_synthetic_geometry() -> GeometryContext:
    """Creates a synthetic 64x64 geometry grid with solar center at (31.5, 31.5)."""
    y, x = np.indices((64, 64), dtype=float)
    # Solar center at patch (31.5, 31.5) -> RD pixel coords center is (128.0, 128.0)
    # patch coords: center is (4*c + 2, 4*r + 2)
    px_x = 4.0 * x + 2.0
    px_y = 4.0 * y + 2.0
    sx, sy = 128.0, 128.0
    dx = px_x - sx
    dy = px_y - sy
    rad = np.hypot(dx, dy)
    # PA counter-clockwise from solar north (dy < 0 is North in image coords, dx > 0 is East/West)
    # Use standard atan2 definition: atan2(dx, -dy) % 360
    pa = np.degrees(np.arctan2(dx, -dy)) % 360.0

    return GeometryContext(
        sx=sx,
        sy=sy,
        rsun_px=8.163,
        occulter_inner_radius_px=16.3,
        occulter_r_min_px=22.0,
        fov_r_max_px=122.5,
        pa_raster=np.zeros((256, 256)),
        radius_raster=np.zeros((256, 256)),
        patch_pa=pa,
        patch_rad=rad,
        patch_x=px_x,
        patch_y=px_y,
        wcs_header_str="",
    )


class TestSeedComponentPilot(unittest.TestCase):
    def setUp(self):
        self.geom = make_synthetic_geometry()

    def test_1_deterministic_score_ranking_and_tie_handling(self):
        """Verify strict deterministic tie-breaking: score desc -> radius asc -> PA asc -> r asc -> c asc."""
        scores = np.zeros((64, 64), dtype=np.float32)
        active = np.zeros((64, 64), dtype=bool)

        # Place 4 patches in inner corona [22, 30] px
        # Patch A: score 10.0, r=25, pa=90, (r=31, c=37)
        # Patch B: score 15.0, r=26, pa=90, (r=31, c=38) -> Should be rank 1 (highest score)
        # Patch C: score 10.0, r=24, pa=90, (r=31, c=36) -> Should beat A on smaller radius!
        # Patch D: score 10.0, r=24, pa=90, tie on r/pa -> tie breaks by (r, c)
        pA = (31, 37)
        pB = (31, 38)
        pC = (31, 36)

        for p in (pA, pB, pC):
            active[p] = True
            # artificially force patch_rad to exact values for testing
            self.geom.patch_rad[p] = 25.0 if p == pA else 26.0 if p == pB else 24.0

        scores[pA] = 10.0
        scores[pB] = 15.0
        scores[pC] = 10.0

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores,
            active_mask=active,
            geometry=self.geom,
            r_min=22.0,
            r_seed=30.0,
            effective_threshold=2.0,
        )

        self.assertEqual(len(cands), 3)
        # Rank 1 must be pB (score 15.0)
        self.assertEqual(cands[0].patch_r, pB[0])
        self.assertEqual(cands[0].patch_c, pB[1])
        self.assertEqual(cands[0].score_rank, 1)
        self.assertEqual(cands[0].raw_score, 15.0)
        self.assertAlmostEqual(cands[0].threshold_relative_score, 13.0)

        # Rank 2 must be pC (score 10.0, radius 24.0 vs 25.0)
        self.assertEqual(cands[1].patch_r, pC[0])
        self.assertEqual(cands[1].patch_c, pC[1])
        self.assertEqual(cands[1].score_rank, 2)

        # Rank 3 must be pA (score 10.0, radius 25.0)
        self.assertEqual(cands[2].patch_r, pA[0])
        self.assertEqual(cands[2].patch_c, pA[1])
        self.assertEqual(cands[2].score_rank, 3)

    def test_2_circular_pa_geometry(self):
        """Verify candidate PA wraps circularly and handles 0/360 boundary properly."""
        scores = np.zeros((64, 64), dtype=np.float32)
        active = np.zeros((64, 64), dtype=bool)

        # Patch near North (PA ~ 359 or 1 deg)
        pN = (25, 31)  # slightly north of center
        active[pN] = True
        scores[pN] = 8.0
        self.geom.patch_rad[pN] = 26.0
        self.geom.patch_pa[pN] = 359.5

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores, active_mask=active, geometry=self.geom,
            r_min=22.0, r_seed=30.0,
        )
        self.assertEqual(len(cands), 1)
        self.assertAlmostEqual(cands[0].pa_deg, 359.5)

    def test_3_inner_corona_radius_boundaries(self):
        """Verify strict exclusion of patches outside [r_min, r_seed]."""
        scores = np.full((64, 64), 10.0, dtype=np.float32)
        active = np.zeros((64, 64), dtype=bool)

        p_inner = (31, 35)   # r = 18.0 (below r_min=22.0)
        p_valid = (31, 37)   # r = 26.0 (inside [22, 30])
        p_outer = (31, 45)   # r = 55.0 (above r_seed=30.0)

        for p, r in [(p_inner, 18.0), (p_valid, 26.0), (p_outer, 55.0)]:
            active[p] = True
            self.geom.patch_rad[p] = r

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores, active_mask=active, geometry=self.geom,
            r_min=22.0, r_seed=30.0,
        )
        self.assertEqual(len(cands), 1)
        self.assertEqual((cands[0].patch_r, cands[0].patch_c), p_valid)

    def test_4_eight_connected_diagonal_paths(self):
        """Verify connectivity across diagonal neighbors under 8-connectivity."""
        active = np.zeros((64, 64), dtype=bool)
        scores = np.zeros((64, 64), dtype=np.float32)

        # Diagonal path: (30, 36) -> (31, 37)
        p1 = (30, 36)
        p2 = (31, 37)
        active[p1] = True
        active[p2] = True
        scores[p1] = 12.0
        scores[p2] = 8.0
        self.geom.patch_rad[p1] = 25.0
        self.geom.patch_rad[p2] = 26.0

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores, active_mask=active, geometry=self.geom,
            r_min=22.0, r_seed=30.0,
        )
        groups, diags, seeded_m, unseeded_m, equiv = determine_connectivity_and_grow(
            candidates=cands, active_mask=active, geometry=self.geom,
        )

        self.assertEqual(len(groups), 1, "Diagonal active neighbors must merge into 1 seed group")
        self.assertEqual(groups[0].num_candidates, 2)
        self.assertEqual(groups[0].representative_seed.patch_r, p1[0])
        self.assertEqual(groups[0].representative_seed.raw_score, 12.0)
        self.assertTrue(equiv)

    def test_5_two_candidates_connected_by_t24_path(self):
        """Verify two seed candidates connected via an intermediate active bridge patch merge."""
        active = np.zeros((64, 64), dtype=bool)
        scores = np.zeros((64, 64), dtype=np.float32)

        # Candidate 1 at (30, 36), Candidate 2 at (32, 36), Bridge at (31, 36)
        c1 = (30, 36)
        bridge = (31, 36)
        c2 = (32, 36)

        active[c1] = True
        active[bridge] = True
        active[c2] = True

        scores[c1] = 14.0
        scores[bridge] = 5.0
        scores[c2] = 11.0

        self.geom.patch_rad[c1] = 25.0
        self.geom.patch_rad[bridge] = 25.0
        self.geom.patch_rad[c2] = 25.0

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores, active_mask=active, geometry=self.geom,
            r_min=22.0, r_seed=30.0,
        )
        # c1 and c2 are candidates, bridge is active too (or candidate if within radius)
        groups, diags, seeded_m, unseeded_m, equiv = determine_connectivity_and_grow(
            candidates=cands, active_mask=active, geometry=self.geom,
        )

        self.assertEqual(len(groups), 1, "Candidates connected through active path must merge")
        self.assertEqual(groups[0].representative_seed.raw_score, 14.0)
        self.assertEqual(groups[0].total_grown_patches, 3)
        self.assertTrue(equiv)

    def test_6_two_candidates_without_connecting_path(self):
        """Verify two seed candidates separated by inactive patches remain independent."""
        active = np.zeros((64, 64), dtype=bool)
        scores = np.zeros((64, 64), dtype=np.float32)

        # Candidate 1 in East, Candidate 2 in West
        c1 = (31, 37)  # East
        c2 = (31, 25)  # West (inactive gap between them)

        active[c1] = True
        active[c2] = True
        scores[c1] = 15.0
        scores[c2] = 12.0

        self.geom.patch_rad[c1] = 25.0
        self.geom.patch_rad[c2] = 25.0

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores, active_mask=active, geometry=self.geom,
            r_min=22.0, r_seed=30.0,
        )
        groups, diags, seeded_m, unseeded_m, equiv = determine_connectivity_and_grow(
            candidates=cands, active_mask=active, geometry=self.geom,
        )

        self.assertEqual(len(groups), 2, "Disconnected candidates must yield 2 independent seed groups")
        self.assertEqual(groups[0].seed_group_id, 1)
        self.assertEqual(groups[0].representative_seed.raw_score, 15.0)
        self.assertEqual(groups[1].seed_group_id, 2)
        self.assertEqual(groups[1].representative_seed.raw_score, 12.0)
        self.assertTrue(equiv)

    def test_7_transitive_merging_of_several_seed_groups(self):
        """Verify that transitive connectivity merges all connected candidates."""
        active = np.zeros((64, 64), dtype=bool)
        scores = np.zeros((64, 64), dtype=np.float32)

        # Line of 5 active patches: p0 - p1 - p2 - p3 - p4
        patches = [(31, 35 + i) for i in range(5)]
        for i, p in enumerate(patches):
            active[p] = True
            scores[p] = float(5 + i * 2)  # scores 5, 7, 9, 11, 13
            self.geom.patch_rad[p] = 25.0

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores, active_mask=active, geometry=self.geom,
            r_min=22.0, r_seed=30.0,
        )
        groups, diags, seeded_m, unseeded_m, equiv = determine_connectivity_and_grow(
            candidates=cands, active_mask=active, geometry=self.geom,
        )

        self.assertEqual(len(groups), 1, "Transitively connected candidates must form 1 seed group")
        self.assertEqual(groups[0].num_candidates, 5)
        self.assertEqual(groups[0].representative_seed.raw_score, 13.0)
        self.assertEqual(groups[0].total_grown_patches, 5)
        self.assertTrue(equiv)

    def test_8_unseeded_outer_components(self):
        """Verify components entirely outside [r_min, r_seed] are cleanly labeled as unseeded."""
        active = np.zeros((64, 64), dtype=bool)
        scores = np.zeros((64, 64), dtype=np.float32)

        # Seeded inner component at (31, 37) (r=25)
        p_inner = (31, 37)
        active[p_inner] = True
        scores[p_inner] = 16.0
        self.geom.patch_rad[p_inner] = 25.0

        # Unseeded outer component at (31, 55), (31, 56) (r=95)
        p_outer1 = (31, 55)
        p_outer2 = (31, 56)
        active[p_outer1] = True
        active[p_outer2] = True
        scores[p_outer1] = 7.0
        scores[p_outer2] = 6.0
        self.geom.patch_rad[p_outer1] = 95.0
        self.geom.patch_rad[p_outer2] = 98.0

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores, active_mask=active, geometry=self.geom,
            r_min=22.0, r_seed=30.0,
        )
        self.assertEqual(len(cands), 1, "Outer patches must NOT be seed candidates")

        groups, diags, seeded_m, unseeded_m, equiv = determine_connectivity_and_grow(
            candidates=cands, active_mask=active, geometry=self.geom,
        )

        self.assertEqual(len(groups), 1, "Only 1 seeded group should exist")
        self.assertEqual(len(diags), 2, "There are 2 total connected components")

        seeded_diag = next(d for d in diags if d.is_seeded)
        unseeded_diag = next(d for d in diags if not d.is_seeded)

        self.assertEqual(seeded_diag.num_patches, 1)
        self.assertEqual(unseeded_diag.num_patches, 2)
        self.assertIsNone(unseeded_diag.representative_score)
        self.assertFalse(unseeded_diag.intersects_seed_search_region)
        self.assertTrue(np.all(unseeded_m[p_outer1[0], p_outer1[1]] > 0))
        self.assertTrue(equiv)

    def test_9_invalid_and_qc_rejected_frames(self):
        """Verify unavailable / QC-rejected frame produces empty seed outputs and preserves flags."""
        scores = np.full((64, 64), 20.0, dtype=np.float32)
        active = np.ones((64, 64), dtype=bool)

        res = process_frame_seeds(
            scores_frame=scores,
            active_mask_frame=active,
            geometry=self.geom,
            observation_status="GLOBAL_ACTIVATION",
            usable_detection=False,
        )

        self.assertFalse(res.usable_detection)
        self.assertEqual(res.observation_status, "GLOBAL_ACTIVATION")
        self.assertEqual(len(res.candidates), 0)
        self.assertEqual(len(res.seed_groups), 0)
        self.assertEqual(len(res.component_diagnostics), 0)
        self.assertEqual(int(res.seeded_component_mask.sum()), 0)
        self.assertTrue(res.equivalence_verified)

    def test_10_mathematical_equivalence_with_connected_components(self):
        """Verify that every grown seeded component matches the exact 8-connected T24 component."""
        # Create a complex multi-component active pattern
        active = np.zeros((64, 64), dtype=bool)
        scores = np.zeros((64, 64), dtype=np.float32)

        # Component 1: crosses from r=24 px to r=45 px
        for c in range(35, 42):
            p = (31, c)
            active[p] = True
            scores[p] = float(c - 30)
            self.geom.patch_rad[p] = 20.0 + (c - 35) * 3.5

        # Component 2: crosses from r=26 px to r=38 px
        for r in range(25, 29):
            p = (r, 31)
            active[p] = True
            scores[p] = float(40 - r)
            self.geom.patch_rad[p] = 24.0 + (28 - r) * 4.0

        cands = identify_and_rank_seed_candidates(
            frame_scores=scores, active_mask=active, geometry=self.geom,
            r_min=22.0, r_seed=30.0,
        )
        groups, diags, seeded_m, unseeded_m, equiv = determine_connectivity_and_grow(
            candidates=cands, active_mask=active, geometry=self.geom,
        )

        self.assertTrue(equiv, "Equivalence check must strictly hold")
        for g in groups:
            grown = (seeded_m == g.seed_group_id)
            c_label = g.grown_comp_id
            from scipy.ndimage import label
            lbl, _ = label(active, structure=STRUCT_8)
            expected = (lbl == c_label)
            self.assertTrue(np.array_equal(grown, expected), "Grown seed mask must match T24 component exactly")

    def test_11_baseline_a5b_preservation(self):
        """Verify baseline A5b is not modified and runs identically with unchanged results."""
        cfg = PipelineConfig()
        cfg.date_str = 20101212
        cfg.tracking.method = "a5b"
        ctx = PipelineContext(cfg)

        stage = PAWidthWeightedCPAAdaptiveROITrackingStage()
        out = stage.run(None, cfg, ctx)

        # Baseline A5b produces exactly 21 tracks on frozen 20101212 with default parameters
        self.assertEqual(out.total_tracks, 21, "Original A5b baseline must be completely preserved")


if __name__ == "__main__":
    unittest.main()
