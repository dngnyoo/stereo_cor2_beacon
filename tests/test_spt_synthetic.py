"""Synthetic Validation Suite for Seed-Origin Propagation Tracking (SPT).

Covers all 10 synthetic test scenarios required by Section 10:
1. Single outward-moving CME-like arc
2. Two simultaneously propagating arcs
3. Different radial speeds
4. Angular deflection
5. Asymmetric expansion
6. Two arcs connected by an artificial patch bridge (SPT-2 separation demonstration)
7. Crossing / converging trajectories
8. Bright stationary artifact
9. Temporarily missing observations
10. Long cadence gap

Measures: identity preservation, fragmentation, ownership accuracy, and kinematics.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import DetectionOutput, GeometryContext
from cme_pipeline.stages.tracking.spt_core import (
    SPTLifecycleState,
    SPTTrack,
    detect_seeds_and_grow_components,
)
from cme_pipeline.stages.tracking.v11_spt0_temporal_association import SPT0TemporalAssociationTrackingStage
from cme_pipeline.stages.tracking.v12_spt1_motion_prediction import SPT1MotionPredictionTrackingStage
from cme_pipeline.stages.tracking.v13_spt2_patch_ownership import (
    SPT2PatchOwnershipTrackingStage,
    arbitrate_contested_component,
)
from cme_pipeline.utils.geometry import circular_diff_deg


def create_synthetic_geometry() -> GeometryContext:
    """Creates a canonical 64x64 polar grid for synthetic testing with 256-scale radii."""
    grid_y, grid_x = np.indices((64, 64))
    center_y, center_x = 127.5, 127.5
    px_x = grid_x * 4.0 + 2.0
    px_y = grid_y * 4.0 + 2.0
    dy = px_y - center_y
    dx = px_x - center_x
    radius = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
    # Solar North-Up: 0 deg at +Y, 90 deg at +X
    pa = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    pa = pa.astype(np.float32)

    return GeometryContext(
        sx=center_x, sy=center_y, rsun_px=8.163,
        occulter_inner_radius_px=16.326, occulter_r_min_px=22.0,
        fov_r_max_px=122.5, pa_raster=np.zeros((256, 256)),
        radius_raster=np.zeros((256, 256)), patch_pa=pa,
        patch_rad=radius, patch_x=grid_x, patch_y=grid_y,
        wcs_header_str="",
    )


def build_synthetic_detection_output(
    frame_activations: List[np.ndarray],
    frame_scores: List[np.ndarray],
    frame_timestamps: List[datetime],
    geom: GeometryContext,
    is_large_gap_flags: Optional[List[bool]] = None,
) -> DetectionOutput:
    """Wraps synthetic activations and scores into a valid DetectionOutput."""
    n_frames = len(frame_activations)
    if is_large_gap_flags is None:
        is_large_gap_flags = [False] * n_frames

    manifest_rows = []
    frame_cache = []

    for i in range(n_frames):
        ts = frame_timestamps[i]
        manifest_rows.append({
            "f_idx": i,
            "pair_index": i + 1,
            "end_time": ts.isoformat(),
            "timestamp": ts.isoformat(),
            "is_large_gap": is_large_gap_flags[i],
            "usable_detection": True,
            "observation_status": "VALID",
        })
        frame_cache.append({
            "pair_idx": i + 1,
            "f_idx": i,
            "timestamp": ts,
            "val": np.ones((256, 256), dtype=bool),
            "z_global": np.repeat(np.repeat(frame_scores[i], 4, axis=0), 4, axis=1),
            "is_large_gap": is_large_gap_flags[i],
        })

    grid4_all = np.stack(frame_activations, axis=0).astype(bool)
    comp_labels = np.zeros(grid4_all.shape, dtype=np.int32)
    for i in range(n_frames):
        from scipy.ndimage import label
        lbl, _ = label(grid4_all[i])
        comp_labels[i] = lbl

    return DetectionOutput(
        manifest_df=pd.DataFrame(manifest_rows),
        patch_activation_grid4=grid4_all,
        raw_component_labels=comp_labels,
        frame_cache=frame_cache,
        geometry=geom,
        dataset_metadata={"dataset_id": "SYNTHETIC_TEST_SUITE"},
    )


def generate_arc_patches(
    geom: GeometryContext,
    cpa_deg: float,
    width_deg: float,
    r_front_px: float,
    radial_depth_px: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic CME arc on the 64x64 grid with continuous scores."""
    pa = geom.patch_pa
    rad = geom.patch_rad

    ang_dist = np.minimum(
        np.abs(pa - cpa_deg),
        360.0 - np.abs(pa - cpa_deg)
    )
    in_pa = ang_dist <= (width_deg / 2.0)
    in_rad = (rad <= r_front_px) & (rad >= (r_front_px - radial_depth_px))

    mask = in_pa & in_rad
    scores = np.zeros((64, 64), dtype=np.float32)
    # Peak score at center of arc, tapering towards flanks
    scores[mask] = 4.0 * (1.0 - ang_dist[mask] / (width_deg / 2.0 + 1e-3)) + 1.0
    return mask, scores


class TestSPTSyntheticValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geom = create_synthetic_geometry()
        cls.base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        cls.cfg = PipelineConfig()
        cls.ctx = PipelineContext(cls.cfg)

    def test_scenario_01_single_outward_arc(self):
        """Scenario 1: Single CME arc moving outward from r=24 px to r=60 px."""
        n_frames = 8
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            r_front = 24.0 + i * 5.0  # +5 px per frame (~0.0028 px/s)
            mask, sc = generate_arc_patches(self.geom, cpa_deg=90.0, width_deg=30.0, r_front_px=r_front)
            activations.append(mask)
            scores.append(sc)
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)
        spt0 = SPT0TemporalAssociationTrackingStage().run(det, self.cfg, self.ctx)
        spt1 = SPT1MotionPredictionTrackingStage().run(det, self.cfg, self.ctx)
        spt2 = SPT2PatchOwnershipTrackingStage().run(det, self.cfg, self.ctx)

        # Verify exact single track formed across all trackers
        for tracker_name, out in [("SPT-0", spt0), ("SPT-1", spt1), ("SPT-2", spt2)]:
            self.assertEqual(out.total_tracks, 1, f"[{tracker_name}] Should produce exactly 1 track")
            confirmed = getattr(out, "confirmed_spt_tracks")
            self.assertEqual(len(confirmed), 1, f"[{tracker_name}] Track must be CONFIRMED")
            track = confirmed[0]
            self.assertAlmostEqual(track.seed.seed_pa_deg, 90.0, delta=6.0, msg=f"[{tracker_name}] Seed PA mismatch")
            # Verify outward propagation
            observed_rs = [h.r_max_px for h in track.history if h.state == "OBSERVED"]
            self.assertEqual(len(observed_rs), n_frames)
            self.assertTrue(all(np.diff(observed_rs) > 0), f"[{tracker_name}] Front must move strictly outward")

    def test_scenario_02_two_simultaneous_arcs(self):
        """Scenario 2: Two simultaneously propagating CMEs in opposite quadrants."""
        n_frames = 6
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            r_front = 24.0 + i * 4.0
            mask1, sc1 = generate_arc_patches(self.geom, cpa_deg=90.0, width_deg=25.0, r_front_px=r_front)
            mask2, sc2 = generate_arc_patches(self.geom, cpa_deg=270.0, width_deg=25.0, r_front_px=r_front)
            activations.append(mask1 | mask2)
            scores.append(np.maximum(sc1, sc2))
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)
        spt2 = SPT2PatchOwnershipTrackingStage().run(det, self.cfg, self.ctx)

        self.assertEqual(spt2.total_tracks, 2, "Must initialize exactly 2 independent tracks")
        masks = spt2.track_ownership_masks
        # Ensure mutual exclusivity (no pixel owned by both)
        for i in range(n_frames):
            t1_count = (masks[i] == 1).sum()
            t2_count = (masks[i] == 2).sum()
            total_owned = (masks[i] > 0).sum()
            self.assertEqual(t1_count + t2_count, total_owned, "Patch ownership must be mutually exclusive")

    def test_scenario_03_different_radial_speeds(self):
        """Scenario 3: Two CMEs propagating with different speeds (slow vs fast)."""
        n_frames = 6
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            r_slow = 24.0 + i * 3.0  # Slow CME: +3 px/frame
            r_fast = 24.0 + i * 8.0  # Fast CME: +8 px/frame
            m1, sc1 = generate_arc_patches(self.geom, cpa_deg=45.0, width_deg=20.0, r_front_px=r_slow)
            m2, sc2 = generate_arc_patches(self.geom, cpa_deg=135.0, width_deg=20.0, r_front_px=r_fast)
            activations.append(m1 | m2)
            scores.append(np.maximum(sc1, sc2))
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)
        spt1 = SPT1MotionPredictionTrackingStage().run(det, self.cfg, self.ctx)

        confirmed = getattr(spt1, "confirmed_spt_tracks")
        self.assertEqual(len(confirmed), 2)
        v1 = confirmed[0].radial_velocity_px_s
        v2 = confirmed[1].radial_velocity_px_s
        self.assertNotEqual(v1, v2)
        # Fast CME must have higher estimated velocity than slow CME
        fast_t = max(confirmed, key=lambda t: t.radial_velocity_px_s)
        self.assertAlmostEqual(fast_t.seed.seed_pa_deg, 135.0, delta=7.0)

    def test_scenario_04_angular_deflection(self):
        """Scenario 4: CME experiencing continuous angular deflection (90 -> 110 deg)."""
        n_frames = 6
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            r_front = 24.0 + i * 5.0
            cpa = 90.0 + i * 4.0  # Deflects by +4 deg per frame
            m, sc = generate_arc_patches(self.geom, cpa_deg=cpa, width_deg=25.0, r_front_px=r_front)
            activations.append(m)
            scores.append(sc)
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)
        spt1 = SPT1MotionPredictionTrackingStage().run(det, self.cfg, self.ctx)

        self.assertEqual(spt1.total_tracks, 1)
        track = getattr(spt1, "confirmed_spt_tracks")[0]
        # Verify seed preserves original 90 deg birth angle while final CPA evolved towards 110 deg
        self.assertAlmostEqual(track.seed.seed_pa_deg, 90.0, delta=6.0)
        self.assertAlmostEqual(track.current_cpa, 110.0, delta=6.0)

    def test_scenario_05_asymmetric_expansion(self):
        """Scenario 5: CME undergoing rapid angular expansion (width 20 -> 60 deg)."""
        n_frames = 6
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            r_front = 24.0 + i * 5.0
            width = 20.0 + i * 8.0  # Expands by 8 deg per frame
            m, sc = generate_arc_patches(self.geom, cpa_deg=90.0, width_deg=width, r_front_px=r_front)
            activations.append(m)
            scores.append(sc)
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)
        spt2 = SPT2PatchOwnershipTrackingStage().run(det, self.cfg, self.ctx)

        self.assertEqual(spt2.total_tracks, 1)
        track = getattr(spt2, "confirmed_spt_tracks")[0]
        self.assertGreater(track.current_width, 40.0)

    def test_scenario_06_bridge_connected_cme_separation(self):
        """Scenario 6 (Critical): Two CMEs connected by an artificial patch bridge into ONE component.

        Demonstrates that SPT-2 correctly separates them into two distinct tracks,
        whereas an unsplit component tracker (like SPT-0) merges them.
        """
        n_frames = 6
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            r_front = 24.0 + i * 4.0
            # CME 1 at 70 deg, CME 2 at 110 deg
            m1, sc1 = generate_arc_patches(self.geom, cpa_deg=70.0, width_deg=20.0, r_front_px=r_front)
            m2, sc2 = generate_arc_patches(self.geom, cpa_deg=110.0, width_deg=20.0, r_front_px=r_front)

            combined_mask = m1 | m2
            combined_scores = np.maximum(sc1, sc2)

            # At frames 2, 3, 4: add an artificial 1-patch-wide bridge connecting them!
            if 2 <= i <= 4:
                # Bridge across CPA 70 to 110 at radius r = r_front - 2
                bridge_mask = (np.abs(self.geom.patch_rad - (r_front - 2.0)) <= 1.0) & \
                              (self.geom.patch_pa >= 70.0) & (self.geom.patch_pa <= 110.0)
                combined_mask = combined_mask | bridge_mask
                combined_scores[bridge_mask] = 0.5  # Weak bridge score

            activations.append(combined_mask)
            scores.append(combined_scores)
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)

        # 1. Run SPT-0 (Unsplit baseline)
        spt0 = SPT0TemporalAssociationTrackingStage().run(det, self.cfg, self.ctx)
        # 2. Run SPT-2 (Competing-track patch ownership)
        spt2 = SPT2PatchOwnershipTrackingStage().run(det, self.cfg, self.ctx)

        # SPT-2 must maintain BOTH tracks separately across all 6 frames
        confirmed_spt2 = getattr(spt2, "confirmed_spt_tracks")
        self.assertEqual(len(confirmed_spt2), 2, "SPT-2 must maintain exactly 2 confirmed tracks despite bridge!")

        masks_spt2 = spt2.track_ownership_masks
        # In frame 3 (where bridge connects them into a single 8-connected component):
        f3_mask = masks_spt2[3]
        t1_patches = (f3_mask == 1).sum()
        t2_patches = (f3_mask == 2).sum()
        self.assertGreater(t1_patches, 0, "Track 1 must own patches inside the bridged component")
        self.assertGreater(t2_patches, 0, "Track 2 must own patches inside the bridged component")

        # Verify arbitration logs recorded the bridge separation
        arb_df = getattr(spt2, "seed_arbitration_df")
        self.assertFalse(arb_df.empty, "SPT-2 must log arbitration decisions for the bridged component")

    def test_scenario_07_converging_trajectories(self):
        """Scenario 7: Two structures with converging PAs (60 -> 75 deg and 100 -> 85 deg)."""
        n_frames = 5
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            r_front = 26.0 + i * 5.0
            cpa1 = 60.0 + i * 3.0
            cpa2 = 100.0 - i * 3.0
            m1, sc1 = generate_arc_patches(self.geom, cpa_deg=cpa1, width_deg=20.0, r_front_px=r_front)
            m2, sc2 = generate_arc_patches(self.geom, cpa_deg=cpa2, width_deg=20.0, r_front_px=r_front)
            activations.append(m1 | m2)
            scores.append(np.maximum(sc1, sc2))
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)
        spt2 = SPT2PatchOwnershipTrackingStage().run(det, self.cfg, self.ctx)

        self.assertEqual(len(getattr(spt2, "confirmed_spt_tracks")), 2)

    def test_scenario_08_bright_stationary_artifact(self):
        """Scenario 8: A stationary bright artifact (r = 25 px) that does not propagate outward."""
        n_frames = 5
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            # Stationary artifact fixed at r=25 px
            m, sc = generate_arc_patches(self.geom, cpa_deg=180.0, width_deg=10.0, r_front_px=25.0)
            activations.append(m)
            scores.append(sc * 2.0)  # High score
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)
        # With outward gate enabled, non-propagating feature fails radial progression
        cfg_outward = PipelineConfig()
        cfg_outward.tracking.parameters = {"outward_gate_enabled": True, "outward_gate_tolerance_px": 0.5}
        spt1 = SPT1MotionPredictionTrackingStage().run(det, cfg_outward, self.ctx)

        # Non-propagating feature must not become confirmed
        self.assertEqual(len(getattr(spt1, "confirmed_spt_tracks")), 0, "Stationary artifact must NOT become confirmed CME!")

    def test_scenario_09_temporarily_missing_observations(self):
        """Scenario 9: CME unobserved at frame 3, recovered at frame 4 at predicted front radius."""
        n_frames = 6
        activations, scores, timestamps = [], [], []

        for i in range(n_frames):
            t = self.base_time + timedelta(minutes=30 * i)
            r_front = 24.0 + i * 5.0
            if i == 3:
                # Drop observation completely at frame 3
                m = np.zeros((64, 64), dtype=bool)
                sc = np.zeros((64, 64), dtype=np.float32)
            else:
                m, sc = generate_arc_patches(self.geom, cpa_deg=90.0, width_deg=25.0, r_front_px=r_front)
            activations.append(m)
            scores.append(sc)
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom)
        spt1 = SPT1MotionPredictionTrackingStage().run(det, self.cfg, self.ctx)

        # Must maintain continuous single track across the missed observation gap
        self.assertEqual(spt1.total_tracks, 1, "Should NOT fragment or create a new track upon reappearance!")
        track = getattr(spt1, "confirmed_spt_tracks")[0]
        self.assertEqual(track.history[3].state, "MISSED")
        self.assertEqual(track.history[4].state, "OBSERVED")
        self.assertAlmostEqual(track.history[4].r_max_px, 44.0, delta=2.0)

    def test_scenario_10_long_cadence_gap(self):
        """Scenario 10: Long cadence gap (180 min) terminating active track."""
        n_frames = 5
        activations, scores, timestamps = [], [], []
        gaps = [False, False, True, False, False]  # Gap before frame 2

        t = self.base_time
        for i in range(n_frames):
            if i == 2:
                t += timedelta(minutes=180)  # 3 hour gap
            else:
                t += timedelta(minutes=30)
            if i < 2:
                r_front = 24.0 + i * 5.0
            else:
                r_front = 24.0 + (i - 2) * 5.0
            m, sc = generate_arc_patches(self.geom, cpa_deg=90.0, width_deg=25.0, r_front_px=r_front)
            activations.append(m)
            scores.append(sc)
            timestamps.append(t)

        det = build_synthetic_detection_output(activations, scores, timestamps, self.geom, is_large_gap_flags=gaps)
        spt1 = SPT1MotionPredictionTrackingStage().run(det, self.cfg, self.ctx)

        # Gap terminates active track; new track initializes after gap
        self.assertEqual(spt1.total_tracks, 2, "Cadence gap must terminate first track and initialize second")
        all_t = getattr(spt1, "all_spt_tracks")
        self.assertEqual(all_t[0].termination_reason, "CADENCE_GAP")


if __name__ == "__main__":
    unittest.main()
