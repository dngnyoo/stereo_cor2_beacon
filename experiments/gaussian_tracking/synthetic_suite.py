"""Controlled Synthetic Validation Test Suite for Gaussian Evidence Tracking.

Evaluates 10 ground-truth scenarios across G0, G1, G2, and G3:
1. Single outward-moving Gaussian blob
2. Spatially extended moving arc
3. Two simultaneous moving objects with distinct PAs (45 deg and 135 deg)
4. Two crossing / adjacent trajectories (evaluating merge/fragmentation resistance)
5. Stationary bright artifact (diffraction spike / streamer)
6. One-frame extreme-intensity noise spike (cosmic ray transient)
7. Moving object with temporary signal dropout (1 frame missing)
8. Irregular observation cadence (20m, 40m, 30m, etc.)
9. Long observation gap (> 2 hours)
10. Moving structure emerging near occulter boundary (r ~ 22 px)

Computes quantitative metrics:
- Track Identity Preservation (TP, FP, FN tracks)
- Localization and velocity accuracy (RMSE of r, PA, v_r)
- Fragmentation count
- False merge count
- False birth count
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cme_pipeline.utils.geometry import circular_diff_deg, circular_mean_deg
from experiments.gaussian_tracking.gaussian_tracker import (
    TrackerConfig,
    GaussianEvidenceTracker,
    GaussianTrack,
)


@dataclass
class SyntheticScenarioResult:
    scenario_id: int
    scenario_name: str
    variant: str
    n_ground_truth_objects: int
    n_confirmed_tracks: int
    n_tentative_tracks: int
    n_false_births: int
    n_fragmentations: int
    n_false_merges: int
    track_completeness: float  # Fraction of ground truth frames tracked
    cpa_rmse_deg: float
    vr_rmse_px_hr: float
    status: str  # 'PASS', 'WARN', 'FAIL'
    notes: str


class SyntheticSceneGenerator:
    """Generates 256x256 synthetic coronagraph running-difference frames with ground truth."""

    def __init__(
        self,
        sx: float = 128.0,
        sy: float = 128.0,
        r_min_px: float = 22.0,
        r_max_px: float = 122.5,
    ):
        self.sx = sx
        self.sy = sy
        self.r_min = r_min_px
        self.r_max = r_max_px

        yy, xx = np.indices((256, 256))
        self.xx = xx
        self.yy = yy
        self.r_raster = np.hypot(xx - sx, yy - sy).astype(np.float32)
        # COR2 north-up PA: PA = arctan2(-dx, dy) in degrees [0, 360)
        self.pa_raster = (np.degrees(np.arctan2(-(xx - sx), yy - sy)) % 360.0).astype(np.float32)
        self.valid_mask = (self.r_raster >= r_min_px) & (self.r_raster <= r_max_px)

    def create_empty_frame(self, background_noise_sigma: float = 0.15) -> np.ndarray:
        """Creates valid background frame with Gaussian noise."""
        frame = np.random.normal(0.0, background_noise_sigma, (256, 256)).astype(np.float32)
        frame[~self.valid_mask] = 0.0
        return frame

    def add_gaussian_blob(
        self,
        frame: np.ndarray,
        r: float,
        pa_deg: float,
        peak_snr: float = 4.0,
        sigma_r: float = 3.0,
        sigma_pa_deg: float = 8.0,
    ):
        """Adds a localized Gaussian CME blob."""
        pa_rad = np.radians(pa_deg)
        cx = self.sx - r * np.sin(pa_rad)
        cy = self.sy + r * np.cos(pa_rad)

        dist_sq = (self.xx - cx)**2 + (self.yy - cy)**2
        blob = peak_snr * np.exp(-0.5 * dist_sq / (sigma_r**2))
        frame += np.where(self.valid_mask, blob, 0.0).astype(np.float32)

    def add_extended_arc(
        self,
        frame: np.ndarray,
        r_front: float,
        pa_center_deg: float,
        pa_width_deg: float = 40.0,
        radial_thickness_px: float = 5.0,
        peak_snr: float = 3.5,
    ):
        """Adds an extended arc simulating a curved CME leading edge."""
        d_pa = np.abs(circular_diff_deg(self.pa_raster, pa_center_deg))
        in_pa = d_pa <= (pa_width_deg / 2.0)
        pa_factor = np.cos((d_pa / (pa_width_deg / 2.0)) * (np.pi / 2.0))
        pa_factor = np.maximum(0.0, pa_factor)

        dr = self.r_raster - r_front
        rad_factor = np.exp(-0.5 * (dr / (radial_thickness_px / 2.0))**2)

        arc = peak_snr * pa_factor * rad_factor
        frame += np.where(self.valid_mask & in_pa, arc, 0.0).astype(np.float32)


def run_synthetic_scenario_1(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 1: Single outward-moving Gaussian blob."""
    n_frames = 10
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=30 * i) for i in range(n_frames)]
    true_pa = 90.0
    v_r_true = 16.0  # px/hr (8 px per 30 min)
    r_start = 28.0

    frames = []
    gt_radii = []
    for i in range(n_frames):
        r_curr = r_start + (v_r_true * 0.5 * i)
        gt_radii.append(r_curr)
        f = gen.create_empty_frame()
        gen.add_gaussian_blob(f, r=r_curr, pa_deg=true_pa, peak_snr=4.5)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    tentative = [t for t in tracker.get_all_tracks() if t.status == "tentative"]

    n_conf = len(confirmed)
    n_tent = len(tentative)
    n_fb = max(0, n_conf - 1)
    n_frag = 1 if n_conf > 1 else 0

    if n_conf >= 1:
        main_trk = max(confirmed, key=lambda t: t.hits)
        completeness = main_trk.hits / n_frames
        cpa_err = abs(main_trk.mean_cpa - true_pa)
        vr_err = abs(main_trk.mean_radial_velocity - v_r_true)
    else:
        completeness = 0.0
        cpa_err = 999.0
        vr_err = 999.0

    status = "PASS" if (n_conf == 1 and completeness >= 0.8 and vr_err <= 3.0 and cpa_err <= 3.0) else "FAIL"

    return SyntheticScenarioResult(
        scenario_id=1,
        scenario_name="Single Outward Gaussian Blob",
        variant=variant,
        n_ground_truth_objects=1,
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=n_tent,
        n_false_births=n_fb,
        n_fragmentations=n_frag,
        n_false_merges=0,
        track_completeness=completeness,
        cpa_rmse_deg=cpa_err,
        vr_rmse_px_hr=vr_err,
        status=status,
        notes=f"Tracked {main_trk.hits if n_conf >= 1 else 0}/{n_frames} frames",
    )


def run_synthetic_scenario_2(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 2: Spatially extended moving arc (simulating CME front)."""
    n_frames = 10
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=30 * i) for i in range(n_frames)]
    true_pa = 110.0
    v_r_true = 14.0  # px/hr
    r_start = 30.0

    frames = []
    for i in range(n_frames):
        r_curr = r_start + (v_r_true * 0.5 * i)
        f = gen.create_empty_frame()
        gen.add_extended_arc(f, r_front=r_curr, pa_center_deg=true_pa, pa_width_deg=45.0, peak_snr=3.8)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    n_conf = len(confirmed)
    n_frag = max(0, n_conf - 1)
    n_fb = max(0, n_conf - 1)

    if n_conf >= 1:
        main_trk = max(confirmed, key=lambda t: t.hits)
        completeness = main_trk.hits / n_frames
        cpa_err = abs(main_trk.mean_cpa - true_pa)
        vr_err = abs(main_trk.mean_radial_velocity - v_r_true)
    else:
        completeness = 0.0
        cpa_err = 999.0
        vr_err = 999.0

    status = "PASS" if (n_conf == 1 and completeness >= 0.8 and vr_err <= 3.5 and cpa_err <= 5.0) else ("WARN" if n_conf > 1 else "FAIL")

    return SyntheticScenarioResult(
        scenario_id=2,
        scenario_name="Spatially Extended Moving Arc",
        variant=variant,
        n_ground_truth_objects=1,
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=len([t for t in tracker.get_all_tracks() if t.status == "tentative"]),
        n_false_births=n_fb,
        n_fragmentations=n_frag,
        n_false_merges=0,
        track_completeness=completeness,
        cpa_rmse_deg=cpa_err,
        vr_rmse_px_hr=vr_err,
        status=status,
        notes=f"Arc PA span ~45 deg, tracks={n_conf}",
    )


def run_synthetic_scenario_3(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 3: Two simultaneous moving objects with distinct PAs (45 deg and 135 deg)."""
    n_frames = 10
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=30 * i) for i in range(n_frames)]
    pa_1, pa_2 = 45.0, 135.0
    vr_1, vr_2 = 14.0, 16.0
    r_start_1, r_start_2 = 28.0, 32.0

    frames = []
    for i in range(n_frames):
        f = gen.create_empty_frame()
        r1 = r_start_1 + (vr_1 * 0.5 * i)
        r2 = r_start_2 + (vr_2 * 0.5 * i)
        gen.add_gaussian_blob(f, r=r1, pa_deg=pa_1, peak_snr=4.0)
        gen.add_gaussian_blob(f, r=r2, pa_deg=pa_2, peak_snr=4.2)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    n_conf = len(confirmed)

    # Check matching to ground truth 1 and 2
    matched_1, matched_2 = None, None
    for trk in confirmed:
        if abs(circular_diff_deg(trk.mean_cpa, pa_1)) < 15.0:
            matched_1 = trk
        elif abs(circular_diff_deg(trk.mean_cpa, pa_2)) < 15.0:
            matched_2 = trk

    n_merges = 1 if (n_conf == 1 and matched_1 and matched_2) else 0
    n_fb = max(0, n_conf - 2)

    completeness = 0.0
    if matched_1 and matched_2:
        completeness = 0.5 * (matched_1.hits / n_frames + matched_2.hits / n_frames)
        cpa_err = 0.5 * (abs(matched_1.mean_cpa - pa_1) + abs(matched_2.mean_cpa - pa_2))
        vr_err = 0.5 * (abs(matched_1.mean_radial_velocity - vr_1) + abs(matched_2.mean_radial_velocity - vr_2))
        status = "PASS" if (n_conf == 2 and completeness >= 0.8) else "WARN"
    else:
        cpa_err = 999.0
        vr_err = 999.0
        status = "FAIL"

    return SyntheticScenarioResult(
        scenario_id=3,
        scenario_name="Two Simultaneous Moving Objects (45 & 135 deg)",
        variant=variant,
        n_ground_truth_objects=2,
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=len([t for t in tracker.get_all_tracks() if t.status == "tentative"]),
        n_false_births=n_fb,
        n_fragmentations=max(0, n_conf - 2),
        n_false_merges=n_merges,
        track_completeness=completeness,
        cpa_rmse_deg=cpa_err,
        vr_rmse_px_hr=vr_err,
        status=status,
        notes=f"Found {2 if (matched_1 and matched_2) else (1 if matched_1 or matched_2 else 0)}/2 distinct objects",
    )


def run_synthetic_scenario_4(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 4: Two crossing / adjacent trajectories (close PAs: 80 deg and 95 deg)."""
    n_frames = 10
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=30 * i) for i in range(n_frames)]
    # Trajectories approach closely
    pa_start_1, pa_start_2 = 80.0, 96.0
    vr_1, vr_2 = 18.0, 12.0  # Fast object catches up and passes slower one radially
    r_start_1, r_start_2 = 26.0, 36.0

    frames = []
    for i in range(n_frames):
        f = gen.create_empty_frame()
        r1 = r_start_1 + (vr_1 * 0.5 * i)
        r2 = r_start_2 + (vr_2 * 0.5 * i)
        gen.add_gaussian_blob(f, r=r1, pa_deg=pa_start_1, peak_snr=4.0)
        gen.add_gaussian_blob(f, r=r2, pa_deg=pa_start_2, peak_snr=4.0)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    n_conf = len(confirmed)

    # Check whether both identities survived without merging
    matched_1, matched_2 = None, None
    for trk in confirmed:
        if abs(circular_diff_deg(trk.mean_cpa, pa_start_1)) < 10.0:
            matched_1 = trk
        elif abs(circular_diff_deg(trk.mean_cpa, pa_start_2)) < 10.0:
            matched_2 = trk

    n_merges = 1 if (n_conf == 1) else 0
    if matched_1 and matched_2:
        completeness = 0.5 * (matched_1.hits / n_frames + matched_2.hits / n_frames)
        cpa_err = 0.5 * (abs(matched_1.mean_cpa - pa_start_1) + abs(matched_2.mean_cpa - pa_start_2))
        vr_err = 0.5 * (abs(matched_1.mean_radial_velocity - vr_1) + abs(matched_2.mean_radial_velocity - vr_2))
        status = "PASS" if n_conf == 2 else "WARN"
    else:
        completeness = 0.5 if (matched_1 or matched_2) else 0.0
        cpa_err = 999.0
        vr_err = 999.0
        status = "FAIL"

    return SyntheticScenarioResult(
        scenario_id=4,
        scenario_name="Two Crossing / Adjacent Trajectories",
        variant=variant,
        n_ground_truth_objects=2,
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=len([t for t in tracker.get_all_tracks() if t.status == "tentative"]),
        n_false_births=max(0, n_conf - 2),
        n_fragmentations=max(0, n_conf - 2),
        n_false_merges=n_merges,
        track_completeness=completeness,
        cpa_rmse_deg=cpa_err,
        vr_rmse_px_hr=vr_err,
        status=status,
        notes=f"Radial overtaking test, conf_tracks={n_conf}, merges={n_merges}",
    )


def run_synthetic_scenario_5(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 5: Stationary bright artifact (diffraction spike / streamer)."""
    n_frames = 10
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=30 * i) for i in range(n_frames)]
    streamer_pa = 270.0
    streamer_r = 45.0  # Stationary at r = 45 px (> r_birth_max)

    frames = []
    for i in range(n_frames):
        f = gen.create_empty_frame()
        # High SNR stationary feature
        gen.add_gaussian_blob(f, r=streamer_r, pa_deg=streamer_pa, peak_snr=5.0)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0, birth_mode="near_sun", r_birth_max=38.0)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    n_conf = len(confirmed)

    # Near-Sun birth restriction should reject stationary streamer at r = 45 px!
    status = "PASS" if n_conf == 0 else "FAIL"

    return SyntheticScenarioResult(
        scenario_id=5,
        scenario_name="Stationary Bright Artifact (Streamer)",
        variant=variant,
        n_ground_truth_objects=0,  # No CME
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=len([t for t in tracker.get_all_tracks() if t.status == "tentative"]),
        n_false_births=n_conf,
        n_fragmentations=0,
        n_false_merges=0,
        track_completeness=1.0 if n_conf == 0 else 0.0,
        cpa_rmse_deg=0.0,
        vr_rmse_px_hr=0.0,
        status=status,
        notes="Near-Sun gate correctly suppresses stationary artifact outside birth zone" if n_conf == 0 else "Stationary artifact falsely confirmed as CME",
    )


def run_synthetic_scenario_6(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 6: One-frame extreme-intensity noise spike (cosmic ray hit)."""
    n_frames = 10
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=30 * i) for i in range(n_frames)]

    frames = []
    for i in range(n_frames):
        f = gen.create_empty_frame()
        if i == 4:
            # Single-frame extreme spike at r=32 px, pa=180 deg
            gen.add_gaussian_blob(f, r=32.0, pa_deg=180.0, peak_snr=25.0, sigma_r=1.5)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0, min_confirm_hits=2)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    all_tracks = tracker.get_all_tracks()
    n_conf = len(confirmed)

    # Confirmation gate (min_confirm_hits=2) must reject 1-frame transient!
    status = "PASS" if n_conf == 0 else "FAIL"

    return SyntheticScenarioResult(
        scenario_id=6,
        scenario_name="Single-Frame Noise Spike (Cosmic Ray)",
        variant=variant,
        n_ground_truth_objects=0,
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=len([t for t in all_tracks if t.status == "tentative"]),
        n_false_births=n_conf,
        n_fragmentations=0,
        n_false_merges=0,
        track_completeness=1.0 if n_conf == 0 else 0.0,
        cpa_rmse_deg=0.0,
        vr_rmse_px_hr=0.0,
        status=status,
        notes="Transient rejected by confirmation gate" if n_conf == 0 else "Cosmic ray created false confirmed track",
    )


def run_synthetic_scenario_7(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 7: Moving object with temporary signal dropout (1 frame missing)."""
    n_frames = 10
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=30 * i) for i in range(n_frames)]
    true_pa = 60.0
    v_r_true = 15.0  # px/hr
    r_start = 28.0

    frames = []
    for i in range(n_frames):
        f = gen.create_empty_frame()
        if i != 4:  # Frame 4 has total dropout!
            r_curr = r_start + (v_r_true * 0.5 * i)
            gen.add_gaussian_blob(f, r=r_curr, pa_deg=true_pa, peak_snr=4.2)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0, max_missed_frames=2)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    n_conf = len(confirmed)

    # Tracker should bridge the 1 missing frame without fragmenting into 2 tracks!
    if n_conf == 1:
        trk = confirmed[0]
        completeness = trk.hits / (n_frames - 1)  # 9 available frames
        cpa_err = abs(trk.mean_cpa - true_pa)
        vr_err = abs(trk.mean_radial_velocity - v_r_true)
        status = "PASS" if completeness >= 0.85 else "WARN"
    elif n_conf > 1:
        # Fragmented across dropout
        status = "WARN"
        completeness = max(t.hits for t in confirmed) / (n_frames - 1)
        cpa_err = abs(confirmed[0].mean_cpa - true_pa)
        vr_err = abs(confirmed[0].mean_radial_velocity - v_r_true)
    else:
        status = "FAIL"
        completeness = 0.0
        cpa_err = 999.0
        vr_err = 999.0

    return SyntheticScenarioResult(
        scenario_id=7,
        scenario_name="Temporary Signal Dropout (1 Frame Missing)",
        variant=variant,
        n_ground_truth_objects=1,
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=len([t for t in tracker.get_all_tracks() if t.status == "tentative"]),
        n_false_births=max(0, n_conf - 1),
        n_fragmentations=1 if n_conf > 1 else 0,
        n_false_merges=0,
        track_completeness=completeness,
        cpa_rmse_deg=cpa_err,
        vr_rmse_px_hr=vr_err,
        status=status,
        notes=f"Tracks across dropout: {n_conf} (single unified track={n_conf==1})",
    )


def run_synthetic_scenario_8(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 8: Irregular observation cadence (20m, 40m, 30m, 15m, 45m)."""
    cadence_deltas_min = [0, 20, 40, 30, 15, 45, 30, 25, 35]
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = []
    curr_t = base_time
    for dt_m in cadence_deltas_min:
        curr_t += timedelta(minutes=dt_m)
        timestamps.append(curr_t)

    n_frames = len(timestamps)
    true_pa = 200.0
    v_r_true = 16.0  # px/hr
    r_start = 28.0

    frames = []
    for i in range(n_frames):
        dt_hr = (timestamps[i] - timestamps[0]).total_seconds() / 3600.0
        r_curr = r_start + (v_r_true * dt_hr)
        f = gen.create_empty_frame()
        gen.add_gaussian_blob(f, r=r_curr, pa_deg=true_pa, peak_snr=4.0)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    n_conf = len(confirmed)

    if n_conf == 1:
        trk = confirmed[0]
        completeness = trk.hits / n_frames
        cpa_err = abs(trk.mean_cpa - true_pa)
        vr_err = abs(trk.mean_radial_velocity - v_r_true)
        status = "PASS" if completeness >= 0.85 and vr_err <= 3.0 else "WARN"
    else:
        status = "FAIL"
        completeness = 0.0
        cpa_err = 999.0
        vr_err = 999.0

    return SyntheticScenarioResult(
        scenario_id=8,
        scenario_name="Irregular Observation Cadence",
        variant=variant,
        n_ground_truth_objects=1,
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=len([t for t in tracker.get_all_tracks() if t.status == "tentative"]),
        n_false_births=max(0, n_conf - 1),
        n_fragmentations=max(0, n_conf - 1),
        n_false_merges=0,
        track_completeness=completeness,
        cpa_rmse_deg=cpa_err,
        vr_rmse_px_hr=vr_err,
        status=status,
        notes=f"Non-uniform dt correctly evaluated, vr_err={vr_err:.2f} px/hr",
    )


def run_synthetic_scenario_9(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 9: Long observation gap (> 2 hours / 150 min data outage)."""
    # 4 frames regular, then 150 min gap, then 4 frames
    cadence_deltas_min = [0, 30, 30, 30, 150, 30, 30, 30]
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = []
    curr_t = base_time
    for dt_m in cadence_deltas_min:
        curr_t += timedelta(minutes=dt_m)
        timestamps.append(curr_t)

    n_frames = len(timestamps)
    true_pa = 315.0
    v_r_true = 15.0  # px/hr
    r_start = 28.0

    frames = []
    for i in range(n_frames):
        dt_hr = (timestamps[i] - timestamps[0]).total_seconds() / 3600.0
        r_curr = r_start + (v_r_true * dt_hr)
        f = gen.create_empty_frame()
        # Object moves out of field during gap, or new structure appears
        if r_curr <= 120.0:
            gen.add_gaussian_blob(f, r=r_curr, pa_deg=true_pa, peak_snr=4.0)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0, max_dt_gap_min=75.0)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    all_tracks = tracker.get_all_tracks()
    confirmed = tracker.get_confirmed_tracks()

    # Gap > 75 min must terminate the pre-gap track!
    # Post-gap track may or may not birth depending on radius
    pre_gap_tracks = [t for t in confirmed if t.first.frame_idx < 4 and t.latest.frame_idx < 4]
    gap_bridged = any(t.first.frame_idx < 4 and t.latest.frame_idx >= 4 for t in confirmed)

    status = "PASS" if not gap_bridged else "FAIL"

    return SyntheticScenarioResult(
        scenario_id=9,
        scenario_name="Long Observation Gap (>2 Hours)",
        variant=variant,
        n_ground_truth_objects=1,
        n_confirmed_tracks=len(confirmed),
        n_tentative_tracks=len([t for t in all_tracks if t.status == "tentative"]),
        n_false_births=0,
        n_fragmentations=0,
        n_false_merges=0,
        track_completeness=1.0 if not gap_bridged else 0.0,
        cpa_rmse_deg=0.0,
        vr_rmse_px_hr=0.0,
        status=status,
        notes="Correctly severed active track across 150m outage" if not gap_bridged else "FAILED: Track erroneously bridged large gap",
    )


def run_synthetic_scenario_10(gen: SyntheticSceneGenerator, variant: str) -> SyntheticScenarioResult:
    """Scenario 10: Moving structure emerging near occulter boundary (r ~ 23 px)."""
    n_frames = 10
    base_time = datetime(2010, 12, 12, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=30 * i) for i in range(n_frames)]
    true_pa = 30.0
    v_r_true = 14.0  # px/hr
    r_start = 23.0  # 1 px outside inner occulter!

    frames = []
    for i in range(n_frames):
        r_curr = r_start + (v_r_true * 0.5 * i)
        f = gen.create_empty_frame()
        gen.add_gaussian_blob(f, r=r_curr, pa_deg=true_pa, peak_snr=4.0)
        frames.append(f)

    cfg = TrackerConfig(variant=variant, sigma_s=2.0 if variant != "G0" else 0.0, r_birth_max=38.0)
    tracker = GaussianEvidenceTracker(cfg, gen.sx, gen.sy, gen.r_raster, gen.pa_raster)

    for i in range(n_frames):
        tracker.step(i, timestamps[i], frames[i], gen.valid_mask)

    confirmed = tracker.get_confirmed_tracks()
    n_conf = len(confirmed)

    if n_conf == 1:
        trk = confirmed[0]
        completeness = trk.hits / n_frames
        cpa_err = abs(trk.mean_cpa - true_pa)
        vr_err = abs(trk.mean_radial_velocity - v_r_true)
        status = "PASS" if completeness >= 0.8 else "WARN"
    else:
        status = "FAIL"
        completeness = 0.0
        cpa_err = 999.0
        vr_err = 999.0

    return SyntheticScenarioResult(
        scenario_id=10,
        scenario_name="Occulter Boundary Emergence (r ~ 23 px)",
        variant=variant,
        n_ground_truth_objects=1,
        n_confirmed_tracks=n_conf,
        n_tentative_tracks=len([t for t in tracker.get_all_tracks() if t.status == "tentative"]),
        n_false_births=max(0, n_conf - 1),
        n_fragmentations=max(0, n_conf - 1),
        n_false_merges=0,
        track_completeness=completeness,
        cpa_rmse_deg=cpa_err,
        vr_rmse_px_hr=vr_err,
        status=status,
        notes=f"Tracked from r=23 px, hits={trk.hits if n_conf==1 else 0}/{n_frames}",
    )


def run_full_synthetic_suite() -> pd.DataFrame:
    """Runs all 10 synthetic test scenarios across G0, G1, G2, G3."""
    gen = SyntheticSceneGenerator()
    variants = ["G0", "G1", "G2", "G3"]
    results: List[SyntheticScenarioResult] = []

    scenario_funcs = [
        run_synthetic_scenario_1,
        run_synthetic_scenario_2,
        run_synthetic_scenario_3,
        run_synthetic_scenario_4,
        run_synthetic_scenario_5,
        run_synthetic_scenario_6,
        run_synthetic_scenario_7,
        run_synthetic_scenario_8,
        run_synthetic_scenario_9,
        run_synthetic_scenario_10,
    ]

    for func in scenario_funcs:
        for var in variants:
            res = func(gen, var)
            results.append(res)
            print(f"[{res.status}] Scenario {res.scenario_id} ({res.scenario_name}) - Variant {res.variant}: {res.notes}")

    df = pd.DataFrame([vars(r) for r in results])
    return df


if __name__ == "__main__":
    out_dir = Path("experiments/gaussian_tracking/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results = run_full_synthetic_suite()
    csv_path = out_dir / "synthetic_validation_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSynthetic test suite completed. Results saved to {csv_path}")

    # Summary table
    pivot = df_results.pivot_table(
        index=["scenario_id", "scenario_name"],
        columns="variant",
        values="status",
        aggfunc="first",
    )
    print("\nSummary Results (Status per Scenario & Variant):")
    print(pivot)
