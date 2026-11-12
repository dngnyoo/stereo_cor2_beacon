#!/usr/bin/env python3
"""Run Recovery -> Validation on independently recomputed E1 tracking.

Regression replay is explicitly confined to a previously generated E1 track
CSV and NEW live C0 detection products. Never reads historical E1 track
states as pipeline inputs or rewrites general 63-pair preprocessing products.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import DetectionOutput, GeometryContext, TrackingOutput
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output
from cme_pipeline.core.pipeline import resolve_stages
from cme_pipeline.utils.geometry import compute_wcs_rasters
from astropy.io import fits
from astropy.wcs import WCS


def build_tracking(detection_dir: Path, states_path: Path, cfg: PipelineConfig) -> TrackingOutput:
    det = load_live_tracking_input(detection_dir)
    # Reproduce benchmark's policy: tracking only on VALID analysis-date pairs.
    manifest = det.manifest_df
    keep = np.flatnonzero(
        manifest["warmup_or_eval"].eq("EVALUATION").to_numpy()
        & manifest["usable_detection"].astype(bool).to_numpy()
    )
    if not len(keep):
        raise ValueError("No VALID evaluation observations")
    cut_manifest = manifest.iloc[keep].copy().reset_index(drop=True)
    cut_manifest["f_idx"] = np.arange(len(keep))
    cut_manifest["gap_from_prev_min"] = 0.0
    cut_manifest["is_large_gap"] = False
    frames = []
    prev_ts = None
    for local, original in enumerate(keep):
        fc = dict(det.frame_cache[int(original)])
        ts = pd.Timestamp(fc["timestamp"])
        gap = 0.0 if prev_ts is None else (ts - prev_ts).total_seconds() / 60.0
        large = bool(prev_ts is not None and gap * 60 > cfg.tracking.max_cadence_gap_sec)
        fc.update({"f_idx": local, "gap_from_prev_min": gap, "is_large_gap": large})
        frames.append(fc)
        cut_manifest.at[local, "gap_from_prev_min"] = gap
        cut_manifest.at[local, "is_large_gap"] = large
        prev_ts = ts
    states = pd.read_csv(states_path)
    allowed = set(cut_manifest["pair_index"].astype(int))
    if states.empty or not set(states["pair_index"].astype(int)).issubset(allowed):
        raise ValueError("Recomputed states do not align with selected evaluation pairs")
    observed = states.loc[states["state"] == "OBSERVED"]
    if observed.duplicated(["pair_index", "component_id"]).any():
        raise ValueError("Duplicate component assignments in E1 states")
    # Match the validated historical E1 geometry convention. This explicit
    # historical convention is limited to this replay runner, not generic C0.
    header = det.geometry.wcs_header_str
    wcs = WCS(fits.Header.fromstring(header, sep="\n"))
    sx, sy = 126.618, 127.503
    rasters = compute_wcs_rasters(wcs=wcs, center_xy=(sx, sy), patch_size=4)
    geometry = GeometryContext(
        sx=sx, sy=sy, rsun_px=8.163, occulter_inner_radius_px=16.326,
        occulter_r_min_px=cfg.segmentation.occulter_r_min_px,
        fov_r_max_px=cfg.segmentation.fov_r_max_px,
        wcs_header_str=header, **rasters
    )
    observed_birth = (observed.sort_values(["track_id", "pair_index"])
                      .groupby("track_id").first())
    anchors = sorted(int(t) for t, row in observed_birth.iterrows()
                     if float(row["birth_r_min_px"]) <= cfg.birth_gate.max_birth_radius_px)
    grid = np.asarray(det.patch_activation_grid4)[keep].copy()
    component_labels = np.asarray(det.raw_component_labels)[keep].copy()
    # Every observed state must correspond to the actual C0 component.
    by_pair = {int(v): i for i, v in enumerate(cut_manifest["pair_index"])}
    for row in observed.itertuples(index=False):
        idx = by_pair[int(row.pair_index)]
        if int(row.component_id) <= 0 or not np.any(component_labels[idx] == int(row.component_id)):
            raise ValueError(f"Missing C0 component for {row.component_uid}")
    meta = dict(det.dataset_metadata)
    meta.update({
        "replay_mode": "recomputed_e1_frozen_evaluation",
        "recomputed_track_states_path": str(states_path.resolve()),
        "eval_pairs": len(keep),
        "excluded_warmup_pairs": len(manifest) - len(keep),
    })
    raw = TrackingOutput(
        manifest_df=cut_manifest, track_states_df=states,
        patch_component_labels=component_labels, patch_activation_grid4=grid,
        frame_cache=frames, geometry=geometry,
        total_tracks=int(states["track_id"].nunique()),
        anchor_tracks=anchors, tracking_method="e1",
        tracking_parameters={"alpha": 0.0, "patch_size": 4,
                             "max_cadence_gap_sec": cfg.tracking.max_cadence_gap_sec},
        dataset_metadata=meta
    )
    return adapt_tracking_output(raw, max_birth_radius_px=cfg.birth_gate.max_birth_radius_px)


def run(args):
    if args.run_name == "baseline_rule_b":
        raise ValueError("Refusing to overwrite baseline_rule_b")
    cfg = PipelineConfig.from_yaml(args.config)
    if cfg.tracking.method != "e1":
        raise ValueError("Frozen E1 replay requires e1 tracking configuration")
    cfg.run_name = args.run_name
    cfg.repo_root = str(args.repo.resolve())
    cfg.cache_enabled = False
    run_dir = args.repo.resolve() / cfg.output_base_dir / cfg.run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing replay run: {run_dir}")
    out = build_tracking(args.detection_dir.resolve(), args.recomputed_states.resolve(), cfg)
    ctx = PipelineContext(cfg)
    stages = resolve_stages(cfg)
    # Tracking was already independently recomputed: no Frozen E1 CSV read.
    stage_track = ctx.stages_dir / "tracking"
    stage_track.mkdir(exist_ok=True)
    out.track_states_df.to_csv(stage_track / "recomputed_track_states.csv", index=False)
    out.manifest_df.to_csv(stage_track / "frame_manifest.csv", index=False)
    current = out
    stage_names = []
    for stage in stages[1:]:
        ctx.info(f"[Frozen replay] Running {stage.name}")
        current = stage.run(current, cfg, ctx)
        stage_names.append(stage.name)
    seg = current.segmentation_output
    sizes = seg.size_gate_output
    assoc = sizes.birth_gate_output.association_output
    report = {
        "status": "DOWNSTREAM_REPLAY_EXECUTED_NOT_YET_CERTIFIED",
        "source": {
            "detection_dir": str(args.detection_dir.resolve()),
            "recomputed_track_states": str(args.recomputed_states.resolve()),
        },
        "eval_pairs": len(out.manifest_df),
        "e1_tracks": out.total_tracks,
        "candidate_families": assoc.candidate_family_count,
        "admitted_families": len(sizes.admitted_family_ids),
        "admitted_tracks": len(sizes.admitted_track_ids),
        "recovered_patches": sizes.total_recovered_patches,
        "union_mask_pixels": int(seg.final_pixel_segmentation_union.sum()),
        "completed_stages": stage_names,
        "baseline_reference_metrics": {
            "e1_tracks": 219, "candidate_families": 185,
            "admitted_families": 15, "admitted_tracks": 37,
            "recovered_patches": 5174, "union_mask_pixels": 50665,
        },
    }
    reference_mask = args.repo.resolve() / "runs/baseline_rule_b/arrays/final_pixel_segmentation_union.npy"
    if reference_mask.exists():
        frozen = np.load(reference_mask, mmap_mode="r")
        now = seg.final_pixel_segmentation_union
        if frozen.shape == now.shape:
            report["local_baseline_pixel_mask_exact"] = bool(np.array_equal(frozen, now))
            report["local_baseline_pixel_mismatch_count"] = int(np.count_nonzero(frozen != now))
        else:
            report["local_baseline_mask_shape_mismatch"] = [list(frozen.shape), list(now.shape)]
    (ctx.run_dir / "frozen_replay_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_rule_b.yaml"))
    parser.add_argument("--run-name", default="frozen_replay_e1_downstream")
    parser.add_argument("--detection-dir", type=Path, default=Path(
        "runs/dual_archive_20101212/frozen_replay/detection_c0_gaussian"))
    parser.add_argument("--recomputed-states", type=Path, default=Path(
        "runs/dual_archive_20101212/frozen_replay/tracking_e1_recomputed/03_patch_e1_track_states.csv"))
    run(parser.parse_args())
