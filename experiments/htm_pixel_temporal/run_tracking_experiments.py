"""Controlled Experiments and Quantitative Comparison: A5b vs H3 vs H4.

Evaluates 7 tracking configurations across 2010-12-12 STEREO-A COR2:
1. A5b Baseline
2. H3 with H1 static temporal accumulation
3. H3 with H2 motion-compensated temporal accumulation
4. H4 with H1 (conservative tie-breaker)
5. H4 with H2 (conservative tie-breaker)
6. H4 with H1 (direct support)
7. H4 with H2 (direct support)

Measures:
- CPA jitter & stability
- Track continuity, duration, and fragmentation
- Identity-switch proxies
- Kinematic outward consistency
- Intervention counts & audit log
- Downstream gate admission and segmentation
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.pipeline import resolve_stages
from cme_pipeline.core.stage import TrackingOutput
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.stages.tracking.registry import get_tracking_stage
from cme_pipeline.utils.geometry import circular_diff_deg


def compute_tracking_metrics(out: TrackingOutput, variant_name: str) -> Dict[str, Any]:
    """Computes comprehensive tracking quality and stability metrics."""
    df_states = out.track_states_df.copy()
    obs_states = df_states[df_states["state"] == "OBSERVED"].copy()

    total_tracks = int(out.total_tracks)
    total_obs = len(obs_states)

    if obs_states.empty:
        return {
            "variant": variant_name,
            "total_tracks": total_tracks,
            "total_obs_states": 0,
            "mean_cpa_jitter_deg": 0.0,
            "median_cpa_jitter_deg": 0.0,
            "mean_lifespan_frames": 0.0,
            "max_lifespan_frames": 0,
            "identity_switch_proxies": 0,
            "outward_consistency_pct": 0.0,
            "num_interventions": 0,
        }

    # Group by track and sort by pair_index
    obs_states["pair_index"] = obs_states["pair_index"].astype(int)
    obs_states = obs_states.sort_values(["track_id", "pair_index"])

    jitters: List[float] = []
    lifespans: List[int] = []
    id_switches: int = 0
    outward_steps: int = 0
    total_steps: int = 0

    for tid, grp in obs_states.groupby("track_id"):
        lifespans.append(len(grp))
        if len(grp) < 2:
            continue

        if "tracked_cpa" in grp.columns and grp["tracked_cpa"].notna().any():
            cpas = grp["tracked_cpa"].values
        elif "a5b_tracked_cpa" in grp.columns and grp["a5b_tracked_cpa"].notna().any():
            cpas = grp["a5b_tracked_cpa"].values
        else:
            cpas = grp["observed_component_cpa"].values if "observed_component_cpa" in grp.columns else grp["central_pa"].values

        front_col = "front_radius" if "front_radius" in grp.columns else ("leading_front_radius" if "leading_front_radius" in grp.columns else "radial_front")
        fronts = grp[front_col].values if front_col in grp.columns else None

        for i in range(1, len(grp)):
            d_cpa = abs(float(circular_diff_deg(cpas[i], cpas[i - 1])))
            jitters.append(d_cpa)
            if d_cpa > 30.0:
                id_switches += 1

            if fronts is not None:
                total_steps += 1
                # Outward motion condition: displacement >= -2.0 px (accounting for noise/grid quantization)
                if fronts[i] >= fronts[i - 1] - 2.0:
                    outward_steps += 1

    interventions = getattr(out, "extra_metadata", {}).get("interventions", [])
    if not interventions and hasattr(out, "interventions_df"):
        interventions = out.interventions_df.to_dict(orient="records")

    mean_jitter = float(np.mean(jitters)) if jitters else 0.0
    median_jitter = float(np.median(jitters)) if jitters else 0.0
    mean_life = float(np.mean(lifespans)) if lifespans else 0.0
    max_life = int(np.max(lifespans)) if lifespans else 0
    outward_pct = float(outward_steps / total_steps * 100.0) if total_steps > 0 else 100.0

    # Specific CME branch tracking:
    # Track 1: SE lobe (CPA ~110°-130°)
    # Track 2: NE lobe (CPA ~30°-50°)
    t1_obs = obs_states[obs_states["track_id"] == 1]
    t2_obs = obs_states[obs_states["track_id"] == 2]

    t1_cpa_col = "tracked_cpa" if ("tracked_cpa" in t1_obs.columns and t1_obs["tracked_cpa"].notna().any()) else "a5b_tracked_cpa"
    t1_cpa_mean = float(t1_obs[t1_cpa_col].mean()) if not t1_obs.empty else np.nan
    t2_cpa_col = "tracked_cpa" if ("tracked_cpa" in t2_obs.columns and t2_obs["tracked_cpa"].notna().any()) else "a5b_tracked_cpa"
    t2_cpa_mean = float(t2_obs[t2_cpa_col].mean()) if not t2_obs.empty else np.nan

    return {
        "variant": variant_name,
        "total_tracks": total_tracks,
        "total_obs_states": total_obs,
        "mean_cpa_jitter_deg": round(mean_jitter, 2),
        "median_cpa_jitter_deg": round(median_jitter, 2),
        "mean_lifespan_frames": round(mean_life, 2),
        "max_lifespan_frames": max_life,
        "identity_switch_proxies": id_switches,
        "outward_consistency_pct": round(outward_pct, 1),
        "num_interventions": len(interventions),
        "track1_cpa_mean": round(t1_cpa_mean, 1) if not np.isnan(t1_cpa_mean) else None,
        "track1_obs_count": len(t1_obs),
        "track2_cpa_mean": round(t2_cpa_mean, 1) if not np.isnan(t2_cpa_mean) else None,
        "track2_obs_count": len(t2_obs),
    }


def run_downstream_stages(track_out: TrackingOutput, config: PipelineConfig, context: PipelineContext) -> Dict[str, Any]:
    """Runs downstream pipeline stages: Recovery, Association, Gates, Segmentation, Validation."""
    stages = resolve_stages(config)
    # Skip the tracking stage (first in list)
    curr_input: Any = track_out
    stage_outputs: Dict[str, Any] = {"tracking": track_out}

    for s in stages[1:]:
        curr_input = s.run(curr_input, config, context)
        stage_outputs[s.name] = curr_input

    # Extract downstream summary metrics
    size_gate_out = stage_outputs.get("size_gate")
    seg_out = stage_outputs.get("segmentation")
    val_out = stage_outputs.get("validation")

    admitted_fams = len(size_gate_out.admitted_family_ids) if size_gate_out else 0
    admitted_tracks = len(size_gate_out.admitted_track_ids) if size_gate_out else 0
    seg_pixels = seg_out.total_segmented_pixels if seg_out else 0

    seeds_iou = None
    if val_out is not None and hasattr(val_out, "comparison_metrics_df") and not val_out.comparison_metrics_df.empty:
        seeds_row = val_out.comparison_metrics_df[val_out.comparison_metrics_df["Reference"] == "SEEDS"]
        if not seeds_row.empty:
            seeds_iou = float(seeds_row.iloc[0]["Angular IoU"])

    return {
        "admitted_families": admitted_fams,
        "admitted_tracks": admitted_tracks,
        "total_segmented_pixels": seg_pixels,
        "seeds_mean_iou": seeds_iou,
    }


def main():
    print("=" * 80)
    print("Stereo_COR2 Phase 3: Controlled Tracking Experiments (A5b, H3, H4)")
    print("=" * 80)

    det_out = load_live_tracking_input("runs/clean_cor2a_verify/detection")
    if not np.isfinite(det_out.geometry.rsun_px) or det_out.geometry.rsun_px <= 0:
        det_out.geometry.rsun_px = 8.163
        det_out.geometry.occulter_inner_radius_px = 2.0 * 8.163
    out_dir = repo_root / "experiments/htm_pixel_temporal/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "a5b_baseline",
            "method": "a5b",
            "params": {},
        },
        {
            "name": "h3_h1",
            "method": "h3",
            "params": {
                "temporal_variant": "H1",
                "temporal_mode": "linear_zpos",
                "temporal_half_life_min": 45.0,
            },
        },
        {
            "name": "h3_h2",
            "method": "h3",
            "params": {
                "temporal_variant": "H2",
                "temporal_mode": "linear_zpos",
                "temporal_half_life_min": 45.0,
                "temporal_radial_speed_px_30m": 8.0,
            },
        },
        {
            "name": "h4_h1_tie",
            "method": "h4",
            "params": {
                "temporal_variant": "H1",
                "temporal_mode": "linear_zpos",
                "association_strategy": "tie_breaker",
                "temporal_half_life_min": 45.0,
            },
        },
        {
            "name": "h4_h2_tie",
            "method": "h4",
            "params": {
                "temporal_variant": "H2",
                "temporal_mode": "linear_zpos",
                "association_strategy": "tie_breaker",
                "temporal_half_life_min": 45.0,
                "temporal_radial_speed_px_30m": 8.0,
            },
        },
        {
            "name": "h4_h1_direct",
            "method": "h4",
            "params": {
                "temporal_variant": "H1",
                "temporal_mode": "linear_zpos",
                "association_strategy": "direct_support",
                "temporal_half_life_min": 45.0,
            },
        },
        {
            "name": "h4_h2_direct",
            "method": "h4",
            "params": {
                "temporal_variant": "H2",
                "temporal_mode": "linear_zpos",
                "association_strategy": "direct_support",
                "temporal_half_life_min": 45.0,
                "temporal_radial_speed_px_30m": 8.0,
            },
        },
    ]

    all_metrics: List[Dict[str, Any]] = []

    for exp in experiments:
        name = exp["name"]
        method = exp["method"]
        params = exp["params"]

        print(f"\n>>> Running Experiment: {name} (method={method}) <<<")
        cfg = PipelineConfig()
        cfg.run_name = f"exp_{name}"
        cfg.tracking.method = method
        cfg.tracking.parameters = params
        cfg.tracking.outward_gate_enabled = True
        cfg.tracking.outward_gate_tolerance_px = 4.0
        cfg.tracking.max_missed_frames = 1
        cfg.tracking.terminate_on_qc = True

        ctx = PipelineContext(cfg)
        stage = get_tracking_stage(method)

        t_start = datetime.now()
        track_out = stage.run(det_out, cfg, ctx)
        t_track = (datetime.now() - t_start).total_seconds()

        # Compute tracking metrics
        m = compute_tracking_metrics(track_out, name)
        m["tracking_runtime_sec"] = round(t_track, 2)

        # Save track states and interventions
        track_out.track_states_df.to_csv(out_dir / f"track_states_{name}.csv", index=False)
        if hasattr(track_out, "interventions_df") and not track_out.interventions_df.empty:
            track_out.interventions_df.to_csv(out_dir / f"interventions_{name}.csv", index=False)
        elif "interventions" in track_out.dataset_metadata:
            pd.DataFrame(track_out.dataset_metadata["interventions"]).to_csv(
                out_dir / f"interventions_{name}.csv", index=False
            )

        # Run downstream pipeline
        print(f"  Running downstream stages for {name}...")
        t_down_start = datetime.now()
        down_m = run_downstream_stages(track_out, cfg, ctx)
        t_down = (datetime.now() - t_down_start).total_seconds()
        m.update(down_m)
        m["downstream_runtime_sec"] = round(t_down, 2)

        print(f"  Metrics for {name}:")
        for k, v in m.items():
            print(f"    {k}: {v}")

        all_metrics.append(m)

    df_results = pd.DataFrame(all_metrics)
    summary_path = out_dir / "tracking_comparison_metrics.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"\nAll experiments complete. Summary written to {summary_path}")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
