"""Config-driven live COR2 execution from FITS, RD, or C0/C3 detection.

Live tracking uses the registered stage implementation; the historical E1
recomputation/replay remains isolated in live_full_pipeline.py.
SEEDS validation uses the standalone timestamp-based evaluator, never the
historical 32-frame-index validator.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline, resolve_stages
from cme_pipeline.data_preparation import prepare_source, validate_date
from cme_pipeline.live_detection import run_detection
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.north_up_preprocessing import run_no_msb_north_up


def run_configured_data(
    *, config_path: str | Path, date: str, run_name: str,
    detection_dir: str | Path | None = None,
    preprocessed_dir: str | Path | None = None,
    source: str | None = None,
    input_dir: str | Path | None = None,
    acquisition_dir: str | Path | None = None,
    warmup_days: int = 2,
    phase_sec: float | None = None,
    mask_path: str | Path | None = None,
    method: str | None = None,
    denoiser: str | None = None,
    resume: bool = False,
) -> Path:
    """Run a configured tracker through Products and independent SEEDS validation."""
    from cme_pipeline.live_full_pipeline import check_run_request, stage_ready

    cfg = PipelineConfig.from_yaml(config_path)
    date = validate_date(date)
    detector = cfg.detection.method.lower()
    smooth = cfg.detection.denoiser.lower()
    if method is not None and method != detector:
        raise ValueError("CLI --method conflicts with detection.method in YAML")
    if denoiser is not None and denoiser != smooth:
        raise ValueError("CLI --denoiser conflicts with detection.denoiser in YAML")
    if detector not in ("c0", "c3") or smooth not in ("gaussian", "median"):
        raise ValueError(f"Unsupported detection configuration: {detector}/{smooth}")
    if cfg.tracking.method.lower() == "e1":
        raise ValueError(
            "Live E1 still requires the separate independent recomputation adapter; "
            "use C0/E1 legacy run-data while the E1 migration is tested."
        )
    if sum(v is not None for v in (detection_dir, preprocessed_dir, source)) != 1:
        raise ValueError("Select exactly one input: --source, --preprocessed-dir, or --detection-dir")
    if source is not None and source not in ("local", "download"):
        raise ValueError("Source must be local or download")
    if source == "local" and not input_dir:
        raise ValueError("--source local requires --input-dir")
    if warmup_days < 0:
        raise ValueError("--warmup-days must be nonnegative")

    # The catalog reports front heights in R_sun; never silently use NaN or a
    # pixel scale borrowed from another observation sequence.
    rsun_px = cfg.geometry.rsun_px
    if rsun_px is None or not math.isfinite(float(rsun_px)) or float(rsun_px) <= 0:
        raise ValueError(
            "geometry.rsun_px must be a positive, independently verified solar "
            "radius in pixels for this dataset (required by CME products)."
        )
    if not cfg.products.catalog:
        raise ValueError("Config-driven SEEDS comparison requires products.catalog: true")

    cfg.date_str = int(date)
    cfg.run_name = run_name
    cfg.repo_root = str(Path(cfg.repo_root).resolve())
    repo = Path(cfg.repo_root)
    run_dir = (repo / cfg.output_base_dir / run_name).resolve()
    if not run_name or run_name == "baseline_rule_b":
        raise ValueError("Choose a new run name, not baseline_rule_b")

    seeds_csv = repo / "data" / "catalogs" / "seeds" / date / "seeds_frames.csv"
    use_seeds = any(str(c).casefold() == "seeds" for c in cfg.validation.catalogs)
    if use_seeds and not seeds_csv.is_file():
        raise FileNotFoundError(
            f"SEEDS frame reference missing: {seeds_csv}. "
            f"Run scripts.fetch_seeds_monthly and scripts.fetch_seeds_details first, "
            f"or exclude SEEDS from validation.catalogs."
        )

    request = {
        "schema_version": 3,
        "run_name": run_name,
        "date": date,
        "config": cfg.to_dict(),
        "source": source,
        "input_dir": str(Path(input_dir).resolve()) if input_dir else None,
        "preprocessed_dir": str(Path(preprocessed_dir).resolve()) if preprocessed_dir else None,
        "detection_dir": str(Path(detection_dir).resolve()) if detection_dir else None,
        "acquisition_dir": str(Path(acquisition_dir).resolve()) if acquisition_dir else None,
        "warmup_days": warmup_days,
        "phase_sec": phase_sec,
        "mask_path": str(Path(mask_path).resolve()) if mask_path else None,
    }
    if check_run_request(run_dir, request, resume=resume):
        print(f"Verified complete run: {run_dir}")
        return run_dir

    if source is not None:
        acq_dir = Path(acquisition_dir).resolve() if acquisition_dir else run_dir / "acquisition"
        selected = acq_dir / "input_manifest.json"
        if not (resume and selected.is_file()):
            selected = prepare_source(
                date=date, source=source, product="d7", input_dir=input_dir,
                output_dir=acq_dir, warmup_days=warmup_days, phase_sec=phase_sec,
                raw_root=repo / "data" / "raw" / "cor2" if source == "download" else None,
            )
        preprocessed_dir = acq_dir / "preprocessed"
        if not (resume and stage_ready(preprocessed_dir, (
            "preprocessing_metadata.json", "01_rd_manifest.csv",
            "north_up_RD_DNs_per_second.npy", "north_up_pair_valid_masks.npy",
            "target_wcs_header.txt",
        ))):
            preprocessed_dir = run_no_msb_north_up(selected, mask_path=mask_path)

    if preprocessed_dir is not None:
        preprocessed_dir = Path(preprocessed_dir).resolve()
        detected = run_dir / "detection"
        if not (resume and stage_ready(detected, (
            "detection_metadata.json", "frame_detection_summary.csv",
            "patch_state_p04.npy", "patch_component_labels.npy",
        ))):
            detected = run_detection(
                preprocessed_dir, method=detector, denoiser=smooth,
                requested_date=date, output_dir=detected,
            )
        detection_dir = detected

    assert detection_dir is not None
    detection_dir = Path(detection_dir).resolve()
    meta = json.loads((detection_dir / "detection_metadata.json").read_text())
    if (str(meta.get("requested_date")) != date
            or meta.get("detector") != detector
            or meta.get("denoiser") != smooth):
        raise ValueError("Detection metadata differs from requested date or YAML configuration")

    detected = load_live_tracking_input(
        detection_dir,
        max_cadence_gap_sec=float(cfg.tracking.max_cadence_gap_sec),
        gaussian_sigma=cfg.recovery.gaussian_sigma,
        occulter_r_min_px=cfg.segmentation.occulter_r_min_px,
        fov_r_max_px=cfg.segmentation.fov_r_max_px,
    )
    detected.geometry.rsun_px = float(rsun_px)
    # The legacy recovery operator requires the physical 2-R_sun inner
    # occulter boundary; the live loader leaves this field unspecified.
    detected.geometry.occulter_inner_radius_px = 2.0 * float(rsun_px)
    if not detected.manifest_df["warmup_or_eval"].eq("EVALUATION").any():
        raise ValueError("No requested-date evaluation frames")

    # Old validator uses historical 32-frame indexes and is intentionally
    # excluded. The independent official SEEDS validator runs after Products.
    stages = [stage for stage in resolve_stages(cfg) if stage.name != "validation"]
    result = Pipeline(cfg, stages=stages).run(initial_input=detected)
    tracking = result["stage_outputs"]["tracking"]
    if tracking.track_ownership_masks is not None:
        np.save(run_dir / "arrays" / "track_ownership_masks.npy",
                tracking.track_ownership_masks)
    tracking.track_states_df.to_csv(
        run_dir / "tables" / "live_tracking_states.csv", index=False
    )
    products = result["stage_outputs"]["products"]
    if not (run_dir / "tables" / "cme_frame_measurements.csv").is_file():
        raise RuntimeError("Products stage did not produce CME frame measurements")

    comparison = None
    if use_seeds:
        from scripts.validate_seeds import main as validate_seeds_main
        rc = validate_seeds_main([
            "--run-dir", str(run_dir),
            "--seeds-csv", str(seeds_csv),
        ])
        if rc not in (None, 0):
            raise RuntimeError(f"Independent SEEDS validation failed: {rc}")
        comparison = json.loads(
            (run_dir / "validation" / "seeds" / "summary.json").read_text()
        )

    summary = {
        "status": "INTEGRATED_EXECUTED_NOT_FULLY_CERTIFIED",
        "date": date,
        "run_name": run_name,
        "detector": detector,
        "denoiser": smooth,
        "tracking_method": cfg.tracking.method,
        "association_method": cfg.association.method,
        "rsun_px": float(rsun_px),
        "occulter_inner_radius_px": 2.0 * float(rsun_px),
        "input_pairs": len(detected.manifest_df),
        "evaluation_pairs": int(detected.manifest_df["warmup_or_eval"].eq("EVALUATION").sum()),
        "tracks": tracking.total_tracks,
        "cme_events": len(products.event_catalog_df),
        "seeds_comparison": comparison,
        "caveat": "SEEDS agreement is not CME ground truth; speed fits are exploratory.",
    }
    (run_dir / "integrated_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return run_dir
