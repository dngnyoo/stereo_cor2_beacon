"""Single-entry COR2 C0/E1 integration: FITS or RD or detection through CME products.

No historical Tracking CSV is used as an input. The original benchmark
remains available through the legacy 'run' CLI and explicit --frozen-regression.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline, resolve_stages


def run_live_full(*, config_path: str | Path, date: str,
                  run_name: str, detection_dir: str | Path | None = None,
                  preprocessed_dir: str | Path | None = None,
                  source: str | None = None, input_dir: str | Path | None = None,
                  acquisition_dir: str | Path | None = None,
                  warmup_days: int = 2, phase_sec: float | None = None,
                  mask_path: str | Path | None = None,
                  method: str = "c0", denoiser: str = "gaussian",
                  frozen_regression: bool = False) -> Path:
    from cme_pipeline.data_preparation import validate_date, prepare_source
    from cme_pipeline.north_up_preprocessing import run_no_msb_north_up
    from cme_pipeline.live_detection import run_detection
    from scripts.recompute_e1_tracking import run as recompute_e1
    from scripts.run_frozen_e1_downstream import build_tracking

    date = validate_date(date)
    if method != "c0" or denoiser != "gaussian":
        raise ValueError("The validated integrated E1 path requires C0 with gaussian denoising")
    choices = sum(x is not None for x in (detection_dir, preprocessed_dir, source))
    if choices != 1:
        raise ValueError("Provide exactly one: --detection-dir, --preprocessed-dir, or --source")
    if frozen_regression and date != "20101212":
        raise ValueError("Frozen regression is only defined for 20101212")
    cfg = PipelineConfig.from_yaml(config_path)
    if cfg.tracking.method != "e1":
        raise ValueError("Integrated C0/E1 path requires tracking.method: e1")
    if not run_name or run_name == "baseline_rule_b":
        raise ValueError("Use a nonempty run name other than baseline_rule_b")
    cfg.repo_root = str(Path(cfg.repo_root).resolve())
    cfg.run_name = run_name
    cfg.date_str = int(date)
    cfg.cache_enabled = False
    repo = Path(cfg.repo_root)
    run_dir = (repo / cfg.output_base_dir / run_name).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing pipeline run: {run_dir}")
    if source is not None:
        if source not in {"local", "download"}:
            raise ValueError("source must be local or download")
        location = Path(acquisition_dir or repo / "runs" / f"acquisition_{date}_{run_name}")
        if location.resolve().exists():
            raise FileExistsError(f"Refusing to overwrite acquisition directory: {location}")
        manifest = prepare_source(
            date=date, source=source, product="d7", input_dir=input_dir,
            output_dir=location, warmup_days=warmup_days,
            phase_sec=phase_sec, frozen_reference_pairs=False,
        )
        preprocessed_dir = run_no_msb_north_up(manifest, mask_path=mask_path)
    if preprocessed_dir is not None:
        # Frozen selection is deliberately opt-in. General RD data are never trimmed.
        if frozen_regression:
            from scripts.build_frozen_replay import build
            preprocessed_dir = build(
                repo, Path(preprocessed_dir),
                run_dir.parent / f"{run_name}_frozen_input" / "preprocessed"
            )
        detection_dir = run_detection(
            preprocessed_dir, method=method, denoiser=denoiser, requested_date=date,
            output_dir=run_dir.parent / f"{run_name}_detection"
        )
    assert detection_dir is not None
    detection_dir = Path(detection_dir).resolve()
    metadata = json.loads((detection_dir / "detection_metadata.json").read_text())
    if str(metadata.get("requested_date")) != date:
        raise ValueError("Analysis date disagrees with detector metadata")
    if metadata.get("detector") != method or metadata.get("denoiser") != denoiser:
        raise ValueError("Detection method/denoiser disagree with selected arguments")
    if frozen_regression and not json.loads(
        (Path(metadata["input_dir"]) / "preprocessing_metadata.json").read_text()
    ).get("frozen_reference_pairs"):
        raise ValueError("Frozen regression requires explicit 60-pair fixture")

    e1_dir = run_dir.parent / f"{run_name}_e1_recomputed"
    if e1_dir.exists():
        raise FileExistsError(f"Refusing to overwrite tracking directory: {e1_dir}")
    outcome = recompute_e1(
        detection_dir, repo, e1_dir, compare_frozen=frozen_regression,
        historical_geometry=frozen_regression
    )
    if outcome != 0:
        raise RuntimeError("E1 recomputation failed Frozen reference comparison")
    if frozen_regression:
        # Verify all shared numeric and categorical Track State fields, not only IDs.
        from scripts.compare_frozen_e1_numeric import check as compare_e1_numeric
        frozen_track_csv = (repo / "validation_20101212/experiments/"
                            "11_d7_patch_first_tracking_feasibility/results/tables/"
                            "03_patch_e1_track_states.csv")
        comparison_result = compare_e1_numeric(
            frozen_track_csv, e1_dir / "03_patch_e1_track_states.csv",
            e1_dir / "numeric_regression", atol=1e-4, rtol=1e-6
        )
        if comparison_result:
            raise RuntimeError("Frozen E1 numerical regression failed")
    tracking = build_tracking(
        detection_dir, e1_dir / "03_patch_e1_track_states.csv", cfg,
        historical_geometry=frozen_regression
    )
    stages = resolve_stages(cfg)
    if date != "20101212":
        # The external catalog comparison is explicitly tied to 2010-12-12.
        stages = [stage for stage in stages if stage.name != "validation"]
    pipeline = Pipeline(cfg, stages=stages)
    result = pipeline.run(initial_input=tracking)
    if result["status"] != "SUCCESS":
        raise RuntimeError("Integrated pipeline failed")
    ctx = pipeline.context
    tracking.track_states_df.to_csv(
        ctx.stages_dir / "recomputed_e1_track_states.csv", index=False
    )
    seg = result["stage_outputs"]["segmentation"]
    sizes = seg.size_gate_output
    association = sizes.birth_gate_output.association_output
    steps = result["executed_stages"]
    report = {
        "status": "INTEGRATED_EXECUTED_NOT_FULLY_CERTIFIED",
        "date": date,
        "detector": method,
        "run_name": run_name,
        "frozen_regression": frozen_regression,
        "e1_numeric_regression": "PASS" if frozen_regression else "NOT_APPLICABLE",
        "tracking_input": str((e1_dir / "03_patch_e1_track_states.csv").resolve()),
        "detection_input": str(detection_dir),
        "input_pairs": int(json.loads((Path(metadata["input_dir"]) / "preprocessing_metadata.json")
                                     .read_text()).get("n_rd_pairs", -1)),
        "evaluation_pairs": len(tracking.manifest_df),
        "tracks": tracking.total_tracks,
        "candidate_families": association.candidate_family_count,
        "admitted_families": len(sizes.admitted_family_ids),
        "admitted_tracks": len(sizes.admitted_track_ids),
        "recovered_patches": sizes.total_recovered_patches,
        "union_mask_pixels": int(seg.final_pixel_segmentation_union.sum()),
        "stages": steps,
        "external_validation": "20101212 reference" if date == "20101212" else "SKIPPED_DATE_SPECIFIC",
        "caveats": [
            "General-date E1 uses WCS-derived solar center but historical RSUN_PX/occulter geometry scale.",
            "Only VALID analysis-date observations are passed to E1 tracking; gaps are recomputed.",
            "Scientific validation of other dates and arbitrary physical-unit calibration remains pending.",
        ],
    }
    if frozen_regression:
        reference = repo / "runs/baseline_rule_b/arrays/final_pixel_segmentation_union.npy"
        if reference.exists():
            frozen = np.load(reference, mmap_mode="r")
            mask = seg.final_pixel_segmentation_union
            report["local_frozen_mask_exact"] = bool(
                frozen.shape == mask.shape and np.array_equal(frozen, mask))
            if frozen.shape == mask.shape:
                report["local_frozen_mismatch_pixels"] = int(np.count_nonzero(frozen != mask))
    (ctx.run_dir / "integrated_run_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return ctx.run_dir
