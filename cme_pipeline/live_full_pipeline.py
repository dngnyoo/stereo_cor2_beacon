"""Single-entry COR2 C0/E1 integration: FITS or RD or detection through CME products.

No historical Tracking CSV is used as an input. The original benchmark
remains available through the legacy 'run' CLI and explicit --frozen-regression.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline, resolve_stages

def select_live_stages(stages, *, frozen_regression: bool):
    """Historical external validator requires the explicit frozen 32-frame sequence."""
    if frozen_regression:
        return list(stages)
    return [stage for stage in stages if stage.name != "validation"]


def check_run_request(run_dir: Path, request: dict, *, resume: bool) -> bool:
    """Create a fresh run or verify an interrupted run has identical inputs.

    Returns True when a known complete run can be safely reused.
    """
    state_file = run_dir / "run_request.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite existing pipeline run: {run_dir}")
        if not state_file.is_file():
            raise ValueError(f"Cannot resume run without request provenance: {state_file}")
        stored = json.loads(state_file.read_text(encoding="utf-8"))
        if stored != request:
            raise ValueError(
                "Resume arguments/config/mask changed; use a new --run-name"
            )
        return (run_dir / "integrated_run_summary.json").is_file()
    if resume:
        raise FileNotFoundError(f"Cannot resume nonexistent run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    state_file.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return False


def stage_ready(path: Path, required: tuple[str, ...]) -> bool:
    """Reuse only a completed stage; reject ambiguous partial artifacts."""
    if not path.exists():
        return False
    if all((path / name).is_file() for name in required):
        return True
    raise RuntimeError(
        f"Incomplete stage directory {path}; inspect or archive it "
        "before resuming. Raw FITS remain safe and reusable."
    )


def run_live_full(*, config_path: str | Path, date: str,
                  run_name: str, detection_dir: str | Path | None = None,
                  preprocessed_dir: str | Path | None = None,
                  source: str | None = None, input_dir: str | Path | None = None,
                  acquisition_dir: str | Path | None = None,
                  warmup_days: int = 2, phase_sec: float | None = None,
                  mask_path: str | Path | None = None,
                  method: str = "c0", denoiser: str = "gaussian",
                  frozen_regression: bool = False,
                  resume: bool = False) -> Path:
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
    calibrated_sha = None
    if source is not None:
        # Validate calibration before expensive acquisition and include the
        # actual mask contents in the resume identity.
        from cme_pipeline.north_up_preprocessing import load_cor2_mask
        preflight_mask = mask_path
        if frozen_regression and preflight_mask is None:
            preflight_mask = repo / "Calibration" / "cor2_mask.fts"
        _, _, calibrated_sha = load_cor2_mask(preflight_mask)
    request = {
        "schema_version": 1,
        "date": date, "run_name": run_name,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "source": source,
        "input_dir": str(Path(input_dir).resolve()) if input_dir else None,
        "preprocessed_dir": str(Path(preprocessed_dir).resolve()) if preprocessed_dir else None,
        "detection_dir": str(Path(detection_dir).resolve()) if detection_dir else None,
        "acquisition_dir": str(Path(acquisition_dir).resolve()) if acquisition_dir else None,
        "warmup_days": warmup_days,
        "phase_sec": phase_sec, "mask_sha256": calibrated_sha,
        "mask_path": str(Path(mask_path).resolve()) if mask_path else None,
        "method": method, "denoiser": denoiser,
        "frozen_regression": frozen_regression,
    }
    if check_run_request(run_dir, request, resume=resume):
        print(f"Verified complete run, no work needed: {run_dir}")
        return run_dir
    if source is not None:
        if source not in {"local", "download"}:
            raise ValueError("source must be local or download")
        location = Path(acquisition_dir).resolve() if acquisition_dir else run_dir / "acquisition"
        manifest = location / "input_manifest.json"
        if not (resume and manifest.is_file()):
            if location.exists() and not resume:
                raise FileExistsError(
                    f"Refusing to overwrite acquisition directory: {location}"
                )
            manifest = prepare_source(
                date=date, source=source, product="d7", input_dir=input_dir,
                output_dir=location, warmup_days=warmup_days,
                phase_sec=phase_sec, frozen_reference_pairs=False,
                raw_root=repo / "data" / "raw" / "cor2" if source == "download" else None,
            )
        # Frozen regression retains the historic mask, not the COR2-A default.
        selected_mask = mask_path
        if frozen_regression and selected_mask is None:
            selected_mask = repo / "Calibration" / "cor2_mask.fts"
        candidate = location / "preprocessed"
        if resume and stage_ready(candidate, (
                "preprocessing_metadata.json", "01_rd_manifest.csv",
                "north_up_RD_DNs_per_second.npy", "north_up_pair_valid_masks.npy",
                "target_wcs_header.txt")):
            preprocessed_dir = candidate
        else:
            preprocessed_dir = run_no_msb_north_up(manifest, mask_path=selected_mask)
    if preprocessed_dir is not None:
        # Frozen selection is deliberately opt-in. General RD data are never trimmed.
        if frozen_regression:
            from scripts.build_frozen_replay import build
            candidate = run_dir / "frozen_input" / "preprocessed"
            if resume and stage_ready(candidate, (
                    "preprocessing_metadata.json", "01_rd_manifest.csv",
                    "north_up_RD_DNs_per_second.npy",
                    "north_up_pair_valid_masks.npy", "target_wcs_header.txt")):
                preprocessed_dir = candidate
            else:
                preprocessed_dir = build(repo, Path(preprocessed_dir), candidate)
        candidate = run_dir / "detection"
        if resume and stage_ready(candidate, (
                "detection_metadata.json", "frame_detection_summary.csv",
                "patch_state_p04.npy", "patch_component_labels.npy")):
            detection_dir = candidate
        else:
            detection_dir = run_detection(
                preprocessed_dir, method=method, denoiser=denoiser, requested_date=date,
                output_dir=candidate
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

    e1_dir = run_dir / "tracking"
    if not (resume and stage_ready(e1_dir, (
            "summary.json", "03_patch_e1_track_states.csv",
            "01_patch_component_geometries.csv",
            "02_patch_e1_candidate_evaluations.csv"))):
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
    stages = select_live_stages(resolve_stages(cfg), frozen_regression=frozen_regression)
    pipeline = Pipeline(cfg, stages=stages)
    result = pipeline.run(initial_input=tracking)
    if result["status"] != "SUCCESS":
        raise RuntimeError("Integrated pipeline failed")
    ctx = pipeline.context
    # E1 states already persist in tracking/03_patch_e1_track_states.csv;
    # do not write a second identical stages/ copy.
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
        "raw_data_root": str(repo / "data" / "raw" / "cor2") if source == "download" else None,
        "run_layout_version": 2,
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
        "external_validation": (
            "20101212_FROZEN_REFERENCE" if frozen_regression
            else "SKIPPED_NON_FROZEN_OBSERVATION_SEQUENCE"
        ),
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
