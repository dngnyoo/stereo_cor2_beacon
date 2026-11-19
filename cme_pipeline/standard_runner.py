"""Conservative executable stages 01–05 from an explicitly supplied RD directory.

No automatic upstream discovery, no legacy-run overwrite, no implicit recovery.
Legacy computation functions are invoked unchanged; this adapter owns the new
single-run folder, snapshots, stage artifact contracts and gate-only execution.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.run_layout import initialize_run, complete_stage
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.core.standard_stages import stages_until, STAGE_BY_NAME
from cme_pipeline.live_detection import run_detection
from cme_pipeline.live_tracking_input import load_live_tracking_input
from cme_pipeline.p16_qc_tracking_input import run as apply_spatial_qc
from cme_pipeline.stages.tracking.registry import get_tracking_stage, get_tracking_spec
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output
from cme_pipeline.stages.gates.pre_tracking_component_size import filter_small_components
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage


PREPROCESSING_FILES = {
    "rd": "north_up_RD_DNs_per_second.npy",
    "valid_mask": "north_up_pair_valid_masks.npy",
    "manifest": "01_rd_manifest.csv",
    "geometry": "target_wcs_header.txt",
    "source_provenance": "preprocessing_metadata.json",
}


def _revision() -> str:
    p = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True,
                       check=False)
    return p.stdout.strip() if p.returncode == 0 else "UNKNOWN"


def _params(config: StandardConfig, stage: str) -> dict:
    return dict(config.data.get(stage, {}) or {})


def _stage_outputs(run: Path, names: dict[str, Path]) -> dict[str, str]:
    return {name: str(path.resolve().relative_to(run)) for name, path in names.items()}


def _context(run: Path, stage: str) -> PipelineContext:
    cfg = PipelineConfig()
    # Legacy operators write into context-specific tables/arrays/stages.
    # Confining each legacy context beneath the stage's data directory prevents
    # collisions and preserves one run folder without altering algorithms.
    cfg.repo_root = str(run.parent.parent)
    cfg.output_base_dir = str(run.parent.name)
    cfg.run_name = f"{run.name}/{STAGE_BY_NAME[stage].folder}/data"
    return PipelineContext(cfg)


def _preprocess_import(root: Path, source: Path) -> tuple[Path, dict[str, str]]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for fname in PREPROCESSING_FILES.values():
        if not (source / fname).is_file():
            raise FileNotFoundError(source / fname)
    dest = root / "01_preprocessing" / "data"
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copytree(source, dest)
    outputs = {key: str((dest / filename).relative_to(root))
               for key, filename in PREPROCESSING_FILES.items()}
    return dest, outputs


def execute_to_gates(
    config: StandardConfig, *, until: str, run_dir: str | Path,
    visualize: str | None = None,
) -> Path:
    """Execute up to stage 05 (new runs only); fail closed on unsupported options."""
    order = stages_until(until)
    if len(order) > 5:
        raise NotImplementedError("Integrated execution currently supports --until preprocessing..gates")
    if any(config.vis_mode(s.name.value, visualize) != "off" for s in order):
        raise NotImplementedError(
            "Visualization integration is not implemented; use --visualize off "
            "or post-run legacy visualization until stage-specific renderers are wired."
        )

    data = config.data
    info = data.get("input") or {}
    if not isinstance(info, dict):
        raise ValueError("input must be a mapping")
    # Explicit raw path takes priority. Without one, acquire fresh raw FITS
    # from the existing dual-archive downloader. The historical preprocessed
    # import route is retained for already completed experiments.
    allowed_input = {"raw_dir", "preprocessed_dir", "spatial_metrics_csv",
                     "warmup_days", "mask_path", "phase_sec", "raw_root"}
    if set(info) - allowed_input:
        raise ValueError(f"Unsupported standard input keys: {sorted(set(info) - allowed_input)}")
    if info.get("preprocessed_dir") and info.get("raw_dir"):
        raise ValueError("Choose either raw_dir or preprocessed_dir, not both")
    if info.get("preprocessed_dir") and set(info).intersection(
        {"warmup_days", "mask_path", "phase_sec", "raw_root"}
    ):
        raise ValueError("Raw preparation options cannot accompany preprocessed_dir")
    source = (Path(info["preprocessed_dir"]).expanduser().resolve()
              if info.get("preprocessed_dir") else None)
    raw_dir = (Path(info["raw_dir"]).expanduser().resolve()
               if info.get("raw_dir") else None)
    if raw_dir is not None and not raw_dir.is_dir():
        raise FileNotFoundError(f"Explicit raw_dir does not exist; refusing implicit download: {raw_dir}")
    metrics = (Path(info["spatial_metrics_csv"]).expanduser().resolve()
               if info.get("spatial_metrics_csv") else None)

    if config.method("preprocessing") != "dual_archive_rd":
        raise ValueError("Only preprocessing dual_archive_rd is supported (importing its saved RD)")
    if config.method("detection") != "d4":
        raise ValueError("Only D4 detection is supported in the first integrated baseline")
    if config.method("tracking") not in ("a5b", "a5b_n"):
        raise ValueError("Only A5b and experimental A5b-N tracking are supported")
    if config.method("association") != "identity":
        raise ValueError("Only identity association is supported")
    if config.method("gates") != "consecutive_size_qualified":
        raise ValueError("Only consecutive_size_qualified gates are supported")

    det = _params(config, "detection")
    qc = det.get("qc") or {}
    if det.get("score_mode", "abs") != "abs" or det.get("floor_mode", "outer_p95") != "outer_p95":
        raise ValueError("Baseline requires abs score and outer_p95")
    if det.get("threshold_method", "causal_t24") != "causal_t24":
        raise ValueError("Unsupported threshold implementation")
    if float(det.get("gaussian_sigma", 1.0)) != 1.0 or det.get("denoiser", "gaussian") != "gaussian":
        raise ValueError("Current detector only supports configured Gaussian sigma=1")
    # If the original date-specific metrics are not supplied, calculate them
    # from this Run's freshly saved D4 products with the existing v2 metric.
    if qc.get("normalized_weighted_tile_variance_min") is None and metrics:
        raise ValueError("QC metrics supplied but threshold not configured")

    tr = _params(config, "tracking")
    tr_params = tr.get("parameters") or {}
    tracking_method = config.method("tracking")
    spec = get_tracking_spec(tracking_method)
    unknown_tracking = set(tr_params) - spec.supported_parameters
    if unknown_tracking:
        raise ValueError(f"Unsupported {tracking_method} tracking parameters: {sorted(unknown_tracking)}")
    pre_size = tr.get("pre_size", {"connectivity": 8, "min_component_patches": 8})
    if not isinstance(pre_size, dict):
        raise ValueError("tracking.pre_size must be a mapping")
    if pre_size.get("connectivity", 8) != 8:
        raise ValueError("Only 8-connectivity pre-size is supported")
    if int(pre_size.get("min_component_patches", 8)) < 1:
        raise ValueError("Invalid pre-size patch count")
    if (_params(config, "gates").get("outward_split") or {}).get("enabled", False):
        raise ValueError("Outward split is not part of the baseline")

    date = str(data["experiment"]["date"])
    run = Path(run_dir).expanduser().resolve()
    if source is not None:
        if not source.is_dir():
            raise FileNotFoundError(source)
        source_meta = json.loads((source / "preprocessing_metadata.json").read_text())
        if str(source_meta.get("requested_date", date)) != date:
            raise ValueError("Preprocessed input date differs from experiment.date")
    if metrics is not None and not metrics.is_file():
        raise FileNotFoundError(metrics)
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    mode = ("explicit_preprocessed_import" if source is not None else
            "explicit_local_raw" if raw_dir is not None else "download_raw")
    provenance = {
        "explicit_preprocessed_dir": str(source) if source is not None else None,
        "explicit_raw_dir": str(raw_dir) if raw_dir is not None else None,
        "explicit_spatial_metrics_csv": str(metrics) if metrics else None,
        "mode": mode,
    }
    initialize_run(run, config.data, _revision(), provenance)
    if source is not None:
        pre, pre_outputs = _preprocess_import(run, source)
        pre_inputs = {"fits_sources": str(source)}
        implementation = "dual_archive_rd:explicit_import"
        pre_settings = {"mode": "import"}
    else:
        from cme_pipeline.data_preparation import prepare_source
        from cme_pipeline.north_up_preprocessing import run_no_msb_north_up
        preparation_root = run / "01_preprocessing" / "data" / "acquisition"
        warmup = int(info.get("warmup_days", 2))
        if not 0 <= warmup <= 7:
            raise ValueError("input.warmup_days must be between 0 and 7")
        phase = info.get("phase_sec")
        mask = info.get("mask_path")
        raw_root = info.get("raw_root")
        if raw_root is not None and raw_dir is not None:
            raise ValueError("input.raw_root is for download mode only")
        # Shared, repository-level RAW cache. Run-specific manifests and
        # processed arrays remain inside the immutable Run directory.
        # Explicit input.raw_root still takes priority.
        if raw_dir is None and raw_root is None:
            raw_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "cor2"
        manifest = prepare_source(
            date=date, source="local" if raw_dir is not None else "download",
            product="d7", input_dir=str(raw_dir) if raw_dir is not None else None,
            output_dir=preparation_root, warmup_days=warmup, phase_sec=phase,
            raw_root=raw_root,
        )
        pre = run_no_msb_north_up(manifest, mask_path=mask)
        pre_outputs = _stage_outputs(run, {
            key: pre / filename for key, filename in PREPROCESSING_FILES.items()
        })
        pre_inputs = {"fits_sources": str(raw_dir) if raw_dir is not None else "official_dual_archive",
                      "input_manifest": str(manifest.relative_to(run))}
        implementation = "dual_archive_rd:raw_local" if raw_dir is not None else "dual_archive_rd:raw_download"
        pre_settings = {"mode": mode, "warmup_days": warmup,
                        "mask_path": str(mask) if mask else None,
                        "phase_sec": phase,
                        "shared_raw_root": str(Path(raw_root).resolve()) if raw_root is not None else None}
        pre_outputs["acquisition_manifest"] = str(manifest.relative_to(run))
    complete_stage(run, "preprocessing", pre_inputs,
                   pre_outputs, implementation, pre_settings)
    if len(order) == 1:
        return run

    target = run / "02_detection" / "data"
    raw_dir = target / "raw"
    target.mkdir(parents=True, exist_ok=False)
    run_detection(pre, method="d4", denoiser="gaussian",
                  requested_date=date, output_dir=raw_dir,
                  floor_mode="outer_p95",
                  score_mode="abs",
                  t24_mad_multiplier=float(det.get("t24_mad_multiplier", 3.0)),
                  qc_max_active_fraction=float(qc.get("global_activation_threshold", .70)),
                  absolute_p04_floor=float(det.get("absolute_p04_floor", .5)),
                  outer_floor_inner_px=float(det.get("outer_floor_inner_px", 88)),
                  outer_floor_outer_px=float(det.get("outer_floor_outer_px", 116)))
    detector_dir = raw_dir
    computed_metrics = False
    if qc.get("normalized_weighted_tile_variance_min") is not None:
        if metrics is None:
            from cme_pipeline.p16_spatial_diagnostics import run as compute_p16_spatial_metrics
            metrics_dir = target / "p16_spatial_diagnostics"
            compute_p16_spatial_metrics(raw_dir, metrics_dir, date=date)
            metrics = metrics_dir / "p16_spatial_metrics.csv"
            computed_metrics = True
        detector_dir = apply_spatial_qc(
            raw_dir, metrics, target / "qc",
            date=date, cutoff=float(qc["normalized_weighted_tile_variance_min"])
        )
    det_outputs = _stage_outputs(run, {
        "patch_activation": detector_dir / "patch_state_p04.npy",
        "component_labels": detector_dir / "patch_component_labels.npy",
        "frame_qc": detector_dir / "frame_detection_summary.csv",
        "detection_scores": raw_dir / "frame_detection_summary.csv",
    })
    if metrics is not None:
        # The metrics source is part of Stage 02 provenance and must be
        # content-addressed by the Run layout when computed locally.
        if computed_metrics:
            det_outputs["p16_spatial_metrics"] = str(metrics.resolve().relative_to(run))
        else:
            from cme_pipeline.core.run_layout import _hash_file
            det_outputs["p16_external_metrics_sha256"] = str(
                (target / "p16_external_metrics_sha256.txt").relative_to(run))
            (target / "p16_external_metrics_sha256.txt").write_text(_hash_file(metrics) + "\n")
    # Record the fixed execution order of Detection substages, preserving
    # metrics and decision provenance separately from scientific parameters.
    from cme_pipeline.core.run_layout import _hash_file
    p16_active = qc.get("normalized_weighted_tile_variance_min") is not None
    substeps = [
        {"id": "02.1", "name": "d4_patch_scoring_and_hierarchy", "status": "COMPLETED"},
        {"id": "02.2", "name": "d4_global_activation_qc", "status": "COMPLETED",
         "threshold": float(qc.get("global_activation_threshold", .70))},
        {"id": "02.3", "name": "p16_spatial_metric", "status": "COMPLETED" if p16_active else "DISABLED",
         "method": "existing_p16_spatial_diagnostics_v2" if computed_metrics else
                   "explicit_external_metrics" if p16_active else None,
         "metrics_sha256": _hash_file(metrics) if p16_active else None},
        {"id": "02.4", "name": "p16_spatial_qc", "status": "COMPLETED" if p16_active else "DISABLED",
         "threshold": float(qc["normalized_weighted_tile_variance_min"]) if p16_active else None,
         "rule": "VALID and finite normalized variance < threshold"},
        {"id": "02.5", "name": "final_detection_output", "status": "COMPLETED",
         "usable_detection_sha256": _hash_file(detector_dir / "frame_detection_summary.csv")},
    ]
    substeps_path = target / "detection_substages.json"
    substeps_path.write_text(json.dumps({
        "stage": "detection", "date": date,
        "scientific_config": det, "substeps": substeps,
    }, indent=2) + "\n")
    det_outputs["substage_provenance"] = str(substeps_path.relative_to(run))
    if p16_active:
        decisions_file = target / "qc" / "normalized_qc_decisions.csv"
        det_outputs["p16_qc_decisions"] = str(decisions_file.relative_to(run))
    complete_stage(run, "detection", {"rd": pre_outputs["rd"], "valid_mask": pre_outputs["valid_mask"]},
                   det_outputs, "d4", det)
    if len(order) == 2:
        return run

    detected = load_live_tracking_input(detector_dir)
    if pre_size.get("enabled", True):
        detected = filter_small_components(
            detected, min_component_patches=int(pre_size.get("min_component_patches", 8))
        )
    cfg = PipelineConfig()
    cfg.date_str = int(date)
    cfg.detection.method = "d4"
    cfg.detection.denoiser = "gaussian"
    cfg.tracking.method = tracking_method
    cfg.tracking.parameters = dict(tr_params)
    cfg.association.method = "identity"
    gates = _params(config, "gates")
    cfg.birth_gate.max_birth_radius_px = float(gates.get("birth_max_radius_px", 30))
    persistence = gates.get("persistence") or {}
    if persistence.get("method", "consecutive_size_qualified") != "consecutive_size_qualified":
        raise ValueError("Unsupported gate persistence method")
    cfg.size_gate.min_detected_patches = int(persistence.get("min_detected_patches", 10))
    cfg.size_gate.min_observed_pairs = int(persistence.get("min_observed_pairs", 3))
    cfg.size_gate.size_qualified_persistence = True
    cfg.size_gate.consecutive_size_qualified = True
    track_ctx = _context(run, "tracking")
    tracker = get_tracking_stage(tracking_method)
    tracked = tracker.run(detected, cfg, track_ctx)
    if tracked.track_ownership_masks is None:
        raise RuntimeError(f"{tracking_method} returned no track ownership")
    np.save(track_ctx.arrays_dir / "track_ownership_masks.npy", tracked.track_ownership_masks)
    np.save(track_ctx.arrays_dir / "tracking_input_activation_p04.npy", detected.patch_activation_grid4)
    np.save(track_ctx.arrays_dir / "tracking_input_component_labels.npy", detected.raw_component_labels)
    tracked.track_states_df.to_csv(track_ctx.tables_dir / "live_tracking_states.csv", index=False)
    tracked.manifest_df.to_csv(track_ctx.tables_dir / "live_observation_manifest.csv", index=False)
    tracking_meta = {
        "requested_date": date, "method": tracking_method, "detection_dir": str(detector_dir),
        "total_tracks": int(tracked.total_tracks), "parameters": tracked.tracking_parameters,
        "pre_size": pre_size,
    }
    (track_ctx.run_dir / "tracking_metadata.json").write_text(json.dumps(tracking_meta, indent=2))
    track_artifacts = {
        "track_states": track_ctx.tables_dir / "live_tracking_states.csv",
        "track_ownership": track_ctx.arrays_dir / "track_ownership_masks.npy",
        "tracking_provenance": track_ctx.run_dir / "tracking_metadata.json",
    }
    if tracking_method == "a5b_n":
        diagnostic = getattr(tracked, "normalized_candidate_scores_df", None)
        if diagnostic is None:
            raise RuntimeError("A5b-N missing normalized candidate diagnostics")
        diagnostic_path = track_ctx.tables_dir / "normalized_candidate_scores.csv"
        diagnostic.to_csv(diagnostic_path, index=False)
        track_artifacts["normalized_candidate_scores"] = diagnostic_path
    track_outputs = _stage_outputs(run, track_artifacts)
    complete_stage(run, "tracking", {"patch_activation": det_outputs["patch_activation"]},
                   track_outputs, tracking_method, tr)
    if len(order) == 3:
        return run

    tracked = adapt_tracking_output(tracked, max_birth_radius_px=cfg.birth_gate.max_birth_radius_px)
    assoc_ctx = _context(run, "association")
    associated = IdentityAssociationStage().run(tracked, cfg, assoc_ctx)
    member_path = assoc_ctx.tables_dir / "02_full_day_e1_track_states.csv"
    edges_path = assoc_ctx.tables_dir / "03_full_day_parent_child_edges.csv"
    assoc_outputs = _stage_outputs(run, {
        "family_membership": member_path, "association_decisions": edges_path
    })
    complete_stage(run, "association", {"track_states": track_outputs["track_states"]},
                   assoc_outputs, "identity", _params(config, "association"))
    if len(order) == 4:
        return run

    gate_ctx = _context(run, "gates")
    birth = FamilyBirthRadiusGateStage().run(associated, cfg, gate_ctx)
    size = FamilySizeGateStage().run(birth, cfg, gate_ctx)
    admitted = gate_ctx.run_dir / "admitted_families.json"
    admitted.write_text(json.dumps({
        "family_ids": sorted(size.admitted_family_ids),
        "track_ids": sorted(size.admitted_track_ids)
    }, indent=2))
    qualified = gate_ctx.run_dir / "qualified_pairs.csv"
    size.family_summary_df[["family_id", "size_qualified_pairs",
                            "longest_consecutive_size_qualified"]].to_csv(qualified, index=False)
    gate_outputs = _stage_outputs(run, {
        "gate_decisions": gate_ctx.tables_dir / "04_full_day_family_summary.csv",
        "admitted_family_ids": admitted, "qualified_pair_ranges": qualified,
    })
    complete_stage(run, "gates", {"family_membership": assoc_outputs["family_membership"]},
                   gate_outputs, "birth+consecutive_size_qualified", gates)
    return run
