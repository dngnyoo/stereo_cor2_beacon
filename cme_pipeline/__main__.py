"""Command Line Interface for CME Pipeline.

Usage:
  python -m cme_pipeline run --config configs/baseline_rule_b.yaml
  python -m cme_pipeline run --config configs/baseline_rule_b.yaml --from-stage association
  python -m cme_pipeline status --config configs/baseline_rule_b.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cme_pipeline",
        description="Modular, reproducible CME detection and tracking pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # New isolated standard run. Legacy 'run' remains unchanged.
    standard_run_parser = subparsers.add_parser(
        "run-standard", help="Execute isolated standard stages 01-05; never overwrite old runs."
    )
    standard_run_parser.add_argument("--config", required=True)
    standard_run_parser.add_argument("--until", required=True, choices=(
        "preprocessing", "detection", "tracking", "association", "gates",
    ))
    standard_run_parser.add_argument("--input-run", default=None,
                                     help="Explicit completed Detection Run to branch from (no auto-search)")
    standard_run_parser.add_argument("--run-name", required=True,
                                     help="Unique name for one new folder under runs/")
    standard_run_parser.add_argument("--visualize", choices=("off", "diagnostic", "all"),
                                     default=None)

    # Read-only preflight for the new nine-stage runner (legacy run unchanged).
    plan_parser = subparsers.add_parser(
        "plan", help="Validate standard YAML and print stage plan; performs no execution."
    )
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--until", required=True, choices=(
        "preprocessing", "detection", "tracking", "association", "gates",
        "recovery", "segmentation", "catalog", "validation",
    ))
    plan_parser.add_argument("--visualize", choices=("off", "diagnostic", "all"),
                             default=None)

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Execute the pipeline or resume from a stage.")
    run_parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g. configs/baseline_rule_b.yaml).",
    )
    run_parser.add_argument(
        "--from-stage",
        type=str,
        default=None,
        help="Re-execute pipeline starting from specified stage (e.g. association, size_gate).",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of all stages, ignoring cache.",
    )

    # 'prepare-data' subcommand: input acquisition and provenance inventory.
    # Does NOT imply raw FITS have already been calibrated or detected.
    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Inventory local FITS or download one COR2 product for a selected date.",
    )
    prepare_parser.add_argument("--date", required=True, help="YYYYMMDD")
    prepare_parser.add_argument("--source", choices=("local", "download"), required=True)
    prepare_parser.add_argument(
        "--product", default="d7",
        choices=("d7", "science_raw", "science_seq", "beacon_raw", "beacon_seq"),
    )
    prepare_parser.add_argument("--input-dir", default=None)
    prepare_parser.add_argument("--preprocess", action="store_true",
                                help="Run No-MSB + North-up + 30-minute RD after input selection.")
    prepare_parser.add_argument("--detect", choices=("c0", "c3", "d4"), default=None,
                                help="Run C0 baseline or C3 default detector (requires --preprocess).")
    prepare_parser.add_argument("--denoiser", choices=("gaussian", "median"), default="gaussian")
    prepare_parser.add_argument("--warmup-days", type=int, default=None,
                                help="Preceding UTC dates; defaults to 2 when --detect is used.")
    prepare_parser.add_argument("--mask-path", default=None,
                                help="Override COR2 static occulter mask FITS.")
    prepare_parser.add_argument("--frozen-reference-pairs", action="store_true",
                                help="Select exactly the historical 60 paired FITS (20101212 only).")
    prepare_parser.add_argument("--phase-sec", default=None, type=float,
                                help="30-minute phase in seconds (d7 only; inferred from DATE-OBS by default).")
    prepare_parser.add_argument("--output-dir", default=None)
    prepare_parser.add_argument("--visualize-rd", action="store_true",
                                help="Save 3 sampled RD diagnostic PNGs after preprocessing.")
    prepare_parser.add_argument("--rd-all-frames", action="store_true",
                                help="With --visualize-rd, save all evaluation-date RD frames.")

    rd_parser = subparsers.add_parser(
        "visualize-rd", help="Render diagnostic RD PNGs from existing preprocessing (no recomputation)."
    )
    rd_parser.add_argument("--preprocessed-dir", required=True)
    rd_parser.add_argument("--date", default=None, help="UTC evaluation date YYYYMMDD")
    rd_parser.add_argument("--all-frames", action="store_true")
    rd_parser.add_argument("--max-frames", type=int, default=3)
    rd_parser.add_argument("--output-dir", default=None)
    rd_parser.add_argument("--overwrite", action="store_true",
                           help="Explicitly replace existing RD diagnostic PNGs.")

    # Run an alternative detector on the same RD arrays without recalibrating.
    detect_parser = subparsers.add_parser(
        "detect-data", help="Replay C0, C3 or D4 on existing North-up No-MSB RD arrays."
    )
    detect_parser.add_argument("--preprocessed-dir", required=True)
    detect_parser.add_argument("--method", choices=("c0", "c3", "d4"), default="c3")
    detect_parser.add_argument("--denoiser", choices=("gaussian", "median"), default="gaussian")
    detect_parser.add_argument("--date", default=None, help="Analysis YYYYMMDD; infer from RD metadata if omitted")
    detect_parser.add_argument("--floor-mode", choices=("frame_median", "outer_background", "outer_background_mad", "outer_p95"),
                               default="frame_median", help="C3 P04 floor strategy")
    detect_parser.add_argument("--score-mode", choices=("abs", "positive"), default="abs",
                               help="Patch P95 activity: abs(Z) baseline or max(Z,0) D4 experiment")
    detect_parser.add_argument("--absolute-p04-floor", type=float, default=None,
                               help="D4-only minimum P04 P95(abs(Z)) score; e.g. 0.5")
    detect_parser.add_argument("--t24-mad-multiplier", type=float, default=3.0,
                               help="T24 median + k * 1.4826 * MAD (default 3)")
    detect_parser.add_argument("--outer-floor-k", type=float, default=0.0,
                               help="MAD multiplier for outer_background_mad")
    detect_parser.add_argument("--outer-floor-inner-px", type=float, default=88.0)
    detect_parser.add_argument("--outer-floor-outer-px", type=float, default=116.0)
    detect_parser.add_argument("--output-dir", default=None,
                               help="Unique output path for side-by-side experiments")
    detect_parser.add_argument("--visualize", action=argparse.BooleanOptionalAction,
                               default=True,
                               help="Write all requested-day standard diagnostic PNGs (default: enabled)")
    detect_parser.add_argument("--visualize-all-frames", action="store_true",
                               help="With --visualize, write all evaluation-date frames")

    viz_detection = subparsers.add_parser(
        "visualize-detection",
        help="Render read-only RD, P04 score, and active-mask detection diagnostic PNGs."
    )
    viz_detection.add_argument("--detection-dir", required=True)
    viz_detection.add_argument("--date", default=None)
    viz_detection.add_argument("--all-frames", action="store_true")
    viz_detection.add_argument("--max-frames", type=int, default=6)
    viz_detection.add_argument("--output-dir", default=None)
    viz_detection.add_argument("--overwrite", action="store_true")

    viz_track = subparsers.add_parser(
        "visualize-tracking",
        help="Render standard A5b full-day tracking plots from a saved run."
    )
    viz_track.add_argument("--tracking-run", required=True)
    viz_track.add_argument("--date", default=None)
    viz_track.add_argument("--output-dir", default=None)

    gates_parser = subparsers.add_parser(
        "run-gates", aliases=["run-gates-data"],
        help="Run Identity, Birth and Size/Persistence Gates only, no Recovery."
    )
    gates_parser.add_argument("--tracking-run", required=True)
    gates_parser.add_argument("--run-name", required=True)
    gates_parser.add_argument("--birth-max-radius-px", type=float, default=30.0)
    gates_parser.add_argument("--min-detected-patches", type=int, default=5)
    gates_parser.add_argument("--min-observed-pairs", type=int, default=1,
                              help="Minimum unique OBSERVED frames for family admission")
    gates_parser.add_argument("--size-qualified-persistence", action="store_true",
                              help="Require >= min-detected-patches in >= min-observed-pairs distinct OBSERVED frames")
    gates_parser.add_argument("--consecutive-size-qualified", action="store_true",
                              help="Require size-qualified OBSERVED frames to form an uninterrupted consecutive run")

    recovery_parser = subparsers.add_parser(
        "run-recovery", help="Execute Recovery from saved Gate decisions; never rerun Gates."
    )
    recovery_parser.add_argument("--gates-run", required=True)
    recovery_parser.add_argument("--run-name", required=True)
    recovery_parser.add_argument("--recovery-inner-radius-px", required=True, type=float)

    compare_gates_parser = subparsers.add_parser(
        "compare-gates", help="Full-day baseline/outer P95 recovery and gates comparison."
    )
    compare_gates_parser.add_argument("--baseline-run", required=True)
    compare_gates_parser.add_argument("--experiment-run", required=True)
    compare_gates_parser.add_argument("--date", required=True)
    compare_gates_parser.add_argument("--output-dir", default=None)

    tc_parser = subparsers.add_parser(
        "compare-tracking",
        help="Render full-day side-by-side A5b track ownership over identical RD frames."
    )
    tc_parser.add_argument("--baseline-run", required=True)
    tc_parser.add_argument("--experiment-run", required=True)
    tc_parser.add_argument("--date", required=True)
    tc_parser.add_argument("--output-dir", default=None)

    compare_parser = subparsers.add_parser(
        "compare-detection", help="Full-day baseline vs experiment P04 outlined comparison."
    )
    compare_parser.add_argument("--baseline-dir", required=True)
    compare_parser.add_argument("--experiment-dir", required=True)
    compare_parser.add_argument("--date", required=True)
    compare_parser.add_argument("--output-dir", default=None)

    track_parser = subparsers.add_parser(
        "track-data", help="Run availability-aware A3/A4/A5b tracking on live C0/C3 detection."
    )
    track_parser.add_argument("--detection-dir", required=True)
    track_parser.add_argument("--config", default=None,
                              help="Optional YAML with tracking and pre-size settings")
    track_parser.add_argument("--method", choices=("a3", "a4", "a5b"), default="a5b")
    track_parser.add_argument("--date", required=True, help="Analysis YYYYMMDD")
    track_parser.add_argument("--run-name", default=None)
    track_parser.add_argument("--visualize", action=argparse.BooleanOptionalAction,
                              default=True,
                              help="Write standard full-day A5b PNGs (default: enabled)")

    full_parser = subparsers.add_parser(
        "run-data", help="Run independent C0/E1 from local/downloaded FITS, RD, or Detection."
    )
    full_parser.add_argument("--config", required=True)
    full_parser.add_argument("--date", required=True, help="YYYYMMDD")
    full_parser.add_argument("--run-name", required=True)
    full_input = full_parser.add_mutually_exclusive_group(required=True)
    full_input.add_argument("--source", choices=("local", "download"))
    full_input.add_argument("--preprocessed-dir")
    full_input.add_argument("--detection-dir")
    full_parser.add_argument("--input-dir", help="Required with --source local")
    full_parser.add_argument("--acquisition-dir")
    full_parser.add_argument("--warmup-days", type=int, default=2)
    full_parser.add_argument("--resume", action="store_true",
                             help="Resume an interrupted v2 run after verifying inputs and finished stages.")
    full_parser.add_argument("--phase-sec", type=float)
    full_parser.add_argument("--mask-path")
    full_parser.add_argument("--method", choices=("c0", "c3", "d4"), default=None,
                             help="Optional override check; detector is selected in YAML")
    full_parser.add_argument("--denoiser", choices=("gaussian", "median"), default=None,
                             help="Optional override check; denoiser is selected in YAML")
    full_parser.add_argument(
        "--frozen-regression", action="store_true",
        help="Opt in to 20101212 60-pair fixture and frozen E1 numerical checks."
    )

    # 'status' subcommand
    status_parser = subparsers.add_parser("status", help="Show pipeline stage cache status.")
    status_parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run-standard":
        from cme_pipeline.core.standard_config import StandardConfig
        from cme_pipeline.standard_runner import execute_to_gates
        from pathlib import Path
        from cme_pipeline.core.standard_stages import VIS_MODES
        if (args.run_name in (".", "..") or "/" in args.run_name
                or "\\\\" in args.run_name or args.run_name.startswith(".")):
            raise ValueError("--run-name must be a new simple folder name")
        cfg = StandardConfig.load(args.config)
        root = Path("runs") / args.run_name
        if args.input_run:
            import json
            parent_manifest = json.loads((Path(args.input_run) / "run_manifest.json").read_text())
            parent_stages = parent_manifest["stages"]
            if parent_stages.get("tracking") == "COMPLETED":
                from cme_pipeline.standard_tracking_parent import execute_from_tracking
                finished = execute_from_tracking(
                    cfg, input_run=args.input_run, until=args.until,
                    run_dir=root, visualize=args.visualize)
            elif parent_stages.get("detection") == "COMPLETED":
                from cme_pipeline.standard_input_run import execute_from_detection
                finished = execute_from_detection(
                    cfg, input_run=args.input_run, until=args.until,
                    run_dir=root, visualize=args.visualize)
            else:
                raise ValueError("Input Run has no supported completed Tracking or Detection stage")
        else:
            finished = execute_to_gates(cfg, until=args.until, run_dir=root,
                                       visualize=args.visualize)
        print(f"Standard run outputs: {finished}")
        return

    if args.command == "plan":
        from cme_pipeline.core.standard_config import StandardConfig
        from cme_pipeline.standard_plan import build_plan
        import json
        cfg = StandardConfig.load(args.config)
        print(json.dumps(build_plan(cfg, args.until, args.visualize), indent=2))
        return

    if args.command == "run-data":
        cfg = PipelineConfig.from_yaml(args.config)
        if cfg.tracking.method.lower() == "e1":
            # Preserve the independently recomputed C0/E1 regression path.
            from cme_pipeline.live_full_pipeline import run_live_full
            if args.method not in (None, "c0") or cfg.detection.method != "c0":
                raise ValueError("The E1 replay adapter supports C0 only")
            if args.denoiser not in (None, "gaussian") or cfg.detection.denoiser != "gaussian":
                raise ValueError("The E1 replay adapter supports Gaussian only")
            run_live_full(
                config_path=args.config, date=args.date, run_name=args.run_name,
                detection_dir=args.detection_dir, preprocessed_dir=args.preprocessed_dir,
                source=args.source, input_dir=args.input_dir,
                acquisition_dir=args.acquisition_dir, warmup_days=args.warmup_days,
                phase_sec=args.phase_sec, mask_path=args.mask_path,
                method="c0", denoiser="gaussian",
                frozen_regression=args.frozen_regression, resume=args.resume,
            )
        else:
            if args.frozen_regression:
                raise ValueError("--frozen-regression is reserved for the historical C0/E1 replay")
            from cme_pipeline.configured_run_data import run_configured_data
            run_configured_data(
                config_path=args.config, date=args.date, run_name=args.run_name,
                detection_dir=args.detection_dir, preprocessed_dir=args.preprocessed_dir,
                source=args.source, input_dir=args.input_dir,
                acquisition_dir=args.acquisition_dir, warmup_days=args.warmup_days,
                phase_sec=args.phase_sec, mask_path=args.mask_path,
                method=args.method, denoiser=args.denoiser, resume=args.resume,
            )
        return

    if args.command == "visualize-rd":
        from cme_pipeline.rd_diagnostics import save_rd_diagnostics
        paths = save_rd_diagnostics(
            args.preprocessed_dir, date=args.date, all_frames=args.all_frames,
            max_frames=args.max_frames, output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(f"RD diagnostic images: {len(paths)} saved to {paths[0].parent}")
        return

    if args.command == "prepare-data":
        from cme_pipeline.data_preparation import prepare_source
        from cme_pipeline.data_preparation import validate_date

        date = validate_date(args.date)
        target = args.output_dir or f"runs/data_prep_{date}_{args.product}"
        if args.visualize_rd and not args.preprocess:
            raise ValueError("--visualize-rd requires --preprocess")
        if args.rd_all_frames and not args.visualize_rd:
            raise ValueError("--rd-all-frames requires --visualize-rd")
        if args.detect and not args.preprocess:
            raise ValueError("--detect requires --preprocess")
        if args.preprocess and args.product != "d7":
            raise ValueError("--preprocess currently supports --product d7 only")
        warmup_days = args.warmup_days if args.warmup_days is not None else (2 if args.detect else 0)
        manifest = prepare_source(
            date=date,
            source=args.source,
            product=args.product,
            input_dir=args.input_dir,
            output_dir=target,
            phase_sec=args.phase_sec,
            warmup_days=warmup_days,
            frozen_reference_pairs=args.frozen_reference_pairs,
            raw_root=Path("data/raw/cor2") if args.source == "download" and args.product == "d7" else None,
        )
        print(f"FITS input manifest: {manifest}")
        if args.preprocess:
            from cme_pipeline.north_up_preprocessing import run_no_msb_north_up
            result = run_no_msb_north_up(manifest, mask_path=args.mask_path)
            print(f"North-up No-MSB RD arrays: {result}")
            if args.visualize_rd:
                from cme_pipeline.rd_diagnostics import save_rd_diagnostics
                images = save_rd_diagnostics(result, date=date,
                                             all_frames=args.rd_all_frames)
                print(f"RD diagnostic images: {len(images)} saved to {images[0].parent}")
            if args.detect:
                from cme_pipeline.live_detection import run_detection
                detected = run_detection(result, method=args.detect, denoiser=args.denoiser,
                                         requested_date=date)
                print(f"Detector outputs: {detected}")
                print("Status: DETECTION_READY_TRACKING_NOT_RUN")
            else:
                print("Status: RD_READY_DETECTION_NOT_RUN")
        else:
            print("Status: INVENTORIED_NOT_PREPROCESSED (use --preprocess for RD)")
        return

    if args.command == "detect-data":
        from cme_pipeline.live_detection import run_detection
        from cme_pipeline.data_preparation import validate_date
        detected = run_detection(args.preprocessed_dir, method=args.method,
                                 denoiser=args.denoiser,
                                 requested_date=validate_date(args.date) if args.date else None,
                                 floor_mode=args.floor_mode,
                                 t24_mad_multiplier=args.t24_mad_multiplier,
                                 outer_floor_k=args.outer_floor_k,
                                 outer_floor_inner_px=args.outer_floor_inner_px,
                                 outer_floor_outer_px=args.outer_floor_outer_px,
                                 absolute_p04_floor=args.absolute_p04_floor,
                                 score_mode=args.score_mode,
                                 output_dir=args.output_dir)
        print(f"Detector outputs: {detected}")
        if args.visualize:
            from cme_pipeline.detection_diagnostics import save_detection_diagnostics
            paths = save_detection_diagnostics(
                detected, date=args.date, all_frames=True
            )
            print(f"Detection diagnostic images: {len(paths)} saved to {paths[0].parent}")
        return

    if args.command == "visualize-tracking":
        from cme_pipeline.tracking_diagnostics import save_tracking_diagnostics
        out = save_tracking_diagnostics(
            args.tracking_run, date=args.date, output_dir=args.output_dir
        )
        print(f"Standard tracking PNGs: {out}")
        return

    if args.command == "visualize-detection":
        from cme_pipeline.detection_diagnostics import save_detection_diagnostics
        paths = save_detection_diagnostics(
            args.detection_dir, date=args.date,
            all_frames=args.all_frames, max_frames=args.max_frames,
            output_dir=args.output_dir, overwrite=args.overwrite,
        )
        print(f"Detection diagnostic images: {len(paths)} saved to {paths[0].parent}")
        return

    if args.command in ("run-gates", "run-gates-data"):
        from cme_pipeline.gate_experiment import run_saved_a5b_gates
        out = run_saved_a5b_gates(
            args.tracking_run, run_name=args.run_name,
            birth_max_radius_px=args.birth_max_radius_px,
            min_detected_patches=args.min_detected_patches,
            min_observed_pairs=args.min_observed_pairs,
            size_qualified_persistence=args.size_qualified_persistence,
            consecutive_size_qualified=args.consecutive_size_qualified,
        )
        print(f"Gate decisions (no Recovery): {out}")
        return

    if args.command == "run-recovery":
        from cme_pipeline.run_recovery_data import run_saved_recovery
        out = run_saved_recovery(
            args.gates_run, run_name=args.run_name,
            recovery_inner_radius_px=args.recovery_inner_radius_px,
        )
        print(f"Recovery outputs: {out}")
        return

    if args.command == "compare-gates":
        from cme_pipeline.gate_comparison import compare_gates
        out = compare_gates(
            args.baseline_run, args.experiment_run,
            date=args.date, output_dir=args.output_dir,
        )
        print(f"Gate comparison: {out}")
        return

    if args.command == "compare-tracking":
        from cme_pipeline.tracking_comparison import compare_tracking
        result = compare_tracking(
            args.baseline_run, args.experiment_run,
            date=args.date, output_dir=args.output_dir
        )
        print(f"Tracking comparison saved to {result}")
        return

    if args.command == "compare-detection":
        from cme_pipeline.detection_comparison import compare_detection
        result = compare_detection(
            args.baseline_dir, args.experiment_dir,
            date=args.date, output_dir=args.output_dir
        )
        print(f"Detection comparison saved to {result}")
        return

    if args.command == "track-data":
        from cme_pipeline.data_preparation import validate_date
        from cme_pipeline.live_tracking_input import load_live_tracking_input
        from cme_pipeline.core.context import PipelineContext
        from cme_pipeline.stages.tracking.registry import get_tracking_stage
        import pandas as pd

        date = validate_date(args.date)
        detected = load_live_tracking_input(args.detection_dir)
        if detected.dataset_metadata["detector"] not in ("c0", "c3", "d4"):
            raise ValueError("Live tracking requires a C0/C3 detection directory")
        if not detected.manifest_df["warmup_or_eval"].eq("EVALUATION").any():
            raise ValueError("No frames on requested analysis date")
        if detected.dataset_metadata["dataset_id"].split("_")[0] != date:
            raise ValueError("Requested date and detector run date disagree")
        cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
        cfg.date_str = int(date)
        if args.config and cfg.tracking.method != args.method:
            raise ValueError("YAML tracking.method must match --method")
        cfg.tracking.method = args.method
        if args.config and cfg.detection.method != detected.dataset_metadata["detector"]:
            raise ValueError("YAML detection.method must match saved detection")
        if args.config and cfg.detection.denoiser != detected.dataset_metadata["denoiser"]:
            raise ValueError("YAML detection.denoiser must match saved detection")
        cfg.run_name = args.run_name or f"live_{date}_{detected.dataset_metadata['detector']}_{args.method}"
        if cfg.pre_size_gate.enabled:
            if cfg.pre_size_gate.connectivity != 8:
                raise ValueError("Pre-size gate currently supports only 8-connectivity")
            from cme_pipeline.stages.gates.pre_tracking_component_size import filter_small_components
            detected = filter_small_components(
                detected, min_component_patches=cfg.pre_size_gate.min_component_patches
            )
        ctx = PipelineContext(cfg)
        tracker = get_tracking_stage(args.method)
        output = tracker.run(detected, cfg, ctx)
        unavailable_pairs = set(
            detected.manifest_df.loc[
                ~detected.manifest_df["usable_detection"].astype(bool),
                "pair_index"
            ].astype(int)
        )
        if not output.track_states_df.empty:
            reported = set(output.track_states_df["pair_index"].dropna().astype(int))
            if unavailable_pairs & reported:
                raise RuntimeError("Unavailable detector observations appeared as tracking states")
        output.manifest_df.to_csv(ctx.tables_dir / "live_observation_manifest.csv", index=False)
        output.track_states_df.to_csv(ctx.tables_dir / "live_tracking_states.csv", index=False)
        if output.track_ownership_masks is None:
            raise RuntimeError("A5b tracking must provide patch ownership masks for diagnostics")
        import numpy as np
        np.save(ctx.arrays_dir / "track_ownership_masks.npy", output.track_ownership_masks)
        np.save(ctx.arrays_dir / "tracking_input_activation_p04.npy", detected.patch_activation_grid4)
        np.save(ctx.arrays_dir / "tracking_input_component_labels.npy", detected.raw_component_labels)
        import json
        (ctx.run_dir / "tracking_metadata.json").write_text(json.dumps({
            "requested_date": date,
            "method": args.method,
            "detector": detected.dataset_metadata["detector"],
            "detection_dir": str(Path(args.detection_dir).resolve()),
            "total_tracks": output.total_tracks,
            "n_unavailable": len(unavailable_pairs),
            "pre_size_gate": detected.dataset_metadata.get("pre_tracking_component_gate", {"enabled": False}),
            "effective_tracking_parameters": output.tracking_parameters,
            "config_path": str(Path(args.config).resolve()) if args.config else None,
            "status": "TRACKING_READY_DOWNSTREAM_NOT_RUN",
        }, indent=2))
        print(f"Tracking outputs: {ctx.run_dir}")
        print("Status: TRACKING_READY_DOWNSTREAM_NOT_RUN")
        if args.visualize and args.method == "a5b":
            from cme_pipeline.tracking_diagnostics import save_tracking_diagnostics
            viz = save_tracking_diagnostics(ctx.run_dir, date=date)
            print(f"Standard tracking PNGs: {viz}")
        return

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig.from_yaml(config_path)

    if args.command == "run":
        if args.force:
            config.force_rerun = True
        pipeline = Pipeline(config)
        result = pipeline.run(from_stage=args.from_stage)
        if result["status"] != "SUCCESS":
            sys.exit(1)

    elif args.command == "status":
        pipeline = Pipeline(config)
        pipeline.status()


if __name__ == "__main__":
    main()

