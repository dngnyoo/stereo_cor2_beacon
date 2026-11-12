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
    prepare_parser.add_argument("--detect", choices=("c0", "c3"), default=None,
                                help="Run C0 baseline or C3 default detector (requires --preprocess).")
    prepare_parser.add_argument("--denoiser", choices=("gaussian", "median"), default="gaussian")
    prepare_parser.add_argument("--warmup-days", type=int, default=None,
                                help="Preceding UTC dates; defaults to 2 when --detect is used.")
    prepare_parser.add_argument("--mask-path", default=None,
                                help="Override COR2 static occulter mask FITS.")
    prepare_parser.add_argument("--frozen-reference-pairs", action="store_true",
                                help="Select exactly the historical 60 paired FITS (20101212 only).")
    prepare_parser.add_argument("--phase-sec", default=None, type=float,
                                help="30-minute phase in seconds (d7 only; May 2024 default: 1434).")
    prepare_parser.add_argument("--output-dir", default=None)

    # Run an alternative detector on the same RD arrays without recalibrating.
    detect_parser = subparsers.add_parser(
        "detect-data", help="Replay C0 or C3 on existing North-up No-MSB RD arrays."
    )
    detect_parser.add_argument("--preprocessed-dir", required=True)
    detect_parser.add_argument("--method", choices=("c0", "c3"), default="c3")
    detect_parser.add_argument("--denoiser", choices=("gaussian", "median"), default="gaussian")
    detect_parser.add_argument("--date", default=None, help="Analysis YYYYMMDD; infer from RD metadata if omitted")

    track_parser = subparsers.add_parser(
        "track-data", help="Run availability-aware A3/A4/A5b tracking on live C0/C3 detection."
    )
    track_parser.add_argument("--detection-dir", required=True)
    track_parser.add_argument("--method", choices=("a3", "a4", "a5b"), default="a5b")
    track_parser.add_argument("--date", required=True, help="Analysis YYYYMMDD")
    track_parser.add_argument("--run-name", default=None)

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
    full_parser.add_argument("--phase-sec", type=float)
    full_parser.add_argument("--mask-path")
    full_parser.add_argument("--method", choices=("c0",), default="c0")
    full_parser.add_argument("--denoiser", choices=("gaussian",), default="gaussian")
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
    if args.command == "run-data":
        from cme_pipeline.live_full_pipeline import run_live_full
        run_live_full(
            config_path=args.config, date=args.date, run_name=args.run_name,
            detection_dir=args.detection_dir, preprocessed_dir=args.preprocessed_dir,
            source=args.source, input_dir=args.input_dir,
            acquisition_dir=args.acquisition_dir, warmup_days=args.warmup_days,
            phase_sec=args.phase_sec, mask_path=args.mask_path,
            method=args.method, denoiser=args.denoiser,
            frozen_regression=args.frozen_regression,
        )
        return

    if args.command == "prepare-data":
        from cme_pipeline.data_preparation import prepare_source
        from cme_pipeline.data_preparation import validate_date

        date = validate_date(args.date)
        target = args.output_dir or f"runs/data_prep_{date}_{args.product}"
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
        )
        print(f"FITS input manifest: {manifest}")
        if args.preprocess:
            from cme_pipeline.north_up_preprocessing import run_no_msb_north_up
            result = run_no_msb_north_up(manifest, mask_path=args.mask_path)
            print(f"North-up No-MSB RD arrays: {result}")
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
                                 requested_date=validate_date(args.date) if args.date else None)
        print(f"Detector outputs: {detected}")
        return

    if args.command == "track-data":
        from cme_pipeline.data_preparation import validate_date
        from cme_pipeline.live_tracking_input import load_live_tracking_input
        from cme_pipeline.core.context import PipelineContext
        from cme_pipeline.stages.tracking.registry import get_tracking_stage
        import pandas as pd

        date = validate_date(args.date)
        detected = load_live_tracking_input(args.detection_dir)
        if detected.dataset_metadata["detector"] not in ("c0", "c3"):
            raise ValueError("Live tracking requires a C0/C3 detection directory")
        if not detected.manifest_df["warmup_or_eval"].eq("EVALUATION").any():
            raise ValueError("No frames on requested analysis date")
        if detected.dataset_metadata["dataset_id"].split("_")[0] != date:
            raise ValueError("Requested date and detector run date disagree")
        cfg = PipelineConfig()
        cfg.date_str = int(date)
        cfg.tracking.method = args.method
        cfg.run_name = args.run_name or f"live_{date}_{detected.dataset_metadata['detector']}_{args.method}"
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
        import json
        (ctx.run_dir / "tracking_metadata.json").write_text(json.dumps({
            "requested_date": date,
            "method": args.method,
            "detector": detected.dataset_metadata["detector"],
            "detection_dir": str(Path(args.detection_dir).resolve()),
            "total_tracks": output.total_tracks,
            "n_unavailable": len(unavailable_pairs),
            "status": "TRACKING_READY_DOWNSTREAM_NOT_RUN",
        }, indent=2))
        print(f"Tracking outputs: {ctx.run_dir}")
        print("Status: TRACKING_READY_DOWNSTREAM_NOT_RUN")
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

