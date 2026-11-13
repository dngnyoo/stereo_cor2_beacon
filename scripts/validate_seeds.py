"""SEEDS-only, post-hoc event validation for an already completed COR2 run.

This tool never reads the prior candidate/score fields in the historical SEEDS
reference table, and never feeds catalogue measurements into CME detection.
Metrics are conditional on the selected, evaluable SEEDS observation window;
SEEDS is a comparison catalogue, not exhaustive ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cme_pipeline.utils.geometry import circular_sector_overlap

DEFAULT_SEEDS = (
    Path(__file__).resolve().parents[1]
    / "validation/frozen/20101212/experiments/"
    "04_d7_external_catalog_validation/results/04_event_scores/"
    "seeds_frame_scores.csv"
)

SEEDS_FIELDS = (
    "parent_event_id", "seeds_timestamp", "is_evaluable",
    "seeds_bpa", "seeds_width",
)
DETECTION_FIELDS = (
    "event_id", "pair_index", "timestamp", "cpa_deg",
    "angular_width_deg", "front_rsun",
)


def _require_columns(df: pd.DataFrame, names: tuple[str, ...], name: str) -> None:
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _read_seeds(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    # Accept both the legacy curated reference and the new official detail
    # tables. The latter has source-native column names and no observation-
    # availability decisions; temporal eligibility is handled downstream.
    detail_cols = {"DATE", "TIME", "BPA", "PASTART", "PAEND",
                   "HMLEAD", "parent_event_id"}
    if detail_cols.issubset(table.columns):
        pa_start = pd.to_numeric(table["PASTART"], errors="raise")
        pa_end = pd.to_numeric(table["PAEND"], errors="raise")
        # This dataset uses angular start/end sectors, not event-level WIDTH.
        widths = (pa_end - pa_start) % 360.0
        if (widths <= 0).any():
            raise ValueError("Invalid SEEDS PASTART/PAEND interval")
        table = pd.DataFrame({
            "parent_event_id": table["parent_event_id"],
            "seeds_timestamp": table["DATE"].astype(str) + " " + table["TIME"].astype(str),
            "is_evaluable": True,
            "seeds_bpa": table["BPA"],
            "seeds_width": widths,
            "seeds_lead": table["HMLEAD"],
        })
    _require_columns(table, SEEDS_FIELDS, "SEEDS reference")
    # This file also stores *historical candidate scores*. Explicitly discard
    # them: a comparison of the new pipeline must be computed independently.
    usable = table.loc[
        table["is_evaluable"].astype(str).str.lower().eq("true"),
        list(SEEDS_FIELDS) + (["seeds_lead"] if "seeds_lead" in table else []),
    ].copy()
    usable["timestamp_utc"] = pd.to_datetime(
        usable.pop("seeds_timestamp"), utc=True, errors="raise"
    )
    usable["seeds_bpa"] = pd.to_numeric(usable["seeds_bpa"], errors="raise")
    usable["seeds_width"] = pd.to_numeric(usable["seeds_width"], errors="raise")
    if "seeds_lead" in usable:
        usable["seeds_lead"] = pd.to_numeric(usable["seeds_lead"], errors="coerce")
    else:
        usable["seeds_lead"] = np.nan
    if usable.empty:
        raise ValueError("No evaluable SEEDS frames")
    if (usable["seeds_width"] <= 0).any() or (usable["seeds_width"] > 360).any():
        raise ValueError("Invalid SEEDS angular width")
    return usable.reset_index(drop=True)


def _read_detection_frames(path: Path) -> pd.DataFrame:
    frames = pd.read_csv(path)
    _require_columns(frames, DETECTION_FIELDS, "CME frame measurements")
    frames = frames.loc[:, list(DETECTION_FIELDS)].copy()
    frames["timestamp_utc"] = pd.to_datetime(
        frames.pop("timestamp"), utc=True, errors="raise"
    )
    for key in ("cpa_deg", "angular_width_deg", "front_rsun"):
        frames[key] = pd.to_numeric(frames[key], errors="raise")
    if (frames["angular_width_deg"] < 0).any() or (
        frames["angular_width_deg"] > 360
    ).any():
        raise ValueError("Invalid detected angular width")
    if frames[list(("cpa_deg", "angular_width_deg"))].isna().any().any():
        raise ValueError("Detected angular parameters contain NaNs")
    return frames.reset_index(drop=True)


def _nearest_observation(
    detections: pd.DataFrame, reference_time: pd.Timestamp, tolerance_sec: float
) -> pd.Series | None:
    if detections.empty:
        return None
    delta = (detections["timestamp_utc"] - reference_time).abs().dt.total_seconds()
    selected = int(delta.idxmin())
    if float(delta.loc[selected]) > tolerance_sec:
        return None
    return detections.loc[selected]


def validate_seeds(
    seeds: pd.DataFrame,
    detections: pd.DataFrame,
    *,
    tolerance_sec: float = 900,
    min_iou: float = 0.1,
    min_common_frames: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Match independent event tracks one-to-one using temporal angular IoU.

    For each SEEDS event and detected event, aggregate per-frame angular IoU,
    including zero for an unmatched SEEDS frame. A candidate needs a positive
    angular match on min_common_frames frames. Assignment is globally one-to-one.
    """
    if tolerance_sec < 0 or not 0 < min_iou <= 1 or min_common_frames < 1:
        raise ValueError("Invalid matching parameters")
    if seeds.empty:
        raise ValueError("No evaluable SEEDS frames")

    seeds_ids = sorted(seeds["parent_event_id"].astype(str).unique())
    all_ids = sorted(detections["event_id"].astype(str).unique())
    # Only events observed within a sampled SEEDS evaluation time are scoreable.
    ref_times = seeds["timestamp_utc"]
    eligible = []
    for det_id in all_ids:
        d = detections[detections["event_id"].astype(str).eq(det_id)]
        if any(
            (ref_times - t).abs().dt.total_seconds().min() <= tolerance_sec
            for t in d["timestamp_utc"]
        ):
            eligible.append(det_id)

    pair_rows = []
    comparison = {}
    for si, sid in enumerate(seeds_ids):
        s = seeds[seeds["parent_event_id"].astype(str).eq(sid)]
        for di, did in enumerate(eligible):
            d = detections[detections["event_id"].astype(str).eq(did)]
            frame_scores = []
            frame_radial_errors = []
            positive = 0
            for _, row in s.iterrows():
                match = _nearest_observation(d, row["timestamp_utc"], tolerance_sec)
                if match is None:
                    iou = 0.0
                    radial_error = np.nan
                else:
                    iou = float(circular_sector_overlap(
                        float(row["seeds_bpa"]), float(row["seeds_width"]),
                        float(match["cpa_deg"]), float(match["angular_width_deg"]),
                    )["iou"])
                    radial_error = (
                        abs(float(match["front_rsun"]) - float(row["seeds_lead"]))
                        if pd.notna(row["seeds_lead"]) else np.nan
                    )
                frame_scores.append(iou)
                if iou > 0:
                    positive += 1
                if np.isfinite(radial_error):
                    frame_radial_errors.append(radial_error)
            score = float(np.mean(frame_scores))
            qualifies = positive >= min_common_frames and score >= min_iou
            comparison[si, di] = score if qualifies else 0.0
            pair_rows.append({
                "seeds_event_id": sid,
                "detected_event_id": did,
                "n_seeds_evaluable_frames": len(s),
                "n_overlapping_frames": positive,
                "mean_angular_iou": score,
                "mean_radial_error_rsun": (
                    float(np.mean(frame_radial_errors)) if frame_radial_errors
                    else np.nan
                ),
                "eligible_match": qualifies,
                "assigned_match": False,
            })

    # Give unmatched events zero utility. No pair below threshold can consume a
    # detection: assignments with zero utility are ignored.
    score_matrix = np.zeros((len(seeds_ids), len(eligible)), dtype=float)
    for (si, di), score in comparison.items():
        score_matrix[si, di] = score
    matched = set()
    if score_matrix.size:
        row_idx, col_idx = linear_sum_assignment(-score_matrix)
        matched = {
            (seeds_ids[int(i)], eligible[int(j)])
            for i, j in zip(row_idx, col_idx)
            if score_matrix[i, j] > 0
        }
    pairs = pd.DataFrame(pair_rows, columns=[
        "seeds_event_id", "detected_event_id", "n_seeds_evaluable_frames",
        "n_overlapping_frames", "mean_angular_iou", "mean_radial_error_rsun",
        "eligible_match", "assigned_match",
    ])
    if not pairs.empty:
        pairs["assigned_match"] = [
            (a, b) in matched
            for a, b in zip(pairs["seeds_event_id"], pairs["detected_event_id"])
        ]

    matched_s = {sid for sid, _ in matched}
    matched_d = {did for _, did in matched}
    statuses = pd.DataFrame(
        [{"catalog": "SEEDS", "event_id": sid,
          "status": "MATCHED" if sid in matched_s else "UNMATCHED_REFERENCE"}
         for sid in seeds_ids]
        + [{"catalog": "DETECTION", "event_id": did,
            "status": "MATCHED" if did in matched_d else "UNMATCHED_DETECTION"}
           for did in eligible]
        + [{"catalog": "DETECTION", "event_id": did, "status": "OUTSIDE_EVALUATION"}
           for did in all_ids if did not in eligible],
        columns=["catalog", "event_id", "status"],
    )
    tp = len(matched)
    fp = len(eligible) - tp
    fn = len(seeds_ids) - tp
    summary = {
        "status": "SEEDS_COMPARISON_NOT_GROUND_TRUTH",
        "seeds_evaluable_events": len(seeds_ids),
        "seeds_evaluable_frames": len(seeds),
        "detected_events_total": len(all_ids),
        "detected_events_in_evaluation_window": len(eligible),
        "detected_events_outside_evaluation_window": len(all_ids) - len(eligible),
        "matched_events": tp,
        "unmatched_detected_events": fp,
        "unmatched_seeds_events": fn,
        "precision_conditional": tp / (tp + fp) if tp + fp else None,
        "recall_conditional": tp / (tp + fn) if tp + fn else None,
        "f1_conditional": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "max_time_delta_sec": tolerance_sec,
        "min_mean_angular_iou": min_iou,
        "min_overlapping_frames": min_common_frames,
        "matching_policy": "one-to-one, maximize mean frame angular IoU",
        "limitations": [
            "SEEDS is a reference catalog, not exhaustive ground truth.",
            "Precision is conditional on the SEEDS evaluable-time windows.",
            "Historical reference is curated for 2010-12-12 and does not establish general-date validity.",
            "Frame scores use angular sectors, not pixel-level mask IoU.",
        ],
    }
    return pairs, statuses, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds-csv", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-time-delta-sec", type=float, default=900)
    parser.add_argument("--min-mean-iou", type=float, default=0.1)
    parser.add_argument("--min-common-frames", type=int, default=1)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    frames_file = run_dir / "tables/cme_frame_measurements.csv"
    if not frames_file.is_file():
        raise FileNotFoundError(
            f"Independent frame catalog missing: {frames_file}"
        )
    reference_file = args.seeds_csv.expanduser().resolve()
    if not reference_file.is_file():
        raise FileNotFoundError(f"SEEDS reference missing: {reference_file}")
    seeds = _read_seeds(reference_file)
    detections = _read_detection_frames(frames_file)
    pairs, statuses, summary = validate_seeds(
        seeds, detections, tolerance_sec=args.max_time_delta_sec,
        min_iou=args.min_mean_iou,
        min_common_frames=args.min_common_frames,
    )
    output = args.output_dir or run_dir / "validation/seeds"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / "event_pair_scores.csv", index=False)
    statuses.to_csv(output / "event_status.csv", index=False)
    summary["reference_file"] = str(reference_file)
    summary["detection_file"] = str(frames_file)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
