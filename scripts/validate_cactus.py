"""Post-hoc CACTus event-level angular IoU from saved Stage 08 frame measurements.

Union the angular sectors of every observed frame of each detection, without
filling gaps between sectors. CACTus supplies a single event-level sector.
No radial, pixel, speed, or frame-level accuracy is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def sector(center, width):
    """Return non-wrapping intervals on [0, 360]; zero width has zero measure."""
    center, width = float(center), float(width)
    if not math.isfinite(center) or not math.isfinite(width) or not 0 <= width <= 360:
        raise ValueError("Invalid angular sector")
    if width == 0:
        return []
    if width == 360:
        return [(0.0, 360.0)]
    start = (center - width / 2) % 360
    end = start + width
    return [(start, end)] if end <= 360 else [(start, 360.0), (0.0, end - 360)]


def merge_intervals(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def angular_iou(a, b):
    a, b = merge_intervals(a), merge_intervals(b)
    inter = sum(max(0.0, min(hi, end) - max(lo, start))
                for lo, hi in a for start, end in b)
    union = sum(hi - lo for lo, hi in merge_intervals(a + b))
    return inter / union if union else 0.0


def _prepare(table, id_col, time_col, name):
    needed = {id_col, time_col, "cpa_deg", "angular_width_deg"}
    missing = needed - set(table.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    table = table.copy()
    if table[id_col].isna().any() or table[id_col].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} missing event IDs")
    table[id_col] = table[id_col].astype(str)
    table[time_col] = pd.to_datetime(table[time_col], utc=True, errors="raise")
    if table[time_col].isna().any():
        raise ValueError(f"{name} missing timestamps")
    for row in table.itertuples(index=False):
        sector(row.cpa_deg, row.angular_width_deg)
    return table


def validate_cactus(reference, frames, *, max_onset_delta_min=120.0, min_iou=0.0):
    if (not math.isfinite(max_onset_delta_min) or max_onset_delta_min < 0
            or not math.isfinite(min_iou) or not 0 <= min_iou <= 1):
        raise ValueError("Invalid matching thresholds")
    if "event_type" not in reference:
        raise ValueError("CACTus missing event_type")
    if not reference.event_type.isin(["CME", "Flow"]).all():
        raise ValueError("Unknown CACTus event type")
    refs = _prepare(reference.loc[reference.event_type.eq("CME")],
                    "catalog_event_id", "onset_utc", "CACTus")
    if refs.catalog_event_id.duplicated().any():
        raise ValueError("Duplicate CACTus event IDs")
    if (pd.to_numeric(refs.angular_width_deg) <= 0).any():
        raise ValueError("CACTus width must be positive")
    frames = _prepare(frames, "event_id", "timestamp", "Detection")
    detections = {}
    for ident, group in frames.groupby("event_id", sort=True):
        intervals = merge_intervals([
            interval for row in group.itertuples(index=False)
            for interval in sector(row.cpa_deg, row.angular_width_deg)
        ])
        detections[ident] = (group.timestamp.min(), intervals)
    refs = refs.sort_values("catalog_event_id").reset_index(drop=True)
    ids = list(detections)
    weights = np.zeros((len(refs), len(ids)), dtype=float)
    pair_rows = []
    for i, ref in enumerate(refs.itertuples(index=False)):
        ref_sector = sector(ref.cpa_deg, ref.angular_width_deg)
        for j, ident in enumerate(ids):
            onset, intervals = detections[ident]
            delta = abs((onset - ref.onset_utc).total_seconds()) / 60
            if delta > max_onset_delta_min:
                continue
            iou = angular_iou(intervals, ref_sector)
            if iou > 0 and iou >= min_iou:
                weights[i, j] = iou
            pair_rows.append({
                "catalog_event_id": ref.catalog_event_id, "event_id": ident,
                "onset_delta_min": delta, "angular_iou": iou,
                "matched": False,
            })
    matched = set()
    if weights.size:
        rr, cc = linear_sum_assignment(-weights)
        matched = {(refs.iloc[i].catalog_event_id, ids[j])
                   for i, j in zip(rr, cc) if weights[i, j] > 0}
    for row in pair_rows:
        row["matched"] = (row["catalog_event_id"], row["event_id"]) in matched
    pairs = pd.DataFrame(pair_rows, columns=[
        "catalog_event_id", "event_id", "onset_delta_min", "angular_iou", "matched"
    ])
    matched_refs = {r for r, _ in matched}
    matched_dets = {d for _, d in matched}
    statuses = pd.DataFrame([
        {"source": "CACTus", "event_id": r, "status": "matched" if r in matched_refs else "unmatched"}
        for r in refs.catalog_event_id
    ] + [
        {"source": "detection", "event_id": d, "status": "matched" if d in matched_dets else "unmatched"}
        for d in ids
    ], columns=["source", "event_id", "status"])
    values = [row["angular_iou"] for row in pair_rows if row["matched"]]
    summary = {
        "reference_events": len(refs), "detected_events": len(ids),
        "matched_events": len(matched),
        "unmatched_reference_events": len(refs) - len(matched),
        "unmatched_detected_events": len(ids) - len(matched),
        "mean_matched_angular_iou": float(np.mean(values)) if values else None,
        "max_onset_delta_min": max_onset_delta_min, "min_iou": min_iou,
        "metric": "event_angular_sector_union_iou",
        "note": "Matched-only mean; null if no match. Input scope is caller-selected; unmatched is not proof of false detection. No frame-level or pixel accuracy inferred.",
    }
    return pairs, statuses, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-csv", type=Path, required=True)
    ap.add_argument("--cactus-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-onset-delta-min", type=float, default=120)
    ap.add_argument("--min-iou", type=float, default=0)
    args = ap.parse_args(argv)
    ref = pd.read_csv(args.cactus_csv, dtype={"catalog_event_id": str})
    frames = pd.read_csv(args.frames_csv, dtype={"event_id": str})
    pairs, statuses, summary = validate_cactus(
        ref, frames, max_onset_delta_min=args.max_onset_delta_min, min_iou=args.min_iou)
    summary["inputs"] = {
        key: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in (("frames", args.frames_csv), ("cactus", args.cactus_csv))
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pairs.to_csv(args.output_dir / "event_pair_scores.csv", index=False)
    pairs.loc[pairs.matched.astype(bool)].to_csv(args.output_dir / "matched_events.csv", index=False)
    statuses.to_csv(args.output_dir / "event_status.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
