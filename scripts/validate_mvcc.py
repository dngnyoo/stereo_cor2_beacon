"""Post-hoc MVCC frame angular IoU on independently available RD observations."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scripts.validate_cactus import _prepare, sector, angular_iou


def validate_mvcc(reference, detections, observations, events, *, tolerance_sec=60., min_iou=0.1):
    """Align reference frames to valid RD end times, independently of detections.

    Each reference event uses each RD observation at most once (closest reference
    wins); absent detections on aligned observations score zero.
    """
    if not math.isfinite(tolerance_sec) or tolerance_sec < 0 or not 0 < min_iou <= 1:
        raise ValueError("Invalid matching parameters")
    ref = _prepare(reference, "catalog_event_id", "timestamp_utc", "MVCC")
    det = _prepare(detections, "event_id", "timestamp", "Detection")
    obs = observations[["pair_index", "end_time"]].copy()
    obs["end_time"] = pd.to_datetime(obs.end_time, utc=True, errors="raise")
    for table in (obs, det):
        values = pd.to_numeric(table.pair_index, errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError("Invalid pair index")
        table["pair_index"] = values.astype(int)
    if obs.empty or obs.end_time.isna().any() or obs.pair_index.duplicated().any() or obs.end_time.duplicated().any():
        raise ValueError("Invalid observation manifest")
    if det.duplicated(["event_id", "pair_index"]).any() or ref.duplicated(["catalog_event_id", "timestamp_utc"]).any():
        raise ValueError("Duplicate frame measurements")
    if not set(det.pair_index).issubset(obs.pair_index):
        raise ValueError("Detection pair missing from observation manifest")
    joined = det.merge(obs, on="pair_index", validate="many_to_one")
    if ((joined.timestamp - joined.end_time).abs().dt.total_seconds() > 1).any():
        raise ValueError("Detection timestamp disagrees with manifest")
    if events.catalog_event_id.isna().any() or events.catalog_event_id.duplicated().any():
        raise ValueError("Invalid catalog event IDs")
    if not set(ref.catalog_event_id).issubset(set(events.catalog_event_id.astype(str))):
        raise ValueError("Unknown reference event")
    if (pd.to_numeric(ref.angular_width_deg) <= 0).any():
        raise ValueError("Invalid reference angular width")
    align = []
    for row in ref.sort_values(["catalog_event_id", "timestamp_utc"]).itertuples(index=False):
        delta = (obs.end_time - row.timestamp_utc).abs().dt.total_seconds()
        idx = delta.idxmin()
        gap = float(delta.loc[idx])
        align.append(dict(catalog_event_id=row.catalog_event_id,
                          timestamp_utc=row.timestamp_utc, pair_index=int(obs.loc[idx,"pair_index"]),
                          beacon_timestamp=obs.loc[idx,"end_time"], delta_sec=gap,
                          cpa_deg=row.cpa_deg, angular_width_deg=row.angular_width_deg,
                          status="ALIGNED" if gap <= tolerance_sec else "NO_OBSERVATION"))
    alignment = pd.DataFrame(align, columns=["catalog_event_id","timestamp_utc","pair_index",
        "beacon_timestamp","delta_sec","cpa_deg","angular_width_deg","status"])
    aligned = alignment[alignment.status.eq("ALIGNED")].sort_values(
        ["delta_sec","timestamp_utc"])
    duplicate = aligned.duplicated(["catalog_event_id","pair_index"])
    alignment.loc[aligned.index[duplicate], "status"] = "DUPLICATE_OBSERVATION"
    aligned = alignment[alignment.status.eq("ALIGNED")]
    refs = sorted(aligned.catalog_event_id.unique())
    ids = sorted(det.loc[det.pair_index.isin(aligned.pair_index),"event_id"].unique())
    lookup = {(r.event_id,r.pair_index):r for r in det.itertuples(index=False)}
    weights = np.zeros((len(refs),len(ids)))
    pairs, scores = [], []
    for i, rid in enumerate(refs):
        rows = aligned[aligned.catalog_event_id.eq(rid)]
        for j, did in enumerate(ids):
            values = []
            present = 0
            for r in rows.itertuples(index=False):
                d = lookup.get((did,r.pair_index))
                value = angular_iou(sector(r.cpa_deg,r.angular_width_deg),
                                    sector(d.cpa_deg,d.angular_width_deg)) if d is not None else 0.
                present += d is not None
                values.append(value)
                scores.append(dict(catalog_event_id=rid,event_id=did,pair_index=r.pair_index,
                                   timestamp_utc=r.timestamp_utc,angular_iou=value,
                                   detection_present=d is not None))
            mean = float(np.mean(values))
            if mean >= min_iou:
                weights[i,j] = mean
            pairs.append(dict(catalog_event_id=rid,event_id=did,aligned_frames=len(rows),
                              detected_frames=present,mean_angular_iou=mean,matched=False))
    matches = set()
    if weights.size:
        rr,cc = linear_sum_assignment(-weights)
        matches = {(refs[i],ids[j]) for i,j in zip(rr,cc) if weights[i,j]>0}
    for row in pairs:
        row["matched"] = (row["catalog_event_id"],row["event_id"]) in matches
    matched_refs = {r for r,d in matches}
    matched_ids = {d for r,d in matches}
    statuses = []
    for r in events.catalog_event_id.astype(str):
        status = ("MATCHED" if r in matched_refs else "UNMATCHED_REFERENCE" if r in refs
                  else "NO_COMMON_OBSERVATION" if r in set(ref.catalog_event_id) else "NOT_EVALUABLE")
        statuses.append(dict(source="MVCC",event_id=r,status=status))
    for d in sorted(det.event_id.unique()):
        statuses.append(dict(source="detection",event_id=d,status="MATCHED" if d in matched_ids
                             else "UNMATCHED_DETECTION" if d in ids else "OUTSIDE_EVALUATION"))
    means = [r["mean_angular_iou"] for r in pairs if r["matched"]]
    summary = dict(catalog_events=len(events), reference_events_with_measurements=ref.catalog_event_id.nunique(),
                   evaluable_events=len(refs), reference_frames=len(ref), aligned_frames=len(aligned),
                   no_observation_frames=int(alignment.status.eq("NO_OBSERVATION").sum()),
                   duplicate_observation_frames=int(alignment.status.eq("DUPLICATE_OBSERVATION").sum()),
                   detected_events_in_evaluation_window=len(ids), matched_events=len(matches),
                   mean_matched_angular_iou=float(np.mean(means)) if means else None,
                   tolerance_sec=tolerance_sec,min_iou=min_iou,
                   note="Angular sector IoU; missing detection on available RD scores zero; unavailable observation excluded. Matched-only mean; not exhaustive ground truth.")
    return (pd.DataFrame(pairs,columns=["catalog_event_id","event_id","aligned_frames","detected_frames","mean_angular_iou","matched"]),
            pd.DataFrame(scores,columns=["catalog_event_id","event_id","pair_index","timestamp_utc","angular_iou","detection_present"]),
            alignment,pd.DataFrame(statuses,columns=["source","event_id","status"]),summary)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("frames-csv","mvcc-csv","events-csv","observations-csv","output-dir"):
        ap.add_argument("--"+arg,type=Path,required=True)
    ap.add_argument("--max-time-delta-sec",type=float,default=60.)
    ap.add_argument("--min-mean-iou",type=float,default=0.1)
    a=ap.parse_args(argv)
    paths=[a.mvcc_csv,a.frames_csv,a.observations_csv,a.events_csv]
    pairs,scores,alignment,statuses,summary=validate_mvcc(
        *[pd.read_csv(p) for p in paths],tolerance_sec=a.max_time_delta_sec,min_iou=a.min_mean_iou)
    summary["inputs"]={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for name,table in [("event_pair_scores",pairs),("matched_events",pairs.loc[pairs.matched.astype(bool)]),
                       ("frame_scores",scores),("observation_alignment",alignment),("event_status",statuses)]:
        table.to_csv(a.output_dir/(name+".csv"),index=False)
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
