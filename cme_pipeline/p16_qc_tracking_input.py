"""Create an isolated D4 detector-input variant for normalized P16 QC tracking.

Only evaluation-date VALID frames with normalized variance < cutoff are
converted into unavailable observations. Existing D4 files are untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run(detection_dir, metrics_csv, output_dir, *, cutoff=1.5, date=None):
    src = Path(detection_dir).resolve()
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Cutoff must be finite and positive")
    meta = json.loads((src/"detection_metadata.json").read_text())
    if meta.get("detector") != "d4":
        raise ValueError("D4 input required")
    day = str(date or meta["requested_date"])
    if day != str(meta["requested_date"]):
        raise ValueError("Requested date must match D4 evaluation date")
    pre = Path(meta["input_dir"]).resolve()
    manifest = pd.read_csv(pre/"01_rd_manifest.csv")
    summary = pd.read_csv(src/"frame_detection_summary.csv")
    metrics = pd.read_csv(metrics_csv)
    expected = {"pair_index","end_time","detection_status",
                "p16_normalized_weighted_tile_variance"}
    if not expected.issubset(metrics.columns):
        raise ValueError(f"Metrics missing required columns: {sorted(expected-set(metrics.columns))}")
    if metrics.pair_index.duplicated().any():
        raise ValueError("Duplicate pairs in metrics")
    if not (len(manifest) == len(summary)):
        raise ValueError("Manifest/summary length mismatch")
    if not (np.array_equal(manifest.pair_index.to_numpy(),summary.pair_index.to_numpy())
            and np.array_equal(pd.to_datetime(manifest.end_time,utc=True).to_numpy(),
                               pd.to_datetime(summary.end_time,utc=True).to_numpy())):
        raise ValueError("D4 and RD manifest timestamps or pair IDs mismatch")
    is_day = pd.to_datetime(manifest.end_time,utc=True).dt.strftime("%Y%m%d").eq(day).to_numpy()
    eval_pairs = set(manifest.loc[is_day,"pair_index"].astype(int))
    if set(metrics.pair_index.astype(int)) != eval_pairs:
        raise ValueError("Metrics must cover exactly all evaluation-date pairs")
    lookup = metrics.set_index("pair_index")
    flags = np.zeros(len(summary),dtype=bool)
    for i in np.flatnonzero(is_day):
        pair = int(manifest.iloc[i].pair_index)
        row = lookup.loc[pair]
        if (pd.to_datetime(row.end_time,utc=True) !=
                pd.to_datetime(manifest.iloc[i].end_time,utc=True)):
            raise ValueError(f"Metrics timestamps differ: pair {pair}")
        if str(row.detection_status) != str(summary.iloc[i].status):
            raise ValueError(f"Metrics status mismatch: pair {pair}")
        val = float(row.p16_normalized_weighted_tile_variance)
        flags[i] = bool(summary.iloc[i].status == "VALID" and np.isfinite(val)
                        and val < cutoff)
    original_states = np.load(src/"patch_state_p04.npy",mmap_mode="r")
    original_labels = np.load(src/"patch_component_labels.npy",mmap_mode="r")
    if not (original_states.shape == original_labels.shape ==
            (len(summary),64,64)):
        raise ValueError("Detector patch arrays misaligned")
    # Make independent arrays; never modify original memmaps.
    states = np.asarray(original_states).copy()
    labels = np.asarray(original_labels).copy()
    statuses = summary.status.astype(str).to_numpy()
    usable = summary.usable_detection.astype(str).str.lower().eq("true").to_numpy()
    if not np.array_equal(statuses == "VALID",usable):
        raise ValueError("Invalid original usable/status flags")
    states[flags] = -1
    labels[flags] = -1
    result_summary = summary.copy()
    result_summary.loc[flags,"status"]="SPATIAL_QC"
    result_summary.loc[flags,"usable_detection"]=False
    if "n_active_p04" in result_summary.columns:
        result_summary.loc[flags,"n_active_p04"]=np.nan
    meta["normalized_qc_extension"] = {
        "source_detection_dir":str(src),
        "source_metrics_csv":str(Path(metrics_csv).resolve()),
        "rule":"VALID and finite normalized P16 weighted tile variance < cutoff",
        "cutoff":float(cutoff),
        "evaluation_date_only":day,
        "new_rejections":int(flags.sum()),
        "new_rejected_pairs":manifest.loc[flags,"pair_index"].astype(int).tolist(),
        "no_changes_to_original":True,
    }
    meta["valid_requested_frames"] = int((result_summary.status.eq("VALID").to_numpy() & is_day).sum())
    out.mkdir(parents=True)
    np.save(out/"patch_state_p04.npy",states)
    np.save(out/"patch_component_labels.npy",labels)
    result_summary.to_csv(out/"frame_detection_summary.csv",index=False)
    (out/"detection_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    pd.DataFrame({
        "pair_index":manifest.loc[is_day,"pair_index"].to_numpy(),
        "end_time":manifest.loc[is_day,"end_time"].to_numpy(),
        "original_status":summary.loc[is_day,"status"].to_numpy(),
        "qc_status":result_summary.loc[is_day,"status"].to_numpy(),
        "normalized_variance":[float(lookup.loc[int(pair)].p16_normalized_weighted_tile_variance)
                               for pair in manifest.loc[is_day,"pair_index"]],
    }).to_csv(out/"normalized_qc_decisions.csv",index=False)
    from cme_pipeline.live_tracking_input import load_live_tracking_input
    # Verify the tracking loader accepts status and sentinel arrays.
    check = load_live_tracking_input(out)
    assert int(check.dataset_metadata["n_unavailable"]) == int((result_summary.status != "VALID").sum())
    print(f"Prepared {day}: {int(flags.sum())} new QC unavailable observations")
    print("Pair indices:",meta["normalized_qc_extension"]["new_rejected_pairs"])
    print("Tracking input:",out)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detection-dir",required=True)
    p.add_argument("--metrics",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--date")
    p.add_argument("--cutoff",type=float,default=1.5)
    a=p.parse_args()
    run(a.detection_dir,a.metrics,a.output_dir,date=a.date,cutoff=a.cutoff)


if __name__=="__main__":
    main()
