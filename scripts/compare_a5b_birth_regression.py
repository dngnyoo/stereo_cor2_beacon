"""Read-only regression comparison: legacy birth-restricted vs unrestricted A5b.

Both runs must use exactly the same C3 detection directory and frame ordering.
Track IDs are not assumed stable: compare tracked coverage first, then match
Track IDs by maximum temporal ownership overlap (Hungarian assignment).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def compare_runs(legacy: Path, revised: Path, output: Path, *, birth_radius: float = 30.0):
    legacy, revised = legacy.resolve(), revised.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite regression output: {output}")
    metadata = [json.loads((root / "tracking_metadata.json").read_text()) for root in (legacy, revised)]
    a, b = metadata
    if a["method"] != "a5b" or b["method"] != "a5b":
        raise ValueError("Both tracking runs must be A5b")
    if Path(a["detection_dir"]).resolve() != Path(b["detection_dir"]).resolve():
        raise ValueError("Tracking runs must reference the exact same detection directory")
    if str(a["requested_date"]) != str(b["requested_date"]):
        raise ValueError("Tracking requested dates differ")

    arrays = [np.load(root / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
              for root in (legacy, revised)]
    old, new = arrays
    if old.shape != new.shape or old.ndim != 3 or old.shape[1:] != (64, 64):
        raise ValueError("Ownership masks must have identical (N,64,64) shapes")
    states = [pd.read_csv(root / "tables" / "live_tracking_states.csv")
              for root in (legacy, revised)]
    for df in states:
        if "birth_r_min_px" not in df.columns:
            raise ValueError("Tracking states lack birth_r_min_px; rerun with adapted output")
        if "state" not in df.columns or "pair_index" not in df.columns:
            raise ValueError("Tracking states missing expected columns")

    det = Path(a["detection_dir"]).resolve()
    detected = np.load(det / "patch_state_p04.npy", mmap_mode="r")
    summary = pd.read_csv(det / "frame_detection_summary.csv")
    if detected.shape != old.shape or len(summary) != len(old):
        raise ValueError("Detector output and tracking frames differ")
    if not np.array_equal(summary["pair_index"].astype(int).to_numpy(),
                          pd.read_csv(legacy / "tables" / "live_observation_manifest.csv")["pair_index"].astype(int).to_numpy()):
        raise ValueError("Legacy tracking pair order differs from detection")
    if not np.array_equal(summary["pair_index"].astype(int).to_numpy(),
                          pd.read_csv(revised / "tables" / "live_observation_manifest.csv")["pair_index"].astype(int).to_numpy()):
        raise ValueError("Revised tracking pair order differs from detection")
    if np.any((old > 0) & (detected != 1)) or np.any((new > 0) & (detected != 1)):
        raise ValueError("Tracking claims an undetected patch")

    old_ids = np.unique(old[old > 0]).astype(int)
    new_ids = np.unique(new[new > 0]).astype(int)
    # Sparse overlap through pairwise observed ownership; no dense N*IDs*IDs array.
    overlap = np.zeros((len(old_ids), len(new_ids)), dtype=np.int64)
    old_index = {tid: pos for pos, tid in enumerate(old_ids)}
    new_index = {tid: pos for pos, tid in enumerate(new_ids)}
    frame_rows = []
    for i in range(len(old)):
        old_frame = np.asarray(old[i])
        new_frame = np.asarray(new[i])
        jointly = (old_frame > 0) & (new_frame > 0)
        if np.any(jointly):
            ids, counts = np.unique(
                np.stack([old_frame[jointly], new_frame[jointly]], axis=1),
                axis=0, return_counts=True
            )
            for (oid, nid), count in zip(ids, counts):
                overlap[old_index[int(oid)], new_index[int(nid)]] += int(count)
        o = old_frame > 0
        n = new_frame > 0
        frame_rows.append({
            "pair_index": int(summary.iloc[i]["pair_index"]),
            "end_time": str(summary.iloc[i]["end_time"]),
            "old_owned": int(o.sum()),
            "new_owned": int(n.sum()),
            "common_owned": int((o & n).sum()),
            "lost_from_old": int((o & ~n).sum()),
            "newly_owned": int((~o & n).sum()),
            "same_raw_track_id": int(((old_frame == new_frame) & o & n).sum()),
        })

    def track_births(df):
        obs = df.loc[df["state"].eq("OBSERVED")].sort_values("pair_index")
        first = obs.groupby("track_id", sort=False).first()
        return {int(tid): float(row["birth_r_min_px"]) for tid, row in first.iterrows()}

    births_old = track_births(states[0])
    births_new = track_births(states[1])
    old_area = np.array([int(np.count_nonzero(old == tid)) for tid in old_ids], dtype=np.int64)
    new_area = np.array([int(np.count_nonzero(new == tid)) for tid in new_ids], dtype=np.int64)
    match = []
    if overlap.size:
        rr, cc = linear_sum_assignment(-overlap)
        for oi, ni in zip(rr, cc):
            common = int(overlap[oi, ni])
            if common == 0:
                continue
            union = int(old_area[oi] + new_area[ni] - common)
            match.append({
                "legacy_id": int(old_ids[oi]), "new_id": int(new_ids[ni]),
                "overlap_patches": common,
                "legacy_patches": int(old_area[oi]), "new_patches": int(new_area[ni]),
                "temporal_iou": common / union if union else np.nan,
                "legacy_birth_radius": births_old.get(int(old_ids[oi]), np.nan),
                "new_birth_radius": births_new.get(int(new_ids[ni]), np.nan),
            })

    frame_df = pd.DataFrame(frame_rows)
    match_df = pd.DataFrame(match).sort_values("overlap_patches", ascending=False) if match else pd.DataFrame(
        columns=["legacy_id", "new_id", "overlap_patches", "legacy_patches",
                 "new_patches", "temporal_iou", "legacy_birth_radius", "new_birth_radius"])
    new_birth_df = pd.DataFrame([
        {"new_id": tid, "birth_radius": births_new.get(int(tid), np.nan),
         "post_birth_gate_pass": bool(np.isfinite(births_new.get(int(tid), np.nan)) and births_new[int(tid)] <= birth_radius),
         "matched_legacy_id": next((r["legacy_id"] for r in match if r["new_id"] == int(tid)), None)}
        for tid in new_ids
    ])
    summary_out = {
        "legacy_run": str(legacy), "new_run": str(revised),
        "detection_dir": str(det),
        "frames": len(old), "legacy_tracks": len(old_ids), "new_tracks": len(new_ids),
        "ownership_bitwise_identical": bool(np.array_equal(old, new)),
        "frames_with_coverage_change": int(((frame_df.lost_from_old + frame_df.newly_owned) > 0).sum()),
        "total_old_owned": int(frame_df.old_owned.sum()),
        "total_new_owned": int(frame_df.new_owned.sum()),
        "old_coverage_lost": int(frame_df.lost_from_old.sum()),
        "new_coverage_gained": int(frame_df.newly_owned.sum()),
        "legacy_tracks_with_positive_match": len(match),
        "new_birth_radius_above_threshold": int((new_birth_df.birth_radius > birth_radius).sum()),
        "birth_radius_threshold": float(birth_radius),
        "caveat": "Matching is based on temporal patch overlap, not stable Track IDs. Post-Birth-Gate counts are diagnostic estimates, not executed downstream Gates.",
    }
    output.mkdir(parents=True)
    frame_df.to_csv(output / "frame_comparison.csv", index=False)
    match_df.to_csv(output / "track_matches.csv", index=False)
    new_birth_df.to_csv(output / "new_track_births.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary_out, indent=2) + "\n")
    print(json.dumps(summary_out, indent=2))
    return summary_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--birth-radius", type=float, default=30.0)
    args = parser.parse_args()
    compare_runs(args.legacy_run, args.new_run, args.output_dir, birth_radius=args.birth_radius)


if __name__ == "__main__":
    main()
