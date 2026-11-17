"""Pure observed-only A5b Birth/Size/Persistence gate evaluation (no Recovery).

Uses already saved A5b track ownership and OBSERVED states. Does not modify
Detection or Tracking. Identity Association: one Track = one Family.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


def evaluate_saved_gates(
    tracking_run: str | Path, *, output_dir: str | Path,
    max_birth_radius_px: float = 30.,
    min_detected_patches: int = 10,
    min_observed_pairs: int = 3,
) -> dict:
    run, dest = Path(tracking_run).resolve(), Path(output_dir).resolve()
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    if not np.isfinite(max_birth_radius_px) or max_birth_radius_px <= 0:
        raise ValueError("max_birth_radius_px must be finite and positive")
    if min_detected_patches < 1 or min_observed_pairs < 1:
        raise ValueError("Size and persistence thresholds must be >=1")

    meta = json.loads((run / "tracking_metadata.json").read_text())
    if meta.get("method") != "a5b":
        raise ValueError("Only saved A5b runs are supported")
    own = np.load(run / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    states = pd.read_csv(run / "tables" / "live_tracking_states.csv")
    manifest = pd.read_csv(run / "tables" / "live_observation_manifest.csv")
    detector_dir = Path(meta["detection_dir"]).resolve()
    detection_summary = pd.read_csv(detector_dir / "frame_detection_summary.csv")
    if (own.shape != (len(manifest), 64, 64) or
        len(detection_summary) != len(manifest) or
        not np.array_equal(manifest["pair_index"].to_numpy(), detection_summary["pair_index"].to_numpy())):
        raise ValueError("Saved A5b masks and detector manifest do not align")
    if "birth_r_min_px" not in states.columns:
        raise ValueError("A5b states require birth_r_min_px")
    observed = states.loc[states["state"].eq("OBSERVED")].copy()
    if observed.empty:
        raise ValueError("No OBSERVED tracking states")
    pair_to_frame = dict(zip(manifest["pair_index"].astype(int), range(len(manifest))))
    observed["pair_index"] = observed["pair_index"].astype(int)
    if not observed["pair_index"].isin(pair_to_frame.keys()).all():
        raise ValueError("Observed track state references unknown pair")
    rows = []
    raw_ids = set(int(x) for x in np.unique(own) if x > 0)
    observed_ids = set(observed["track_id"].astype(int))
    if raw_ids != observed_ids:
        raise ValueError("Observed IDs do not match owned Track IDs")
    for tid, group in observed.groupby("track_id"):
        tid = int(tid)
        group = group.sort_values("pair_index")
        birth_r = float(group.iloc[0]["birth_r_min_px"])
        birth_pass = bool(np.isfinite(birth_r) and birth_r <= max_birth_radius_px)
        pair_indices = sorted(set(group["pair_index"].astype(int)))
        n_observed = len(pair_indices)
        peak = 0
        peak_pair = None
        for pair in pair_indices:
            frame = pair_to_frame[pair]
            n = int(np.count_nonzero(own[frame] == tid))
            if n > peak:
                peak, peak_pair = n, pair
        size_pass = peak >= min_detected_patches
        duration_pass = n_observed >= min_observed_pairs
        passed = birth_pass and size_pass and duration_pass
        reason = ("ADMITTED" if passed else
                  "BIRTH_RADIUS" if not birth_pass else
                  "SIZE" if not size_pass else "PERSISTENCE")
        rows.append({
            "family_id": f"FAM_T{tid:02d}", "track_id": tid,
            "status": "ADMITTED" if passed else "REJECTED",
            "rejection_reason": reason,
            "birth_r_min_px": birth_r, "birth_pass": birth_pass,
            "max_detected_patches": peak, "max_detected_pair": peak_pair,
            "n_observed_pairs": n_observed, "size_pass": size_pass,
            "persistence_pass": duration_pass,
            "first_pair_index": pair_indices[0], "last_pair_index": pair_indices[-1],
        })
    family = pd.DataFrame(rows).sort_values("track_id")
    passed_ids = set(family.loc[family["status"] == "ADMITTED", "track_id"].astype(int))
    frame_rows = []
    for i, row in manifest.iterrows():
        owned = np.asarray(own[i])
        selected = np.isin(owned, list(passed_ids)) & (owned > 0)
        frame_rows.append({
            "pair_index": int(row["pair_index"]),
            "end_time": str(row["end_time"]),
            "owned_patches": int(np.count_nonzero(owned)),
            "admitted_patches": int(selected.sum()),
            "rejected_patches": int(np.count_nonzero(owned > 0) - selected.sum()),
        })
    per_frame = pd.DataFrame(frame_rows)
    summary = {
        "source_tracking_run": str(run),
        "source_detection_dir": str(detector_dir),
        "date": str(meta["requested_date"]),
        "max_birth_radius_px": float(max_birth_radius_px),
        "min_detected_patches": int(min_detected_patches),
        "min_observed_pairs": int(min_observed_pairs),
        "tracking_total": len(family),
        "birth_pass": int(family.birth_pass.sum()),
        "admitted_families": len(passed_ids),
        "admitted_observed_patches": int(per_frame.admitted_patches.sum()),
        "rejected_observed_patches": int(per_frame.rejected_patches.sum()),
        "recovery_executed": False,
        "segmentation_executed": False,
    }
    dest.mkdir(parents=True)
    family.to_csv(dest / "family_gate_decisions.csv", index=False)
    per_frame.to_csv(dest / "observed_frame_summary.csv", index=False)
    (dest / "gates_only_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def compare_gate_only_runs(baseline: str | Path, experiment: str | Path, *, output_dir: str | Path):
    base, exp, dest = Path(baseline).resolve(), Path(experiment).resolve(), Path(output_dir).resolve()
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    a = json.loads((base / "gates_only_summary.json").read_text())
    b = json.loads((exp / "gates_only_summary.json").read_text())
    if a["source_detection_dir"] != b["source_detection_dir"] or a["date"] != b["date"]:
        raise ValueError("Runs must share C3 detection and analysis date")
    fa = pd.read_csv(base / "observed_frame_summary.csv")
    fb = pd.read_csv(exp / "observed_frame_summary.csv")
    if not fa[["pair_index", "end_time"]].equals(fb[["pair_index", "end_time"]]):
        raise ValueError("Pair frames or ordering differ")
    combined = fa.merge(fb, on=["pair_index", "end_time"], suffixes=("_baseline", "_experiment"), validate="one_to_one")
    combined["admitted_patch_difference"] = combined["admitted_patches_experiment"] - combined["admitted_patches_baseline"]
    report = {"baseline": a, "experiment": b,
              "admitted_family_delta": b["admitted_families"] - a["admitted_families"],
              "admitted_observed_patch_delta": b["admitted_observed_patches"] - a["admitted_observed_patches"],
              "comparison": "Observed ownership and gates only; no Recovery, Segmentation, or per-pixel overlap"}
    dest.mkdir(parents=True)
    combined.to_csv(dest / "gate_only_frame_comparison.csv", index=False)
    (dest / "gate_only_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report
