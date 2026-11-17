"""Compare saved legacy/new A5b tracking after a pure post-tracking Birth Gate.

Identity Association means one Track per Family. This reproduces the family
anchor decision (first OBSERVED birth_r_min_px <= threshold), with no Size Gate
or Recovery. Uses saved ownership arrays and never changes upstream results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load(root: Path, max_birth_radius_px: float):
    meta = json.loads((root / "tracking_metadata.json").read_text())
    if meta.get("method") != "a5b":
        raise ValueError(f"Not A5b: {root}")
    own = np.load(root / "arrays" / "track_ownership_masks.npy", mmap_mode="r")
    df = pd.read_csv(root / "tables" / "live_tracking_states.csv")
    obs = df.loc[df["state"] == "OBSERVED"].sort_values(["track_id", "pair_index"])
    births = obs.groupby("track_id").first()
    if "birth_r_min_px" not in births:
        raise ValueError(f"Missing birth_r_min_px: {root}")
    born = births["birth_r_min_px"].astype(float)
    passed = sorted(int(tid) for tid, radius in born.items() if np.isfinite(radius) and radius <= max_birth_radius_px)
    if len(own) != len(pd.read_csv(meta["detection_dir"] + "/frame_detection_summary.csv")):
        raise ValueError("Tracking frame count differs from detection")
    return meta, own, born, passed


def compare(legacy: Path, revised: Path, output: Path, *, birth_radius: float = 30.0):
    legacy, revised, output = legacy.resolve(), revised.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not np.isfinite(birth_radius) or birth_radius <= 0:
        raise ValueError("Birth radius must be finite and positive")
    a, old, old_birth, old_pass = _load(legacy, birth_radius)
    b, new, new_birth, new_pass = _load(revised, birth_radius)
    if old.shape != new.shape or old.ndim != 3:
        raise ValueError("Ownership array shapes differ")
    if Path(a["detection_dir"]).resolve() != Path(b["detection_dir"]).resolve():
        raise ValueError("Both runs must share the exact same C3 detection")
    det_dir = Path(a["detection_dir"]).resolve()
    det = pd.read_csv(det_dir / "frame_detection_summary.csv")
    old_manifest = pd.read_csv(legacy / "tables" / "live_observation_manifest.csv")
    new_manifest = pd.read_csv(revised / "tables" / "live_observation_manifest.csv")
    for m in (old_manifest, new_manifest):
        if not np.array_equal(m["pair_index"].astype(int), det["pair_index"].astype(int)):
            raise ValueError("Pair order mismatch")
    old_pass_set = set(old_pass)
    new_pass_set = set(new_pass)
    frame_rows = []
    for i in range(len(det)):
        o = np.isin(old[i], old_pass) if old_pass else np.zeros(old.shape[1:], bool)
        n = np.isin(new[i], new_pass) if new_pass else np.zeros(new.shape[1:], bool)
        frame_rows.append({
            "pair_index": int(det.iloc[i]["pair_index"]),
            "end_time": str(det.iloc[i]["end_time"]),
            "legacy_pass_patches": int(o.sum()),
            "new_pass_patches": int(n.sum()),
            "preserved": int((o & n).sum()),
            "lost_after_birth_gate": int((o & ~n).sum()),
            "gained_after_birth_gate": int((~o & n).sum()),
            "gate_coverage_identical": bool(np.array_equal(o, n)),
        })
    f = pd.DataFrame(frame_rows)
    old_all = np.unique(old[old > 0])
    new_all = np.unique(new[new > 0])
    detail = pd.DataFrame([
        {"version": version, "track_id": int(tid), "birth_r_min_px": float(birth.get(int(tid), np.nan)),
         "passes_birth_gate": bool(int(tid) in passed),
         "owned_patches_before_gate": int(np.count_nonzero(own == tid))}
        for version, own, birth, passed, tids in [
            ("legacy", old, old_birth, old_pass_set, old_all),
            ("new", new, new_birth, new_pass_set, new_all),
        ] for tid in tids
    ])
    summary = {
        "legacy_run": str(legacy), "new_run": str(revised),
        "source_detection": str(det_dir), "birth_max_radius_px": birth_radius,
        "legacy_tracks": int(len(old_all)), "new_tracks": int(len(new_all)),
        "legacy_birth_pass": len(old_pass), "new_birth_pass": len(new_pass),
        "legacy_birth_rejected": int(len(old_all) - len(old_pass)),
        "new_birth_rejected": int(len(new_all) - len(new_pass)),
        "legacy_retained_patch_ownership": int(f.legacy_pass_patches.sum()),
        "new_retained_patch_ownership": int(f.new_pass_patches.sum()),
        "old_retained_patches_lost_in_new": int(f.lost_after_birth_gate.sum()),
        "new_retained_patches_gained_over_old": int(f.gained_after_birth_gate.sum()),
        "frames_with_post_birth_difference": int((~f.gate_coverage_identical).sum()),
        "post_birth_coverage_bitwise_identical": bool(f.gate_coverage_identical.all()),
        "note": "Identity one-to-one Family birth decision only; no Size/Persistence Gate or Recovery.",
    }
    output.mkdir(parents=True)
    f.to_csv(output / "birth_gate_frame_comparison.csv", index=False)
    detail.to_csv(output / "birth_gate_track_decisions.csv", index=False)
    (output / "birth_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nLargest lost-after-Birth-Gate frames:")
    print(f.sort_values("lost_after_birth_gate", ascending=False).head(10)[
        ["pair_index", "lost_after_birth_gate", "gained_after_birth_gate"]].to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--birth-radius", type=float, default=30.0)
    args = parser.parse_args()
    compare(args.legacy_run, args.new_run, args.output_dir, birth_radius=args.birth_radius)


if __name__ == "__main__":
    main()
