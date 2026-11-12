#!/usr/bin/env python3
"""Compare independently recomputed E1 track states to the historical frozen table.

Align on track/pair/state/component identity. This script only reads existing
tracking outputs and reference CSV; it never recomputes or overwrites tracking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["track_id", "pair_index", "state", "component_id", "component_uid"]


def check(reference: Path, recomputed: Path, output: Path, atol: float, rtol: float) -> int:
    ref = pd.read_csv(reference)
    new = pd.read_csv(recomputed)
    for side, df in (("Frozen", ref), ("Recomputed", new)):
        missing = sorted(set(KEY) - set(df.columns))
        if missing:
            raise ValueError(f"{side} missing identity keys: {missing}")
        if df.duplicated(KEY).any():
            raise ValueError(f"{side} contains duplicate state identities")
    joined = ref.merge(new, how="outer", on=KEY, suffixes=("_frozen", "_new"),
                       indicator=True, validate="one_to_one")
    missing_ref = int((joined["_merge"] == "right_only").sum())
    missing_new = int((joined["_merge"] == "left_only").sum())
    matched = joined.loc[joined["_merge"] == "both"].copy()
    common = sorted((set(ref.columns) & set(new.columns)) - set(KEY))
    columns = []
    divergent = []
    for col in common:
        left, right = matched[f"{col}_frozen"], matched[f"{col}_new"]
        # Only values known to be numeric on BOTH sides are treated as numeric.
        # All remaining columns (timestamps, enums, IDs, strings) compare literally.
        if pd.api.types.is_numeric_dtype(ref[col]) and pd.api.types.is_numeric_dtype(new[col]):
            a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
            same = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
            finite_pair = np.isfinite(a) & np.isfinite(b)
            abs_delta = np.abs(a[finite_pair] - b[finite_pair])
            metric = {
                "column": col, "kind": "numeric", "mismatched_rows": int((~same).sum()),
                "max_abs_delta": float(abs_delta.max()) if abs_delta.size else None,
                "mean_abs_delta": float(abs_delta.mean()) if abs_delta.size else None,
            }
        else:
            eq = left.fillna("<MISSING>").astype(str).eq(right.fillna("<MISSING>").astype(str)).to_numpy()
            same = eq
            metric = {"column": col, "kind": "categorical", "mismatched_rows": int((~same).sum())}
        columns.append(metric)
        if not np.all(same):
            bad = matched.loc[~same, KEY].copy()
            bad["column"] = col
            bad["frozen_value"] = left.loc[~same].astype(str).values
            bad["recomputed_value"] = right.loc[~same].astype(str).values
            divergent.append(bad)
    all_differences = (pd.concat(divergent, ignore_index=True) if divergent else
                       pd.DataFrame(columns=KEY + ["column", "frozen_value", "recomputed_value"]))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(output / "numeric_column_summary.csv", index=False)
    all_differences.to_csv(output / "numeric_state_mismatches.csv", index=False)
    joined.loc[joined["_merge"] != "both", KEY + ["_merge"]].to_csv(
        output / "unmatched_state_keys.csv", index=False)
    report = {
        "frozen_states": len(ref), "recomputed_states": len(new),
        "matched_state_keys": len(matched),
        "unmatched_frozen_only": missing_new,
        "unmatched_recomputed_only": missing_ref,
        "columns_compared": len(common),
        "numeric_tolerance_atol": atol, "numeric_tolerance_rtol": rtol,
        "columns_with_mismatches": [r["column"] for r in columns if r["mismatched_rows"]],
        "total_column_value_mismatches": int(sum(r["mismatched_rows"] for r in columns)),
        "status": "PASS" if not missing_ref and not missing_new and not len(all_differences)
                  and set(ref.columns) == set(new.columns) else "MISMATCH",
        "missing_columns_in_recomputed": sorted(set(ref.columns) - set(new.columns)),
        "extra_columns_in_recomputed": sorted(set(new.columns) - set(ref.columns)),
        "note": "Track-state CSV columns only; equality of downstream products is not yet tested.",
    }
    (output / "numeric_comparison_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--recomputed-dir", type=Path, default=Path(
        "runs/dual_archive_20101212/frozen_replay/tracking_e1_recomputed"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "runs/dual_archive_20101212/frozen_replay/comparison/tracking_numeric"))
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-6)
    args = p.parse_args()
    reference = args.repo / ("validation_20101212/experiments/"
                             "11_d7_patch_first_tracking_feasibility/results/tables/"
                             "03_patch_e1_track_states.csv")
    raise SystemExit(check(reference, args.recomputed_dir / "03_patch_e1_track_states.csv",
                           args.output_dir, args.atol, args.rtol))
