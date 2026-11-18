"""Visualize only Size/Persistence-Gate-admitted A5b families on signed RD.

Uses the saved gate family_summary.csv and original track ownership masks;
does not rerun detection, tracking, gates or Recovery.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.tracking_diagnostics import save_tracking_diagnostics


def visualize_admitted_families(gates_run: str | Path, *,
                                output_dir: str | Path | None = None) -> Path:
    gates = Path(gates_run).resolve()
    metadata = json.loads((gates / "gates_summary.json").read_text())
    tracking = Path(metadata["source_tracking_run"]).resolve()
    fam_path = gates / "stages" / "size_gate" / "family_summary.csv"
    if not fam_path.is_file():
        fam_path = gates / "tables" / "04_full_day_family_summary.csv"
    families = pd.read_csv(fam_path)
    required = {"family_id", "status", "member_track_ids", "n_active_pairs",
                "max_detected_patches"}
    if not required.issubset(families.columns):
        raise ValueError("Missing family summary columns: "
                         + str(sorted(required - set(families.columns))))
    admitted = families.loc[families.status.eq("ADMITTED")].copy()
    if len(admitted) != int(metadata["size_gate_admitted"]):
        raise ValueError("Admitted family count differs from saved gate summary")
    # A qualifying run does not imply all epochs of this Family are continuous.
    # Display only the longest contiguous size-qualified segment, without
    # altering the actual Tracking and Gate results on disk.
    strict_segments = bool(metadata.get("consecutive_size_qualified", False))
    allowed_family_pairs = {}
    if strict_segments:
        if "size_qualified_pairs" not in admitted.columns:
            raise ValueError("Missing size_qualified_pairs for strict visualization")
        for _, row in admitted.iterrows():
            fid = str(row.family_id)
            pairs = sorted(set(map(int, ast.literal_eval(str(row.size_qualified_pairs)))))
            segments = []
            current = []
            for pair in pairs:
                if current and pair != current[-1] + 1:
                    segments.append(current)
                    current = []
                current.append(pair)
            if current:
                segments.append(current)
            best = sorted(segments, key=lambda g: (-len(g), g[0]))[0] if segments else []
            if len(best) < int(metadata["min_observed_pairs"]):
                raise ValueError(f"Admitted Family {fid} lacks a valid consecutive segment")
            if int(row.longest_consecutive_size_qualified) != len(best):
                raise ValueError(f"Unexpected consecutive segment length for {fid}")
            allowed_family_pairs[fid] = set(best)
    track_to_family = {}
    for _, row in admitted.iterrows():
        ids = ast.literal_eval(str(row.member_track_ids))
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError(f"Invalid track ID list in {row.family_id}")
        for tid in ids:
            tid = int(tid)
            if tid in track_to_family:
                raise ValueError(f"Track {tid} belongs to more than one admitted family")
            track_to_family[tid] = str(row.family_id)
    target = (Path(output_dir).resolve() if output_dir else
              gates / "visualizations" / "admitted_families")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    save_tracking_diagnostics(
        tracking, date=str(metadata["date"]), output_dir=target,
        included_track_ids=set(track_to_family),
        track_family_ids=track_to_family,
        allowed_family_pairs=allowed_family_pairs if strict_segments else None)
    geometry = target / "track_geometry.csv"
    records = pd.read_csv(geometry) if geometry.stat().st_size else pd.DataFrame()
    if len(records):
        records["family_id"] = records.track_id.map(track_to_family)
        if records.family_id.isna().any():
            raise ValueError("A visualized track does not belong to admitted families")
        records.to_csv(target / "admitted_family_geometry.csv", index=False)
    else:
        pd.DataFrame(columns=["family_id","pair_index","track_id"]).to_csv(
            target / "admitted_family_geometry.csv", index=False)
    admitted.to_csv(target / "admitted_family_summary.csv",index=False)
    if strict_segments:
        pd.DataFrame([
            {"family_id": fid, "first_pair": min(pairs),
             "last_pair": max(pairs), "n_consecutive_pairs": len(pairs)}
            for fid, pairs in sorted(allowed_family_pairs.items())
        ]).to_csv(target / "displayed_consecutive_segments.csv", index=False)
    print(f"Gate-admitted visualization: {len(admitted)} families, "
          f"{len(track_to_family)} tracks; {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates-run", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    visualize_admitted_families(args.gates_run, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
